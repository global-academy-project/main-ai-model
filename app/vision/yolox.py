"""YOLOX-Nano (Apache-2.0) ONNX 사람 검출. 공식 COCO 가중치는 사람만 센다.
안전모는 따로 학습한 YOLOX-Nano (클래스: 안전모 쓴 머리, 맨머리, 11/13)를 같은 방식으로 읽는다 (class_boxes).

흐름: 사진 바이트 → EXIF 회전 적용(브라우저가 보여 주는 방향) → 416 정사각형에 비율 유지로 넣기(왼쪽 위 붙임, 회색 114)
      → ONNX 실행 → 격자 풀기(YOLOX demo_postprocess) → 사람 점수 = 물체 점수 × 사람 클래스 점수 → 기준 이상만 → NMS
      → 원본 사진 크기에 대한 0~1 비율 박스.
- onnxruntime 은 처음 쓸 때 읽는다 (설치 안 돼 있어도 앱은 뜬다).
- 모델 파일은 로컬 경로만 읽는다 (내려받지 않음, 폐쇄망). 파일 준비는 scripts/download_models.py.
- 입력은 YOLOX 데모와 같게 BGR, 0~255 그대로 (정규화 없음).
"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "YOLOX-Nano"
PERSON = 0          # COCO 클래스 번호
HELMET_LABELS = ("helmet", "no_helmet")   # 안전모 모델 클래스 순서 (학습 COCO 의 helmet=1, head=2). 응답 라벨 이름
STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class Box:
    """사진 너비·높이에 대한 0~1 비율 (x, y 는 왼쪽 위)."""

    label: str
    score: float
    x: float
    y: float
    w: float
    h: float


class VisionUnavailable(Exception):
    """모델 파일이 없거나 onnxruntime 을 불러오지 못함 (설정·환경 문제)."""


class BadImage(Exception):
    """사진을 열지 못함."""


def load_image(data: bytes) -> Any:
    """사진 바이트 → EXIF 회전을 적용한 RGB PIL 이미지."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BadImage("사진을 열지 못했습니다.") from exc


def letterbox(image: Any, size: int) -> tuple[np.ndarray, float]:
    """비율을 유지해 size×size 에 넣는다 (왼쪽 위, 나머지 114). 반환: (1,3,size,size) float32 BGR, 줄인 비율."""
    from PIL import Image

    w, h = image.size
    r = min(size / h, size / w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = np.asarray(image.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)[:, :, ::-1]   # RGB → BGR
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r


def decode(raw: np.ndarray, size: int) -> np.ndarray:
    """YOLOX 격자 출력 (N, 5+클래스) → 중심x·중심y·너비·높이를 입력 픽셀 단위로 (YOLOX demo_postprocess)."""
    grids, strides = [], []
    for stride in STRIDES:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(strides).astype(np.float32)
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """겹침(IoU)이 threshold 보다 큰 박스는 점수 높은 것만 남긴다. boxes: x1,y1,x2,y2."""
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= threshold]
    return keep


def people_boxes(raw: np.ndarray, *, size: int, ratio: float, width: int, height: int,
                 score_threshold: float, nms_threshold: float) -> list[Box]:
    """모델 출력 → 사람 박스 (원본 사진에 대한 0~1 비율, 점수 높은 순)."""
    pred = decode(raw.reshape(-1, raw.shape[-1]), size)
    scores = pred[:, 4] * pred[:, 5 + PERSON]
    mask = scores >= score_threshold
    if not mask.any():
        return []
    pred, scores = pred[mask], scores[mask]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) / ratio
    boxes: list[Box] = []
    for i in nms(xyxy, scores, nms_threshold):
        x1, y1, x2, y2 = xyxy[i]
        x1, x2 = np.clip([x1, x2], 0, width) / width
        y1, y2 = np.clip([y1, y2], 0, height) / height
        if x2 > x1 and y2 > y1:
            boxes.append(Box("person", round(float(scores[i]), 4), round(float(x1), 4), round(float(y1), 4),
                             round(float(x2 - x1), 4), round(float(y2 - y1), 4)))
    return boxes


def class_boxes(raw: np.ndarray, *, size: int, ratio: float, width: int, height: int, labels: tuple[str, ...],
                thresholds: tuple[float, ...], nms_threshold: float) -> list[Box]:
    """클래스가 여러 개인 모델(안전모) 출력 → 박스. 칸마다 점수가 가장 높은 클래스 하나만 보고, 클래스별 기준 이상만 남긴다.
    NMS 는 클래스를 가리지 않고 함께 한다 (한 머리가 "안전모"와 "맨머리"로 두 번 잡히지 않게)."""
    pred = decode(raw.reshape(-1, raw.shape[-1]), size)
    cls = pred[:, 5:5 + len(labels)]
    best = cls.argmax(axis=1)
    scores = pred[:, 4] * cls[np.arange(len(best)), best]
    mask = scores >= np.asarray(thresholds, dtype=np.float32)[best]
    if not mask.any():
        return []
    pred, scores, best = pred[mask], scores[mask], best[mask]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) / ratio
    boxes: list[Box] = []
    for i in nms(xyxy, scores, nms_threshold):
        x1, y1, x2, y2 = xyxy[i]
        x1, x2 = np.clip([x1, x2], 0, width) / width
        y1, y2 = np.clip([y1, y2], 0, height) / height
        if x2 > x1 and y2 > y1:
            boxes.append(Box(labels[int(best[i])], round(float(scores[i]), 4), round(float(x1), 4), round(float(y1), 4),
                             round(float(x2 - x1), 4), round(float(y2 - y1), 4)))
    return boxes


class YoloxDetector:
    """ONNX 세션을 한 번 만들어 재사용한다 (onnxruntime 의 run 은 여러 스레드에서 불러도 된다).
    labels 를 주면 클래스가 여러 개인 모델(안전모, thresholds 는 클래스별 기준)로, 주지 않으면 COCO 사람 검출로 읽는다."""

    def __init__(self, model_path: str | None, *, size: int = 416, score_threshold: float = 0.4,
                 nms_threshold: float = 0.45, threads: int | None = None, session: Any = None,
                 labels: tuple[str, ...] | None = None, thresholds: tuple[float, ...] | None = None,
                 setting: str = "VISION_MODEL_PATH") -> None:
        self.model_path = model_path
        self.size = size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.threads = threads
        self.labels = labels
        self.thresholds = thresholds or (score_threshold,) * len(labels or ())
        self.setting = setting
        self._session = session
        self._lock = threading.Lock()

    def _get_session(self) -> Any:
        with self._lock:
            if self._session is not None:
                return self._session
            if not self.model_path or not os.path.isfile(self.model_path):
                raise VisionUnavailable(f"사진 인식 모델 파일이 없습니다 ({self.setting}={self.model_path}).")
            try:
                import onnxruntime as ort
            except (ImportError, OSError) as exc:
                raise VisionUnavailable("onnxruntime 을 불러오지 못했습니다.") from exc
            options = ort.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = self.threads
            self._session = ort.InferenceSession(self.model_path, sess_options=options,
                                                 providers=["CPUExecutionProvider"])
            logger.info("사진 인식 모델을 읽었습니다. path=%s", self.model_path)
            return self._session

    def detect(self, data: bytes) -> list[Box]:
        session = self._get_session()
        image = load_image(data)
        tensor, ratio = letterbox(image, self.size)
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        width, height = image.size
        if self.labels:
            return class_boxes(np.asarray(raw), size=self.size, ratio=ratio, width=width, height=height,
                               labels=self.labels, thresholds=self.thresholds, nms_threshold=self.nms_threshold)
        return people_boxes(np.asarray(raw), size=self.size, ratio=ratio, width=width, height=height,
                            score_threshold=self.score_threshold, nms_threshold=self.nms_threshold)
