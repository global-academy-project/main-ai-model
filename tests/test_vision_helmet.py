"""안전모 모델(클래스: 안전모 쓴 머리, 맨머리) 읽기와 인원·미착용 계산. onnxruntime·모델 파일 없이 가짜 세션으로 돈다."""

from __future__ import annotations

import math

import numpy as np

from app.services.vision_service import HELMET_MODEL_NAME, detect_tbm_photo
from app.vision.yolox import HELMET_LABELS, MODEL_NAME, YoloxDetector, VisionUnavailable, class_boxes
from tests.test_vision import FakeSession, encode, jpeg

HSIZE = 640
HROWS = sum((HSIZE // s) ** 2 for s in (8, 16, 32))   # 8400


def encode_heads(boxes: list[tuple], stride: int = 8) -> np.ndarray:
    """입력 픽셀 기준 (중심x, 중심y, 너비, 높이, 점수, 클래스번호[, stride]) → (1, HROWS, 5+2)"""
    raw = np.zeros((HROWS, 7), dtype=np.float32)
    raw[:, 2:4] = -10
    for cx, cy, w, h, score, cls, *rest in boxes:
        stride = rest[0] if rest else stride
        base = sum((HSIZE // t) ** 2 for t in (8, 16, 32) if t < stride)
        n = HSIZE // stride
        gx, gy = int(cx // stride), int(cy // stride)
        i = base + gy * n + gx
        raw[i, :4] = [cx / stride - gx, cy / stride - gy, math.log(w / stride), math.log(h / stride)]
        raw[i, 4] = 1.0
        raw[i, 5 + cls] = score
    return raw[None]


def helmet(raw: np.ndarray, head_threshold: float = 0.5) -> YoloxDetector:
    return YoloxDetector(None, size=HSIZE, session=FakeSession(raw), labels=HELMET_LABELS,
                         thresholds=(0.4, head_threshold))


def test_class_boxes_label_threshold_and_joint_nms():
    raw = encode_heads([(100, 100, 40, 40, 0.9, 0), (300, 100, 40, 40, 0.45, 1), (500, 100, 40, 40, 0.8, 1)])
    boxes = class_boxes(raw, size=HSIZE, ratio=1.0, width=HSIZE, height=HSIZE, labels=HELMET_LABELS,
                        thresholds=(0.4, 0.5), nms_threshold=0.45)
    assert [b.label for b in boxes] == ["helmet", "no_helmet"]       # 0.45 맨머리는 기준(0.5) 미만이라 버림
    assert boxes[0].x == round(80 / HSIZE, 4)

    same_head = encode_heads([(100, 100, 40, 40, 0.9, 0, 8), (101, 101, 40, 40, 0.7, 1, 16)])   # 같은 머리가 맨머리로도 → 하나만
    boxes = class_boxes(same_head, size=HSIZE, ratio=1.0, width=HSIZE, height=HSIZE, labels=HELMET_LABELS,
                        thresholds=(0.4, 0.5), nms_threshold=0.45)
    assert [b.label for b in boxes] == ["helmet"]


def test_people_and_helmet_missing_from_heads_and_unseen_people():
    # 사람 박스 3개 (416 기준): 2명은 머리가 보이고, 1명은 머리가 안 보임
    person = YoloxDetector(None, session=FakeSession(encode([
        (60, 200, 60, 300, 0.9), (200, 200, 60, 300, 0.9), (350, 200, 60, 300, 0.9)])))
    # 머리 (640 기준, 416 사진이 640 에 그대로 커짐 → 좌표 × 640/416): 2명 머리 + 사람 박스에 안 잡힌 맨머리 1명
    s = HSIZE / 416
    heads = helmet(encode_heads([(60 * s, 70 * s, 20 * s, 20 * s, 0.9, 0), (200 * s, 70 * s, 20 * s, 20 * s, 0.9, 1),
                                 (300 * s, 20 * s, 20 * s, 20 * s, 0.8, 1)]))
    data = detect_tbm_photo(person, jpeg(416, 416), heads).model_dump(mode="json", by_alias=True)["data"]
    assert data["model"] == HELMET_MODEL_NAME
    assert data["people"] == 4               # 머리 3 + 머리가 안 보이는 사람 1
    assert data["helmetMissing"] == 2
    assert sorted(b["label"] for b in data["boxes"]) == ["helmet", "no_helmet", "no_helmet", "person"]


def test_without_helmet_model_or_when_it_fails_people_only():
    person = YoloxDetector(None, session=FakeSession(encode([(100, 100, 64, 128, 0.9)])))
    data = detect_tbm_photo(person, jpeg(416, 416)).model_dump(mode="json", by_alias=True)["data"]
    assert (data["model"], data["people"], data["helmetMissing"]) == (MODEL_NAME, 1, None)

    missing_file = YoloxDetector("/no/such/helmet.onnx", labels=HELMET_LABELS, setting="VISION_HELMET_MODEL_PATH")
    data = detect_tbm_photo(person, jpeg(416, 416), missing_file).model_dump(mode="json", by_alias=True)["data"]
    assert (data["model"], data["people"], data["helmetMissing"]) == (MODEL_NAME, 1, None)

    class Broken(YoloxDetector):
        def detect(self, data):
            raise RuntimeError("onnx 실행 실패")

    data = detect_tbm_photo(person, jpeg(416, 416), Broken(None)).model_dump(mode="json", by_alias=True)["data"]
    assert data["helmetMissing"] is None


def test_missing_helmet_file_names_its_setting():
    try:
        YoloxDetector("/no/such.onnx", labels=HELMET_LABELS, setting="VISION_HELMET_MODEL_PATH").detect(jpeg(10, 10))
    except VisionUnavailable as exc:
        assert "VISION_HELMET_MODEL_PATH" in str(exc)
    else:
        raise AssertionError("VisionUnavailable 이 나야 함")
