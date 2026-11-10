"""TBM 사진 인식 (YOLOX-Nano) 테스트. onnxruntime·모델 파일 없이 가짜 세션으로 돈다 (numpy·Pillow 만 필요)."""

from __future__ import annotations

import io
import importlib.util
import math

import numpy as np
import pytest
from PIL import Image

from app.exceptions import BusinessError
from app.services.vision_service import detect_tbm_photo
from app.vision.yolox import (Box, YoloxDetector, decode, letterbox, load_image, nms, people_boxes)

SIZE = 416
ROWS = sum((SIZE // s) ** 2 for s in (8, 16, 32))   # 3549


def jpeg(w: int, h: int, orientation: int | None = None) -> bytes:
    buf = io.BytesIO()
    image = Image.new("RGB", (w, h), (90, 140, 90))
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buf, "JPEG", exif=exif)
    else:
        image.save(buf, "JPEG")
    return buf.getvalue()


def encode(boxes: list[tuple], stride: int = 32) -> np.ndarray:
    """입력 픽셀 기준 (중심x, 중심y, 너비, 높이, 점수[, stride]) → YOLOX 격자 출력 (1, ROWS, 85)."""
    raw = np.zeros((ROWS, 85), dtype=np.float32)
    raw[:, 2:4] = -10   # 나머지 칸은 아주 작은 박스·점수 0
    for cx, cy, w, h, score, *rest in boxes:
        s = rest[0] if rest else stride
        base = sum((SIZE // t) ** 2 for t in (8, 16, 32) if t < s)
        n = SIZE // s
        gx, gy = int(cx // s), int(cy // s)
        i = base + gy * n + gx
        raw[i, :4] = [cx / s - gx, cy / s - gy, math.log(w / s), math.log(h / s)]
        raw[i, 4] = 1.0
        raw[i, 5] = score   # 사람 클래스
    return raw[None]


class FakeSession:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw, self.inputs = raw, []

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["images"])
        return [self.raw]


def test_exif_rotation_is_applied():
    assert load_image(jpeg(200, 100)).size == (200, 100)
    assert load_image(jpeg(200, 100, orientation=6)).size == (100, 200)   # 세로로 찍은 휴대폰 사진


def test_letterbox_keeps_ratio_top_left_bgr():
    tensor, r = letterbox(Image.new("RGB", (832, 416), (10, 20, 30)), SIZE)
    assert tensor.shape == (1, 3, SIZE, SIZE) and tensor.dtype == np.float32 and r == 0.5
    assert tensor[0, :, 0, 0].tolist() == [30, 20, 10]          # BGR
    assert tensor[0, :, SIZE - 1, 0].tolist() == [114, 114, 114]   # 아래는 회색으로 채움


def test_decode_and_nms():
    raw = encode([(100, 100, 64, 128, 0.9)])[0]
    out = decode(raw, SIZE)
    i = int(np.argmax(out[:, 5]))
    assert np.allclose(out[i, :4], [100, 100, 64, 128], atol=1e-3)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


def test_people_boxes_are_ratios_of_original_photo():
    # 832×416 사진 → 비율 0.5 로 줄여 넣음. 입력 픽셀 박스 (100,100,64,128) = 원본 (200,200,128,256)
    raw = encode([(100, 100, 64, 128, 0.9), (300, 150, 40, 80, 0.3), (102, 101, 64, 128, 0.85, 16)])
    boxes = people_boxes(raw, size=SIZE, ratio=0.5, width=832, height=416, score_threshold=0.4, nms_threshold=0.45)
    assert len(boxes) == 1                       # 0.3 은 기준 미만, 겹친 0.85 는 NMS 로 하나만
    b = boxes[0]
    assert b.label == "person" and b.score == pytest.approx(0.9)
    assert (b.x, b.y, b.w, b.h) == pytest.approx((136 / 832, 72 / 416, 128 / 832, 256 / 416), abs=1e-3)


def test_detector_runs_session_with_letterboxed_input():
    session = FakeSession(encode([(100, 100, 64, 128, 0.9), (300, 100, 64, 128, 0.8)]))
    detector = YoloxDetector(None, session=session)
    boxes = detector.detect(jpeg(832, 416))
    assert [b.label for b in boxes] == ["person", "person"]
    assert session.inputs[0].shape == (1, 3, SIZE, SIZE)


def test_service_shape_and_errors(tmp_path):
    detector = YoloxDetector(None, session=FakeSession(encode([(100, 100, 64, 128, 0.9)])))
    dumped = detect_tbm_photo(detector, jpeg(416, 416)).model_dump(mode="json", by_alias=True)
    data = dumped["data"]
    assert dumped["success"] is True and data["model"] == "YOLOX-Nano"
    assert data["people"] == 1 and data["helmetMissing"] is None
    assert set(data["boxes"][0]) == {"label", "score", "x", "y", "w", "h"}

    with pytest.raises(BusinessError) as missing:   # 모델 파일 없음 → 503 (내려받지 않는다)
        detect_tbm_photo(YoloxDetector(str(tmp_path / "none.onnx")), jpeg(10, 10))
    assert missing.value.status_code == 503
    with pytest.raises(BusinessError) as bad:       # 사진이 아님 → 422
        detect_tbm_photo(detector, b"<html>not a photo</html>")
    assert bad.value.status_code == 422
    with pytest.raises(BusinessError) as empty:
        detect_tbm_photo(detector, b"")
    assert empty.value.status_code == 422


def _has_multipart() -> bool:
    return importlib.util.find_spec("fastapi") is not None and any(
        importlib.util.find_spec(m) for m in ("python_multipart", "multipart"))


@pytest.mark.skipif(not _has_multipart(), reason="fastapi·python-multipart 가 있어야 경로를 시험한다")
def test_route_returns_spring_shape():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.agent.vision import get_detector, router
    from app.exceptions import BusinessError as BE
    from app.handlers import business_error_handler

    app = FastAPI()
    app.add_exception_handler(BE, business_error_handler)
    app.include_router(router)
    app.dependency_overrides[get_detector] = lambda: YoloxDetector(None, session=FakeSession(encode([(100, 100, 64, 128, 0.9)])))
    http = TestClient(app)
    res = http.post("/api/v1/vision/tbm-photo", files={"file": ("tbm.jpg", jpeg(416, 416), "image/jpeg")})
    assert res.status_code == 200 and res.json()["data"]["people"] == 1

    app.dependency_overrides[get_detector] = lambda: YoloxDetector(None)
    res = http.post("/api/v1/vision/tbm-photo", files={"file": ("tbm.jpg", jpeg(10, 10), "image/jpeg")})
    assert res.status_code == 503 and res.json()["success"] is False


def test_box_is_plain_data():
    assert Box("person", 0.9, 0.1, 0.2, 0.3, 0.4).w == 0.3
