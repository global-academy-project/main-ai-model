"""TBM 사진 인식 서비스 (C 담당). FastAPI 를 모른다.

사진 → 사람 박스(YOLOX-Nano) → Spring 응답 모양.
안전모 모델(VISION_HELMET_MODEL_PATH)이 있으면 머리도 찾는다 (11/13):
  - 인원 = 머리 수 + 머리가 안 보이는 사람 박스 수 (겹쳐 서서 몸이 가려도 머리는 보이는 경우가 많고,
    머리가 가려지거나 잘린 사람은 사람 박스로 센다)
  - 안전모 미착용 = 맨머리 수, 박스 라벨 helmet / no_helmet (+ 머리가 안 보이는 사람은 person)
  - 안전모 모델만 실패하면 사람 수만 돌려준다 (helmetMissing null)
오류: 모델 파일·onnxruntime 없음 503, 사진을 열지 못함 422, 그 밖의 실행 실패 502.
Spring 은 실패하면 파일 정보 확인(촬영 날짜·같은 사진)만 남기고 계속한다.
"""

from __future__ import annotations

import logging

from app.exceptions import BusinessError
from app.schemas.base_schemas import DataResponse
from app.schemas.vision import TbmPhotoVision, VisionBox
from app.vision.yolox import MODEL_NAME, BadImage, Box, VisionUnavailable, YoloxDetector

HELMET_MODEL_NAME = MODEL_NAME + "+helmet"

log = logging.getLogger(__name__)


def detect_tbm_photo(detector: YoloxDetector, data: bytes,
                     helmet_detector: YoloxDetector | None = None) -> DataResponse[TbmPhotoVision]:
    if not data:
        raise BusinessError(code="VALIDATION_ERROR", message="사진 파일이 필요합니다.", status_code=422)
    try:
        boxes = detector.detect(data)
    except VisionUnavailable as exc:
        log.warning("사진 인식을 쓸 수 없습니다: %s", exc)
        raise BusinessError(code="SERVICE_UNAVAILABLE", message="사진 인식 모델을 쓸 수 없습니다.", status_code=503) from exc
    except BadImage as exc:
        raise BusinessError(code="VALIDATION_ERROR", message="사진을 열지 못했습니다.", status_code=422) from exc
    except Exception as exc:  # noqa: BLE001 - 모델 실행 오류는 원문을 로그에만
        log.error("사진 인식 실패", exc_info=exc)
        raise BusinessError(code="BAD_GATEWAY", message="사진 인식에 실패했습니다.", status_code=502) from exc
    heads = _heads(helmet_detector, data) if helmet_detector is not None else None
    if heads is None:
        return _response(MODEL_NAME, len(boxes), None, boxes)
    unseen = [b for b in boxes if not any(_center_inside(h, b) for h in heads)]   # 머리가 안 보이는 사람
    missing = sum(1 for h in heads if h.label == "no_helmet")
    return _response(HELMET_MODEL_NAME, len(heads) + len(unseen), missing, heads + unseen)


def _heads(helmet_detector: YoloxDetector, data: bytes) -> list[Box] | None:
    """안전모 모델로 머리 찾기. 실패하면 None (사람 수만 돌려준다)"""
    try:
        return helmet_detector.detect(data)
    except VisionUnavailable as exc:
        log.warning("안전모 인식을 쓸 수 없습니다: %s", exc)
    except Exception as exc:  # noqa: BLE001 - 사람 수는 이미 있으므로 안전모만 빼고 계속
        log.error("안전모 인식 실패", exc_info=exc)
    return None


def _center_inside(head: Box, person: Box) -> bool:
    cx, cy = head.x + head.w / 2, head.y + head.h / 2
    return person.x <= cx <= person.x + person.w and person.y <= cy <= person.y + person.h


def _response(model: str, people: int, helmet_missing: int | None, boxes: list[Box]) -> DataResponse[TbmPhotoVision]:
    return DataResponse[TbmPhotoVision](data=TbmPhotoVision(
        model=model,
        people=people,
        helmet_missing=helmet_missing,   # 안전모 모델이 없으면 null
        boxes=[VisionBox(label=b.label, score=b.score, x=b.x, y=b.y, w=b.w, h=b.h) for b in boxes],
    ))
