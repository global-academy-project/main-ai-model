"""TBM 사진 인식 서비스 (C 담당). FastAPI 를 모른다.

사진 → 사람 박스(YOLOX-Nano) → Spring 응답 모양. 안전모는 학습 전이라 null.
오류: 모델 파일·onnxruntime 없음 503, 사진을 열지 못함 422, 그 밖의 실행 실패 502.
Spring 은 실패하면 파일 정보 확인(촬영 날짜·같은 사진)만 남기고 계속한다.
"""

from __future__ import annotations

import logging

from app.exceptions import BusinessError
from app.schemas.base_schemas import DataResponse
from app.schemas.vision import TbmPhotoVision, VisionBox
from app.vision.yolox import MODEL_NAME, BadImage, VisionUnavailable, YoloxDetector

log = logging.getLogger(__name__)


def detect_tbm_photo(detector: YoloxDetector, data: bytes) -> DataResponse[TbmPhotoVision]:
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
    return DataResponse[TbmPhotoVision](data=TbmPhotoVision(
        model=MODEL_NAME,
        people=len(boxes),
        helmet_missing=None,   # 안전모 클래스는 학습 후 (공식 COCO 가중치에는 없음)
        boxes=[VisionBox(label=b.label, score=b.score, x=b.x, y=b.y, w=b.w, h=b.h) for b in boxes],
    ))
