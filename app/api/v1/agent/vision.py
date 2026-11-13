"""Spring 이 부르는 TBM 사진 인식 경로 (라우터만, C 담당).

    POST /api/v1/vision/tbm-photo   multipart: file
        성공 200  {success: true, data: {model, people, helmetMissing, boxes:[{label, score, x, y, w, h}]}}
        실패      {success: false, error: {...}} (모델 없음 503, 사진 아님 422, 실행 실패 502)

Spring 은 APP_PHOTO_VISION_CLIENT=fastapi 일 때만 부른다 (대기 기본 20초). 박스 좌표는 EXIF 회전을 적용한 사진 기준 0~1 비율.
모델 파일: VISION_MODEL_PATH (scripts/download_models.py 로 미리 받아 둔다. 서버는 내려받지 않는다).
안전모: VISION_HELMET_MODEL_PATH (학습한 모델, 11/13). 있으면 helmetMissing 과 helmet / no_helmet 박스가 온다.
multipart 를 받으려면 python-multipart 패키지가 필요하다 (requirements.txt).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core import Settings, get_settings
from app.exceptions import BusinessError
from app.schemas import DataResponse
from app.schemas.vision import TbmPhotoVision
from app.services.vision_service import detect_tbm_photo
from app.vision.yolox import HELMET_LABELS, YoloxDetector

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])

MAX_BYTES = 20 * 1024 * 1024   # Spring app.attachment.max-bytes 와 같게

_detectors: dict[tuple, YoloxDetector] = {}


def get_detector(settings: Annotated[Settings, Depends(get_settings)]) -> YoloxDetector:
    """설정이 같으면 같은 검출기(ONNX 세션)를 쓴다."""
    key = (settings.vision_model_path, settings.vision_input_size, settings.vision_score_threshold,
           settings.vision_nms_threshold, settings.vision_threads)
    detector = _detectors.get(key)
    if detector is None:
        detector = _detectors.setdefault(key, YoloxDetector(
            settings.vision_model_path, size=settings.vision_input_size,
            score_threshold=settings.vision_score_threshold, nms_threshold=settings.vision_nms_threshold,
            threads=settings.vision_threads))
    return detector


def get_helmet_detector(settings: Annotated[Settings, Depends(get_settings)]) -> YoloxDetector | None:
    """안전모 모델 (VISION_HELMET_MODEL_PATH). 비어 있으면 None → 사람 수만 센다"""
    if not settings.vision_helmet_model_path:
        return None
    key = ("helmet", settings.vision_helmet_model_path, settings.vision_helmet_input_size,
           settings.vision_helmet_score_threshold, settings.vision_head_score_threshold,
           settings.vision_nms_threshold, settings.vision_threads)
    detector = _detectors.get(key)
    if detector is None:
        detector = _detectors.setdefault(key, YoloxDetector(
            settings.vision_helmet_model_path, size=settings.vision_helmet_input_size,
            nms_threshold=settings.vision_nms_threshold, threads=settings.vision_threads, labels=HELMET_LABELS,
            thresholds=(settings.vision_helmet_score_threshold, settings.vision_head_score_threshold),
            setting="VISION_HELMET_MODEL_PATH"))
    return detector


@router.post("/tbm-photo", response_model=DataResponse[TbmPhotoVision])
def tbm_photo(
        file: Annotated[UploadFile, File(description="TBM 사진 (JPEG·PNG)")],
        detector: Annotated[YoloxDetector, Depends(get_detector)],
        helmet_detector: Annotated[YoloxDetector | None, Depends(get_helmet_detector)],
) -> DataResponse[TbmPhotoVision]:
    # 모델 실행이 동기라 def (스레드풀에서 실행)
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise BusinessError(code="VALIDATION_ERROR", message="사진은 20MB 까지 받을 수 있습니다.", status_code=413)
    return detect_tbm_photo(detector, data, helmet_detector)
