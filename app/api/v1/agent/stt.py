"""Spring 이 부르는 음성 → 글 경로 (라우터만, C 담당). 녹음은 저장하지 않는다.

    POST /api/v1/stt/transcribe   multipart: file(녹음), language(기본 ko)
        성공 200  {success: true, data: {text, language, model}}   model = STT_MODEL_NAME (기본 whisper-small)
        실패      {success: false, error: {...}} (모델 없음 503, 녹음 문제 422, 실행 실패 502)

Spring 은 APP_STT_CLIENT=fastapi 일 때만 부른다 (대기 기본 30초, 녹음 최대 30초·2MB).
모델 폴더: STT_MODEL_PATH (scripts/download_models.py --only stt 로 미리 받아 둔다. 서버는 내려받지 않는다).
multipart 를 받으려면 python-multipart 가 필요하다 (requirements.txt).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core import Settings, get_settings
from app.exceptions import BusinessError
from app.schemas import DataResponse
from app.schemas.stt import SttTranscript
from app.services.stt_service import transcribe_audio
from app.stt.whisper import WhisperTranscriber

router = APIRouter(prefix="/api/v1/stt", tags=["stt"])

MAX_BYTES = 2 * 1024 * 1024   # Spring app.stt.max-bytes 와 같게

_transcribers: dict[tuple, WhisperTranscriber] = {}


def get_transcriber(settings: Annotated[Settings, Depends(get_settings)]) -> WhisperTranscriber:
    """설정이 같으면 같은 모델을 쓴다 (처음 쓸 때 한 번 읽음)."""
    key = (settings.stt_model_path, settings.stt_compute_type, settings.stt_threads, settings.stt_beam_size,
           settings.stt_max_seconds, settings.stt_model_name)
    transcriber = _transcribers.get(key)
    if transcriber is None:
        transcriber = _transcribers.setdefault(key, WhisperTranscriber(
            settings.stt_model_path, compute_type=settings.stt_compute_type, threads=settings.stt_threads,
            beam_size=settings.stt_beam_size, max_seconds=settings.stt_max_seconds, name=settings.stt_model_name))
    return transcriber


@router.post("/transcribe", response_model=DataResponse[SttTranscript])
def transcribe(
        file: Annotated[UploadFile, File(description="녹음 (WebM·Ogg·MP4·WAV)")],
        transcriber: Annotated[WhisperTranscriber, Depends(get_transcriber)],
        language: Annotated[str | None, Form()] = "ko",
) -> DataResponse[SttTranscript]:
    # 모델 실행이 동기라 def (스레드풀에서 실행). 녹음은 메모리에서만 쓴다
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise BusinessError(code="VALIDATION_ERROR", message="녹음은 2MB 까지 받을 수 있습니다.", status_code=413)
    return transcribe_audio(transcriber, data, language)
