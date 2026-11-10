"""음성 → 글 서비스 (C 담당). FastAPI 를 모른다. 녹음은 저장하지 않는다.

오류: 모델 폴더·faster-whisper 없음 503, 녹음을 풀지 못함·너무 김·빈 파일 422, 그 밖의 실행 실패 502.
Spring 은 실패하면 "직접 적어 주세요" 로 안내하고 계속한다. 받아 적은 글은 사람이 확인한 뒤 칸 채우기로 넘어간다.
"""

from __future__ import annotations

import logging
import re

from app.exceptions import BusinessError
from app.schemas.base_schemas import DataResponse
from app.schemas.stt import SttTranscript
from app.stt.whisper import MODEL_NAME, BadAudio, SttUnavailable, TooLong, WhisperTranscriber

log = logging.getLogger(__name__)


def transcribe_audio(transcriber: WhisperTranscriber, data: bytes, language: str | None) -> DataResponse[SttTranscript]:
    if not data:
        raise BusinessError(code="VALIDATION_ERROR", message="녹음 파일이 필요합니다.", status_code=422)
    lang = language if language and re.fullmatch(r"[a-z]{2}", language) else "ko"
    try:
        result = transcriber.transcribe(data, lang)
    except SttUnavailable as exc:
        log.warning("STT 를 쓸 수 없습니다: %s", exc)
        raise BusinessError(code="SERVICE_UNAVAILABLE", message="음성 인식 모델을 쓸 수 없습니다.", status_code=503) from exc
    except TooLong as exc:
        raise BusinessError(code="VALIDATION_ERROR", message=str(exc), status_code=422) from exc
    except BadAudio as exc:
        raise BusinessError(code="VALIDATION_ERROR", message="녹음을 풀지 못했습니다.", status_code=422) from exc
    except Exception as exc:  # noqa: BLE001 - 모델 실행 오류는 원문을 로그에만
        log.error("STT 실패", exc_info=exc)
        raise BusinessError(code="BAD_GATEWAY", message="음성 인식에 실패했습니다.", status_code=502) from exc
    return DataResponse[SttTranscript](data=SttTranscript(text=result.text, language=result.language, model=MODEL_NAME))
