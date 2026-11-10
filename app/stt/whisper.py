"""Whisper small (MIT) 을 faster-whisper(CTranslate2) 로 CPU int8 실행. 처음 쓸 때 한 번 읽는다.

- 모델은 로컬 폴더만 읽는다 (STT_MODEL_PATH, 내려받지 않음 → 폐쇄망). 준비는 scripts/download_models.py --only stt.
- 녹음은 메모리에서만 풀고 파일로 남기지 않는다 (faster-whisper 가 PyAV 로 webm·ogg·mp4·wav 를 푼다. 시스템 ffmpeg 불필요).
- 용어 힌트(initial_prompt)로 측정 항목·단위 표기를 유도하고, 받아 적은 뒤 normalize() 로 표기를 맞춘다.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from .normalize import normalize

logger = logging.getLogger(__name__)

MODEL_NAME = "whisper-small"
# 받아 적을 때 이 표기 쪽으로 기울게 하는 예시 문장 (값을 넣는 것이 아니라 표기 힌트)
INITIAL_PROMPT = "산소 20.9%, 탄산가스 0.5%, 일산화탄소 5ppm, 황화수소 3ppm, 가연성가스 LEL 0%, 암모니아, 염소, 감시인 김도윤."


class SttUnavailable(Exception):
    """모델 폴더가 없거나 faster-whisper 를 불러오지 못함."""


class BadAudio(Exception):
    """녹음을 풀지 못함."""


class TooLong(Exception):
    """녹음이 너무 김."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration: float


class WhisperTranscriber:
    def __init__(self, model_path: str | None, *, compute_type: str = "int8", threads: int | None = None,
                 beam_size: int = 1, max_seconds: float = 35.0, model: Any = None) -> None:
        self.model_path = model_path
        self.compute_type = compute_type
        self.threads = threads
        self.beam_size = beam_size
        self.max_seconds = max_seconds
        self._model = model
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.model_path or not os.path.isdir(self.model_path):
                raise SttUnavailable(f"STT 모델 폴더가 없습니다 (STT_MODEL_PATH={self.model_path}).")
            try:
                from faster_whisper import WhisperModel
            except (ImportError, OSError) as exc:
                raise SttUnavailable("faster-whisper 를 불러오지 못했습니다.") from exc
            options: dict[str, Any] = {"device": "cpu", "compute_type": self.compute_type, "local_files_only": True}
            if self.threads:
                options["cpu_threads"] = self.threads
            self._model = WhisperModel(self.model_path, **options)
            logger.info("STT 모델을 읽었습니다. path=%s compute=%s", self.model_path, self.compute_type)
            return self._model

    def transcribe(self, data: bytes, language: str = "ko") -> Transcript:
        model = self._get_model()
        try:
            segments, info = model.transcribe(
                io.BytesIO(data), language=language, initial_prompt=INITIAL_PROMPT,
                beam_size=self.beam_size, condition_on_previous_text=False, vad_filter=False,
            )
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            if duration > self.max_seconds:
                raise TooLong(f"녹음이 {duration:.0f}초입니다 (최대 {self.max_seconds:.0f}초).")
            raw = " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip())
        except (TooLong, SttUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 - PyAV·CTranslate2 오류는 원문을 로그에만
            name = type(exc).__name__
            if type(exc).__module__.split(".")[0] == "av" or name in ("InvalidDataError", "EOFError"):
                raise BadAudio("녹음을 풀지 못했습니다.") from exc
            raise
        return Transcript(text=normalize(raw), language=getattr(info, "language", language) or language,
                          duration=duration)
