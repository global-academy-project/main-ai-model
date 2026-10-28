"""단계 기록 전달. 이 모듈은 DB를 모르고, 기록 실패가 본 작업을 막지 않게 한다."""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import StepRecord, StepRecorder

logger = logging.getLogger(__name__)


class StepEmitter:
    """on_step 콜백을 감싼다. 콜백이 없으면 아무것도 하지 않는다."""

    def __init__(self, on_step: StepRecorder | None) -> None:
        self._on_step = on_step

    def emit(self, name: str, input: dict[str, Any], output: dict[str, Any], started: float) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(StepRecord(name, input, output, int((time.perf_counter() - started) * 1000)))  # type: ignore[arg-type]
        except Exception:  # 기록 실패가 근거 수집을 막지 않게 한다
            logger.warning("단계 기록 실패: step=%s", name, exc_info=True)
