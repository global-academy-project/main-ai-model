"""음성 → 글 (Whisper small, faster-whisper) 테스트. faster-whisper·모델 없이 가짜 모델로 돈다."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from app.exceptions import BusinessError
from app.services.stt_service import transcribe_audio
from app.stt.normalize import korean_number, normalize
from app.stt.whisper import INITIAL_PROMPT, WhisperTranscriber


class FakeWhisper:
    def __init__(self, texts: list[str], duration: float = 5.0, error: Exception | None = None) -> None:
        self.texts, self.duration, self.error, self.calls = texts, duration, error, []

    def transcribe(self, audio, **kw):
        self.calls.append({"bytes": audio.read(), **kw})
        if self.error:
            raise self.error
        return (SimpleNamespace(text=t) for t in self.texts), SimpleNamespace(language="ko", duration=self.duration)


# ---- 표기 맞추기 ----

@pytest.mark.parametrize("spoken, written", [
    ("산소 이십 점 구 퍼센트 황화수소 삼 피피엠 엘이엘 영 프로", "산소 20.9% 황화수소 3ppm LEL 0%"),
    ("산소 이십점구, 가연성가스 영, 감시인 김도윤", "산소 20.9, 가연성가스 0, 감시인 김도윤"),
    ("이산화탄소 영점오 퍼센트", "이산화탄소 0.5%"),
    ("산소 20.9 퍼센트 일산화탄소 5 피피엠", "산소 20.9% 일산화탄소 5ppm"),
    ("황화수소 삼십 피피엠", "황화수소 30ppm"),
    ("산소 십칠 점 오", "산소 17.5"),
])
def test_normalize_measurements(spoken, written):
    assert normalize(spoken) == written


@pytest.mark.parametrize("text", ["이 작업은 오늘 오후에", "산소 이상 없음", "삼성 산소 이삼"])
def test_normalize_leaves_other_korean(text):
    assert normalize(text) == text   # 조사·낱말 속 "이"·"오"·"삼" 은 숫자로 바꾸지 않는다


def test_korean_number():
    assert korean_number("이십점구") == "20.9"
    assert korean_number("천이백삼십사") == "1234"
    assert korean_number("영") == "0"
    assert korean_number("이삼") is None


# ---- 받아 적기 ----

def test_transcribe_uses_hint_and_normalizes():
    fake = FakeWhisper(["산소 이십 점 구 퍼센트", " 황화수소 삼 피피엠"])
    out = WhisperTranscriber(None, model=fake).transcribe(b"audio", "ko")
    assert out.text == "산소 20.9% 황화수소 3ppm"
    call = fake.calls[0]
    assert call["bytes"] == b"audio" and call["language"] == "ko" and call["initial_prompt"] == INITIAL_PROMPT
    assert call["beam_size"] == 1


def test_service_shape_and_errors(tmp_path):
    ok = WhisperTranscriber(None, model=FakeWhisper(["산소 이십 점 구"]))
    dumped = transcribe_audio(ok, b"audio", None).model_dump(mode="json", by_alias=True)
    assert dumped == {"success": True, "data": {"text": "산소 20.9", "language": "ko", "model": "whisper-small"}}

    with pytest.raises(BusinessError) as missing:    # 모델 폴더 없음 → 503 (내려받지 않는다)
        transcribe_audio(WhisperTranscriber(str(tmp_path / "none")), b"audio", "ko")
    assert missing.value.status_code == 503
    with pytest.raises(BusinessError) as long_:      # 너무 긴 녹음 → 422
        transcribe_audio(WhisperTranscriber(None, model=FakeWhisper(["말"], duration=60)), b"audio", "ko")
    assert long_.value.status_code == 422
    with pytest.raises(BusinessError) as empty:
        transcribe_audio(ok, b"", "ko")
    assert empty.value.status_code == 422
    with pytest.raises(BusinessError) as boom:       # 모델 실행 실패 → 502
        transcribe_audio(WhisperTranscriber(None, model=FakeWhisper([], error=RuntimeError("x"))), b"audio", "ko")
    assert boom.value.status_code == 502


def test_language_is_checked():
    fake = FakeWhisper(["hello"])
    transcribe_audio(WhisperTranscriber(None, model=fake), b"a", "KOREAN")
    assert fake.calls[0]["language"] == "ko"   # 이상한 값이면 ko


def test_silence_gives_empty_text():
    out = transcribe_audio(WhisperTranscriber(None, model=FakeWhisper([" ", ""])), b"a", "ko")
    assert out.data.text == ""   # Spring 이 "알아들은 말이 없습니다" 로 안내한다


def _has_multipart() -> bool:
    return importlib.util.find_spec("fastapi") is not None and any(
        importlib.util.find_spec(m) for m in ("python_multipart", "multipart"))


@pytest.mark.skipif(not _has_multipart(), reason="fastapi·python-multipart 가 있어야 경로를 시험한다")
def test_route_returns_spring_shape():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.agent.stt import get_transcriber, router
    from app.handlers import business_error_handler

    app = FastAPI()
    app.add_exception_handler(BusinessError, business_error_handler)
    app.include_router(router)
    fake = FakeWhisper(["산소 이십 점 구 퍼센트"])
    app.dependency_overrides[get_transcriber] = lambda: WhisperTranscriber(None, model=fake)
    http = TestClient(app)
    res = http.post("/api/v1/stt/transcribe", files={"file": ("voice", b"webm-bytes", "audio/webm")}, data={"language": "ko"})
    assert res.status_code == 200 and res.json()["data"] == {"text": "산소 20.9%", "language": "ko", "model": "whisper-small"}

    app.dependency_overrides[get_transcriber] = lambda: WhisperTranscriber(None)
    res = http.post("/api/v1/stt/transcribe", files={"file": ("voice", b"webm-bytes", "audio/webm")})
    assert res.status_code == 503 and res.json()["success"] is False
