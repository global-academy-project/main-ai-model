import asyncio
from types import SimpleNamespace

import pytest

from app.clients.fake_client import FakeTextGenerationClient
from app.clients.openai_client import OpenAITextGenerationClient
from app.core.dependecies.client_dep import get_summary_service, get_text_generation_client
from app.services.summary_service import SummaryCommand, SummaryService
from tests.helpers import make_settings


def summarize(client, text="원문", length="medium", tone="neutral"):
    return asyncio.run(SummaryService(client=client).summarize(SummaryCommand(text=text, length=length, tone=tone)))


class TestSummaryService:
    def test_result_is_stripped_and_keeps_options(self):
        result = summarize(FakeTextGenerationClient(response_text="  요약입니다  \n"), length="short", tone="friendly")
        assert (result.summary, result.length, result.tone) == ("요약입니다", "short", "friendly")

    def test_input_text_is_stripped_before_sending(self):
        client = FakeTextGenerationClient()
        summarize(client, text="  \n본문 내용\n  ")
        assert client.last_input_text == "본문 내용"

    @pytest.mark.parametrize("length,phrase", [("short", "3문장"), ("medium", "5개 문장"), ("long", "자세히")])
    def test_length_instruction(self, length, phrase):
        client = FakeTextGenerationClient()
        summarize(client, length=length)
        assert phrase in client.last_instructions

    @pytest.mark.parametrize("tone,phrase", [("neutral", "중립"), ("professional", "전문적"), ("friendly", "친근")])
    def test_tone_instruction(self, tone, phrase):
        client = FakeTextGenerationClient()
        summarize(client, tone=tone)
        assert phrase in client.last_instructions

    def test_instructions_forbid_adding_facts(self):
        client = FakeTextGenerationClient()
        summarize(client)
        assert "원문에 없는 사실을 추가하지 마세요" in client.last_instructions

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_blank_model_output_is_an_error(self, blank):
        with pytest.raises(ValueError):
            summarize(FakeTextGenerationClient(response_text=blank))


class FakeAsyncResponses:
    def __init__(self, output_text):
        self.output_text, self.calls = output_text, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


def openai_client(output_text):
    client = OpenAITextGenerationClient(api_key="sk-test", model="test-model", timeout_seconds=5, max_output_tokens=77)
    responses = FakeAsyncResponses(output_text)
    client._client = SimpleNamespace(responses=responses)
    return client, responses


class TestOpenAITextGenerationClient:
    def test_returns_stripped_text_and_passes_options(self):
        client, responses = openai_client("  hello \n")
        assert asyncio.run(client.generate(instructions="inst", input_text="text")) == "hello"
        call = responses.calls[0]
        assert (call["model"], call["instructions"], call["input"], call["max_output_tokens"]) == ("test-model", "inst", "text", 77)

    @pytest.mark.parametrize("empty", ["", "   "])
    def test_empty_provider_response_is_an_error(self, empty):
        client, _ = openai_client(empty)
        with pytest.raises(RuntimeError):
            asyncio.run(client.generate(instructions="i", input_text="t"))


class TestClientSelection:
    def test_openai_engine_builds_openai_client(self):
        assert isinstance(get_text_generation_client(make_settings(engine="OPENAI")), OpenAITextGenerationClient)

    def test_fake_provider_for_local_testing(self):
        client = get_text_generation_client(SimpleNamespace(engine="FAKE"))
        assert isinstance(client, FakeTextGenerationClient)

    @pytest.mark.parametrize("engine", ["AUTO", "PDFIUM", "TESSERACT", "PADDLE"])
    def test_non_openai_engine_has_no_summary_provider(self, engine):
        # 알려진 제약: ENGINE 이 OPENAI 가 아니면 요약 클라이언트를 만들 수 없어,
        # 이 의존성을 쓰는 /extract 도 함께 실패한다. (ENGINE 의 두 가지 역할 문제)
        with pytest.raises(RuntimeError, match="지원하지 않는"):
            get_text_generation_client(make_settings(engine=engine))

    @pytest.mark.parametrize("missing", ["openai_api_key", "openai_model"])
    def test_openai_provider_requires_key_and_model(self, missing):
        values = dict(engine="OPENAI", openai_api_key=SimpleNamespace(get_secret_value=lambda: "k"),
                      openai_model="m", ai_timeout_seconds=5, ai_max_output_tokens=10)
        values[missing] = None
        with pytest.raises(RuntimeError):
            get_text_generation_client(SimpleNamespace(**values))

    def test_summary_service_wraps_client(self):
        assert isinstance(get_summary_service(FakeTextGenerationClient()), SummaryService)
