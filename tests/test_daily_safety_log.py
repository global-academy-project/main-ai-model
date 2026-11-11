from app.agent.daily_safety_log import DailySafetyLogAgent
from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse
from app.schemas.agent_chat import AgentChatRequest
from app.schemas.daily_safety_log import DailySafetyLogInput


class _FakeDailySafetyLogClient(LLMClient):
    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        return LLMResponse(
            text=(
                '{"summary":"정상", "tbm":"TBM 요약", "deadline":"마감 요약", '
                '"dailySafetyLog":"안전일지 요약", "warnings":[], "evidence":[]}'
            )
        )


def test_agent_chat_request_accepts_daily_safety_log_aliases():
    request = AgentChatRequest.model_validate(
        {
            "message": "안전일지 초안 생성",
            "scenario": "D",
            "mode": "VERIFY",
            "dailySafetyLog": {"siteId": 4, "date": "2026-11-11"},
        }
    )

    assert request.daily_safety_log is not None
    assert request.daily_safety_log.site_id == 4
    assert request.model_dump(by_alias=True)["dailySafetyLog"]["siteId"] == 4


def test_daily_safety_log_agent_converts_model_json_to_result():
    result = DailySafetyLogAgent(llm=_FakeDailySafetyLogClient()).run(
        DailySafetyLogInput(site_id=4, date="2026-11-11")
    )

    assert result.site_id == 4
    assert result.summary == "정상"
    assert result.daily_safety_log == "안전일지 요약"
