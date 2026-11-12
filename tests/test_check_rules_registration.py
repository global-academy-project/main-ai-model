"""check_rules 실제 도구 등록: MOCK이면 변경 전과 같고, SPRING이면 C의 도구가 mock을 대신하는지.

네트워크·모델 없이 돈다. Spring은 tests/fakes.py의 FakeSpring, LLM은 빈 응답(규칙 계획으로 대체)이다.
"""

import ast
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent import agent_deps
from app.agent.check_rules_setup import build_check_rules_tool
from app.agent.rule_tools import CHECK_RULES_PARAMETERS, make_check_rules
from app.agent.work_register import WorkRegisterAgent
from app.core import Settings
from app.integrations import InternalApiClient
from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse
from app.schemas.agent_work import WorkRegisterRequest
from app.tools.mock.mock_rules import check_rules as mock_check_rules
from app.tools.tool_setup import build_tool_registry

from .fakes import FakeSpring, check, ok, result, timeout

HOT_MESSAGE = "3번 탱크 앞에서 용접 작업합니다. 아세톤 보관 중입니다. 3명이 14시부터 17시까지, 화재감시자 김철수, 용접기 사용"
TOKEN = "t" * 32


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """내 터미널에 남은 환경변수(INTERNAL_API_*, CHECK_RULES_TOOL 등)가 설정 검사 시험을 바꾸지 않게 비운다."""
    for name in ("INTERNAL_API_BASE_URL", "INTERNAL_API_TOKEN", "CHECK_RULES_TOOL", "RAG_ENABLED", "RAG_BACKEND"):
        monkeypatch.delenv(name, raising=False)


def make_settings(tool: str, **extra) -> Settings:
    base = {"network_env": "OPEN", "check_rules_tool": tool, "rag_enabled": False}
    return Settings(_env_file=None, **{**base, **extra})


def spring_settings() -> Settings:
    return make_settings("SPRING", internal_api_base_url="http://spring:8080", internal_api_token=TOKEN)


class EmptyLLM(LLMClient):
    """빈 응답. 계획 단계가 규칙 계획으로 대체된다."""

    name = "empty"

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        return LLMResponse(text="")


def real_tool(fake: FakeSpring):
    client = InternalApiClient("http://spring:8080", TOKEN, connect_timeout=2, read_timeout=3, transport=fake)
    return make_check_rules(client), CHECK_RULES_PARAMETERS


def run_agent(registry, **request) -> object:
    return WorkRegisterAgent(llm=EmptyLLM(), tools=registry).run(WorkRegisterRequest(message=HOT_MESSAGE, **request))


HOT_BLOCK = result("BLOCK", [
    check("HOT_EXTINGUISHER", "form.extinguisher_ready", "MISSING", expected="확인 필요", category="REQUIRED_FIELD",
          message="소화기 비치를 확인해야 합니다."),
    check("HOT_FIRE_WATCHER", "form.watcher_name", "PASS", actual="김철수", expected="입력 필요", category="STAFFING"),
    check("HOT_LEL_MAX", "measurement.LEL", "MISSING", expected="10 미만 (%LEL)", message="가연성 가스 측정값이 없습니다."),
], evaluation_id=None)


# ── MOCK: 변경 전과 같다 ──

def test_mock_returns_no_tool_and_does_not_import_c_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.agent.rule_tools", None)   # 임포트하면 ImportError
    assert build_check_rules_tool(make_settings("MOCK")) is None


def test_default_registry_keeps_the_mock():
    for registry in (build_tool_registry(), build_tool_registry(check_rules_tool=None)):
        tool = next(t for t in registry.tools() if t.name == "check_rules")
        assert tool.mock is True and tool.owner == "C" and tool.description.startswith("[mock]")
        arguments = {"permit_types": ["HOT_WORK_PERMIT"], "measurements": {"LEL": 3}, "assignments": {"FIRE_WATCHER": "김"}}
        assert registry.call("check_rules", arguments) == mock_check_rules(arguments)


def test_mock_agent_run_is_identical_with_and_without_the_new_argument():
    before = run_agent(build_tool_registry(), run_id=1, site_id=1)
    after = run_agent(build_tool_registry(check_rules_tool=None), run_id=1, site_id=1)
    assert after.model_dump(exclude={"steps"}) == before.model_dump(exclude={"steps"})
    assert before.rule_check is not None and any("[mock]" in n for n in before.rule_check.notes)


def test_setup_module_has_no_top_level_import_of_c_module():
    tree = ast.parse((Path(agent_deps.__file__).parent / "check_rules_setup.py").read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [getattr(n, "module", None) or n.names[0].name for n in top_level]
    assert not any("rule_tools" in (name or "") or "integrations" in (name or "") for name in names)


# ── SPRING: C의 도구가 mock을 대신한다 ──

def test_spring_registers_c_tool_in_place_of_mock():
    handler_and_parameters = build_check_rules_tool(spring_settings())
    assert handler_and_parameters is not None
    registry = build_tool_registry(check_rules_tool=handler_and_parameters)
    tool = next(t for t in registry.tools() if t.name == "check_rules")
    assert (tool.mock, tool.owner) == (False, "C")
    assert "[mock]" not in tool.description
    assert tool.parameters is CHECK_RULES_PARAMETERS
    assert [t.name for t in registry.tools()] == [t.name for t in build_tool_registry().tools()]   # 다른 도구는 그대로


def test_client_gets_settings_values_and_seven_two_timeouts(monkeypatch):
    seen = {}

    class SpyClient:
        def __init__(self, base_url, token, **kwargs):
            seen.update(base_url=base_url, token=token, **kwargs)

    monkeypatch.setattr("app.integrations.InternalApiClient", SpyClient)
    build_check_rules_tool(spring_settings())
    assert seen == {"base_url": "http://spring:8080", "token": TOKEN, "connect_timeout": 2.0, "read_timeout": 3.0}


def test_get_tool_registry_follows_the_setting(monkeypatch):
    try:
        for tool, mock_expected in (("MOCK", True), ("SPRING", False)):
            settings = make_settings(tool, internal_api_base_url="http://spring:8080", internal_api_token=TOKEN)
            monkeypatch.setattr(agent_deps, "get_settings", lambda s=settings: s)
            agent_deps.get_tool_registry.cache_clear()
            registry = agent_deps.get_tool_registry()
            assert next(t for t in registry.tools() if t.name == "check_rules").mock is mock_expected
    finally:
        agent_deps.get_tool_registry.cache_clear()


def test_spring_without_url_or_token_fails_at_startup():
    with pytest.raises(ValidationError):
        make_settings("SPRING", internal_api_token=TOKEN)
    with pytest.raises(ValidationError):
        make_settings("SPRING", internal_api_base_url="http://spring:8080")


# ── SPRING: 에이전트 전체 (C의 handler + 가짜 Spring) ──

def test_agent_sends_request_values_and_reads_spring_result():
    fake = FakeSpring(ok(HOT_BLOCK))
    out = run_agent(build_tool_registry(check_rules_tool=real_tool(fake)), run_id=345, site_id=7)

    body = fake.requests[0]["body"]
    assert (body["workType"], body["siteId"], body["agentRunId"]) == ("HOT", 7, 345)
    assert "target" not in body                                     # permit_id가 없으면 저장하지 않는 미리 확인
    assert body["input"]["form"] == {"watcher_name": "김철수"}
    assert fake.requests[0]["url"].endswith("/internal/v1/tools/check_rules")
    assert (fake.requests[0]["connect_timeout"], fake.requests[0]["read_timeout"]) == (2, 3)

    assert out.run_status == "DONE"
    assert out.rule_check is not None and out.rule_check.blocked is True
    assert {e.rule_code: e.status for e in out.rule_check.evaluations} == {
        "HOT_EXTINGUISHER": "MISSING", "HOT_FIRE_WATCHER": "PASS", "HOT_LEL_MAX": "MISSING"}
    assert out.approval_gate.can_request is False
    assert any("소화기" in r for r in out.approval_gate.reasons)
    assert not any(i.code == "CONCURRENT_WORK_CONFLICT" for i in out.verification)   # 실제 도구도 충돌은 아직 빈 목록
    assert any(s.tool_name == "check_rules" and s.status == "OK" for s in out.steps)


def test_single_permit_id_is_passed_so_spring_saves_the_judgment():
    fake = FakeSpring(ok(HOT_BLOCK))
    run_agent(build_tool_registry(check_rules_tool=real_tool(fake)), run_id=345, site_id=7, permit_id=12)
    assert fake.requests[0]["body"]["target"] == {"type": "PERMIT", "id": 12}
    assert len(fake.requests) == 1                                  # 저장 호출은 재시도 없음


def test_request_without_ids_is_blocked_and_never_reaches_spring():
    fake = FakeSpring()
    out = run_agent(build_tool_registry(check_rules_tool=real_tool(fake)))
    assert fake.requests == []
    assert out.run_status == "DONE"
    assert any(i.code == "RULE_CHECK_UNAVAILABLE" for i in out.verification)
    assert out.approval_gate.can_request is False


def test_spring_outage_continues_but_blocks_approval():
    fake = FakeSpring(timeout(), timeout())                         # 미리 확인 호출은 1회 재시도
    out = run_agent(build_tool_registry(check_rules_tool=real_tool(fake)), run_id=345, site_id=7)
    assert len(fake.requests) == 2
    assert out.run_status == "DONE"
    assert any(i.code == "RULE_CHECK_UNAVAILABLE" for i in out.verification)
    assert out.approval_gate.can_request is False
    assert any(s.tool_name == "check_rules" and s.status == "ERROR" for s in out.steps)
