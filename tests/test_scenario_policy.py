"""시나리오 판단 규칙(app/agent/scenario/scenario_policy.py) 시험. 모델은 부르지 않고 가짜 제공자를 쓴다.

검증 범위: 가짜 제공자·가짜 처리기 + 전체 app 코드 실행 (실제 모델의 분류 품질은 확인하지 않는다).
"""

from __future__ import annotations

import pytest

pytest.importorskip("app.agent.scenario.scenario_dispatch")

from app.agent.scenario import scenario_policy  # noqa: E402
from app.agent.scenario.scenario_classifier import ScenarioClassifier  # noqa: E402
from app.agent.scenario.scenario_dispatch import (  # noqa: E402
    PENDING_TEXT_KEY,
    ScenarioDispatcher,
    ScenarioRequest,
)
from app.agent.scenario.scenario_policy import decide  # noqa: E402
from app.agent.scenario.scenario_router import Scenario  # noqa: E402
from app.core import Settings  # noqa: E402
from app.llm.llm_base import LLMClient  # noqa: E402
from app.llm.llm_provider import LLMClientProvider  # noqa: E402
from app.llm.llm_types import LLMResponse  # noqa: E402

A, E, F, D = Scenario.WORK_REGISTER, Scenario.AUDIT_QUERY, Scenario.GUIDE, Scenario.DASHBOARD
ENTRY = (A, E, F)


# ── 규칙 단위 ──


def test_guide_mode_goes_to_f_without_reading_the_text():
    for text in ("지난달 감사 이력 보여줘", "오늘 13시에 화기작업 등록해줘", "아무 말"):
        d = decide(text, mode="GUIDE", allowed=ENTRY)
        assert d.scenario is F and d.source == "explicit" and d.confidence == 1.0
        assert "GUIDE" in d.reason and d.ask is None


def test_guide_mode_outside_chat_scope_redirects():
    d = decide("안녕", mode="GUIDE", allowed=(A, E))
    assert d.scenario is Scenario.UNKNOWN and d.redirect is F and d.ask


@pytest.mark.parametrize("text", [
    "지난달 감사 대응용 실시율 통계 보여줘",
    "지난주 화기작업 기록 조회",       # 작업 단어가 있어도 E 키워드가 있고 A의 강한 단서가 없으면 E
    "어제 허가 이력 알려줘",
    "지난달 화기작업 알려줘",           # 시점 단어(지난달)만으로도 E 쪽 — 작업 단어는 약한 단서일 뿐이다
])
def test_e_keywords_choose_e(text):
    d = decide(text, mode="VERIFY", allowed=ENTRY)
    assert d.scenario is E and d.source == "fallback" and d.confidence == scenario_policy.RULE_CONFIDENCE
    assert "E" in d.reason


@pytest.mark.parametrize("text", [
    "오늘 13시에 화기작업 있어. 필요한 거 만들어줘",
    "내일 배관 작업 등록할게요",
    "3호기 배관 플랜지 교체해야 해",          # 약한 단서(작업 단어)만 있어도 E 키워드가 없으면 A
    "2시부터 5시까지 용접합니다",
])
def test_a_keywords_choose_a(text):
    d = decide(text, mode="VERIFY", allowed=ENTRY)
    assert d.scenario is A and d.source == "fallback" and d.ask is None


@pytest.mark.parametrize("text", [
    "어제 하던 용접 이어서 내일 오전 9시에 등록",   # E 키워드(어제) + A 강한 단서(등록·시각)
    "지난주 14시에 한 작업 알려줘",                  # E 키워드(지난주) + 작업 시각(A의 강한 단서)
    "안녕하세요",                                   # 키워드 없음
    "오늘 마감 현황 알려줘",                         # D 성격 — 입구 밖이라 A로 끌려가지 않고 되묻는다
])
def test_unclear_text_is_asked_back_without_llm(text):
    d = decide(text, mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.source == "fallback" and d.ask
    assert "작업 등록" in d.ask and "기록 조회" in d.ask
    assert "안내" not in d.ask  # F는 현장 모드로 정해지므로 VERIFY 사용자에게 고르게 하지 않는다


def test_mode_none_behaves_like_verify():
    assert decide("지난달 통계", mode=None, allowed=ENTRY).scenario is E
    assert decide("오늘 13시 작업", mode=None, allowed=ENTRY).scenario is A


def test_scope_redirect_for_e_outside_chat_scope():
    d = decide("지난달 감사 이력", mode="VERIFY", allowed=(A, F))
    assert d.scenario is Scenario.UNKNOWN and d.redirect is E and d.ask


def test_empty_text_is_source_empty():
    d = decide("   ", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.source == "empty" and d.ask


def test_policy_disabled_decides_nothing(monkeypatch):
    monkeypatch.setattr(scenario_policy, "POLICY_ENABLED", False)
    assert decide("지난달 통계", mode="VERIFY", allowed=ENTRY) is None
    assert decide("아무 말", mode="GUIDE", allowed=ENTRY) is None


def test_use_llm_hands_only_unclear_text_back(monkeypatch):
    monkeypatch.setattr(scenario_policy, "POLICY_USE_LLM", True)
    assert decide("안녕하세요", mode="VERIFY", allowed=ENTRY) is None
    assert decide("지난달 통계", mode="VERIFY", allowed=ENTRY).scenario is E
    assert decide("안녕하세요", mode="GUIDE", allowed=ENTRY).scenario is F  # 모드로 정한 F는 그대로


# ── 분류기·디스패처 (가짜 제공자: 모델을 만들면 실패한다) ──


class _Json(LLMClient):
    name = "J"

    def __init__(self, text: str) -> None:
        super().__init__(None)
        self.text = text
        self.calls = 0

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        self.calls += 1
        return LLMResponse(text=self.text, stop_reason="end")


def _provider(client: LLMClient | None = None) -> tuple[LLMClientProvider, list[str]]:
    built: list[str] = []

    def factory(engine, model_id):
        built.append(engine)
        if client is None:
            raise AssertionError(f"규칙이 정한 요청에서 모델을 만들었습니다: {engine}")
        return client

    settings = Settings(network_env="OPEN", check_rules_tool="MOCK")
    return LLMClientProvider(settings, {}, factory=factory), built


class _Retriever:
    def retrieve(self, text):  # pragma: no cover - 불리면 실패
        raise AssertionError("규칙이 정한 요청에서 참고 자료 검색을 했습니다")


def _dispatcher(provider, *, chat=ENTRY, retriever=None):
    calls: list[tuple[str, str]] = []

    def handler_for(code):
        def handler(request, llm):
            calls.append((code, request.message))
            return {"handled_by": code}

        return handler

    classifier = ScenarioClassifier(provider, retriever, chat_scenarios=chat)
    d = ScenarioDispatcher(provider, chat_scenarios=chat, classifier=classifier)
    d.register(A, handler_for("A"))
    d.register(F, handler_for("F"))
    return d, calls


def _req(message, mode="VERIFY", **extra):
    return ScenarioRequest(message=message, payload={"mode": mode, **extra})


def test_guide_mode_dispatches_to_f_handler_without_the_router_model():
    provider, built = _provider(_Json("x"))  # 처리기용 모델은 만들어질 수 있다 (F의 엔진)
    d, calls = _dispatcher(provider, retriever=_Retriever())
    result = d.dispatch(_req("지난달 감사 이력 보여줘", mode="GUIDE"))
    assert result.handled and result.decision.scenario is F and calls == [("F", "지난달 감사 이력 보여줘")]
    assert "router" not in built  # 분류용 모델(router)은 만들지 않았다


def test_verify_a_text_dispatches_to_a_and_never_builds_the_router_model():
    provider, built = _provider(_Json("x"))
    d, calls = _dispatcher(provider, retriever=_Retriever())
    result = d.dispatch(_req("오늘 13시에 화기작업 있어. 필요한 거 만들어줘"))
    assert result.handled and result.decision.scenario is A and result.decision.source == "fallback"
    assert calls and calls[0][0] == "A"
    assert "router" not in built


def test_e_text_is_not_handled_and_builds_no_model():
    provider, built = _provider()  # 모델을 만들면 실패
    d, calls = _dispatcher(provider, retriever=_Retriever())
    result = d.dispatch(_req("지난달 감사 대응 통계"))
    assert result.decision.scenario is E and result.handled is False and result.ask is None
    assert calls == [] and built == []


def test_unclear_text_asks_back_and_the_answer_is_joined_and_decided_again():
    provider, built = _provider()
    d, calls = _dispatcher(provider, retriever=_Retriever())
    first = d.dispatch(_req("도와주세요"))
    assert first.ask and first.decision.scenario is Scenario.UNKNOWN and built == []
    provider2, built2 = _provider(_Json("x"))
    d2, calls2 = _dispatcher(provider2, retriever=_Retriever())
    second = d2.dispatch(_req("화기 작업 등록이요", **{PENDING_TEXT_KEY: "도와주세요"}))
    assert second.handled and second.decision.scenario is A
    assert calls2[0][1] == "도와주세요\n화기 작업 등록이요"  # 처리기는 앞선 말까지 이어 붙인 문장을 받는다
    assert "router" not in built2


def test_in_progress_and_explicit_requests_skip_the_policy(monkeypatch):
    monkeypatch.setattr(scenario_policy, "decide", lambda *a, **k: pytest.fail("규칙을 거치면 안 되는 요청입니다"))
    provider, _ = _provider(_Json("x"))
    d, calls = _dispatcher(provider)
    d.dispatch(ScenarioRequest(message="3명", scenario=A, payload={"mode": "VERIFY"}))
    assert calls[0][0] == "A"


def test_classification_api_path_has_no_mode_and_uses_the_same_rules():
    provider, built = _provider()
    c = ScenarioClassifier(provider, _Retriever(), chat_scenarios=ENTRY)
    r = c.classify("지난달 통계 보여줘")  # mode 없음 (분류 API)
    assert r.decision.scenario is E and r.rag_used is False and built == []
    assert r.scenario_model is not None and r.scenario_model.engine == "audit_query"
    unknown = c.classify("안녕하세요")
    assert unknown.decision.scenario is Scenario.UNKNOWN and unknown.scenario_model is None


def test_stub_classifier_without_mode_parameter_still_works():
    """classify(text)만 받는 가짜 분류기(시험용)도 모드가 payload에 없으면 이전처럼 불린다."""
    from app.agent.scenario.scenario_classifier import ClassificationResult
    from app.agent.scenario.scenario_router import ScenarioDecision

    class Stub:
        def classify(self, text):
            dec = ScenarioDecision(A, 0.9, "stub", "llm")
            return ClassificationResult(dec, None, None, False)

    provider, _ = _provider(_Json("x"))
    d = ScenarioDispatcher(provider, chat_scenarios=ENTRY, classifier=Stub())
    d.register(A, lambda req, llm: "ok")
    assert d.dispatch(ScenarioRequest(message="아무 말")).handled


# ── 끄면 이전 방식 ──


def test_policy_off_goes_back_to_the_router_llm(monkeypatch):
    monkeypatch.setattr(scenario_policy, "POLICY_ENABLED", False)
    llm = _Json('{"scenario": "A", "confidence": 0.9, "reason": "r"}')
    provider, built = _provider(llm)
    d, calls = _dispatcher(provider)
    result = d.dispatch(_req("지난달 감사 이력"))  # 규칙이면 E였을 말
    assert result.decision.scenario is A and result.decision.source == "llm" and llm.calls == 1
    assert "router" in built


def test_use_llm_flag_sends_unclear_text_to_the_router_llm(monkeypatch):
    monkeypatch.setattr(scenario_policy, "POLICY_USE_LLM", True)
    llm = _Json('{"scenario": "E", "confidence": 0.9, "reason": "r"}')
    provider, built = _provider(llm)
    d, _ = _dispatcher(provider)
    result = d.dispatch(_req("안녕하세요"))
    assert result.decision.scenario is E and result.decision.source == "llm" and llm.calls == 1


# ── 적용 스크립트 (apply_scenario_policy.py) ──


def _load_apply_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "apply_scenario_policy.py"
    if not path.is_file():
        pytest.skip("apply_scenario_policy.py가 저장소 루트에 없습니다")
    spec = importlib.util.spec_from_file_location("apply_scenario_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _classifier_text(script) -> str:
    return "import x\n" + script.C_IMPORT_OLD + "\nclass C:\n" + script.C_SIG_OLD + '        """doc"""\n' + script.C_BODY_OLD + "        return 1\n"


def _dispatch_text(script) -> str:
    return "import y\n\ndef f():\n" + script.D_OLD  # 마지막 줄이라 줄바꿈이 없다


@pytest.fixture
def env(tmp_path, monkeypatch):
    script = _load_apply_script()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app" / "agent" / "scenario").mkdir(parents=True)
    return script, tmp_path / "app" / "agent" / "scenario"


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_apply_script_apply_is_idempotent_and_revert_restores_bytes(env, newline):
    script, folder = env
    c, d = folder / "scenario_classifier.py", folder / "scenario_dispatch.py"
    oc = _classifier_text(script).replace("\n", newline).encode("utf-8")
    od = _dispatch_text(script).replace("\n", newline).encode("utf-8")
    c.write_bytes(oc)
    d.write_bytes(od)
    assert script.main([]) == 0
    assert b"scenario_policy" in c.read_bytes() and b"scenario_policy" in d.read_bytes()
    assert (b"\r\n" in c.read_bytes()) == (newline == "\r\n")
    patched = (c.read_bytes(), d.read_bytes())
    assert script.main([]) == 0 and (c.read_bytes(), d.read_bytes()) == patched
    assert script.main(["--revert"]) == 0
    assert (c.read_bytes(), d.read_bytes()) == (oc, od)


def test_apply_script_changes_neither_file_if_one_anchor_is_missing(env):
    script, folder = env
    c, d = folder / "scenario_classifier.py", folder / "scenario_dispatch.py"
    c.write_bytes(_classifier_text(script).encode("utf-8"))
    d.write_bytes(b"def f():\n    return 1\n")
    before = (c.read_bytes(), d.read_bytes())
    assert script.main([]) == 1
    assert (c.read_bytes(), d.read_bytes()) == before


def test_apply_script_refuses_a_half_applied_state(env):
    script, folder = env
    c, d = folder / "scenario_classifier.py", folder / "scenario_dispatch.py"
    c.write_bytes(_classifier_text(script).encode("utf-8"))
    d.write_bytes(_dispatch_text(script).encode("utf-8"))
    assert script.main([]) == 0
    d.write_bytes(_dispatch_text(script).encode("utf-8"))  # 디스패처만 원래대로
    assert script.main([]) == 1


def test_apply_script_stops_outside_repo_root(env):
    script, folder = env
    assert not (folder / "scenario_classifier.py").exists()
    assert script.main([]) == 2
