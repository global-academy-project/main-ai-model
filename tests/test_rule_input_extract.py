"""문장 → 규칙 입력 LLM 추출: 모델 없이 가짜 클라이언트로."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.agent_deps import get_llm_provider
from app.agent.rule_input_parse import parse_rule_input
from app.api.v1.agent.rule_input_extract import router
from app.core import get_settings
from app.exceptions import LLMError
from app.handlers import llm_error_handler
from app.llm.llm_fake import FakeLLMClient, text_response
from app.schemas.rule_input_extract import RuleInputExtractRequest
from app.services.rule_input_extract_service import extract_rule_input, run_rule_input_extract

ITEMS = ["O2", "CO2", "LEL", "H2S", "CO", "NH3", "CL2"]
TYPES = ["HOT", "CONFINED", "HEIGHT"]
TEXT = "R-201 탱크 들어가는데 산소 측정해 보니까 생각보다 많이 낮게 나와서 17.0%, 화재감시자 김철수"


def reply(obj: dict) -> FakeLLMClient:
    return FakeLLMClient([text_response(json.dumps(obj, ensure_ascii=False))])


def m(item, value, unit, evidence):
    return {"item": item, "value": value, "unit": unit, "evidence": evidence}


# ---- 검사 (parse) ----

def test_keeps_value_with_verbatim_evidence():
    out = parse_rule_input({"measurements": [m("O2", 17.0, "PCT", "낮게 나와서 17.0%")],
                            "workTypes": [{"type": "CONFINED", "evidence": "탱크 들어가는데"}],
                            "watcherName": "김철수", "watcherEvidence": "화재감시자 김철수"}, TEXT, ITEMS, TYPES)
    assert [(x.item, x.value, x.unit) for x in out.measurements] == [("O2", Decimal("17.0"), "PCT")]
    assert [w.type for w in out.work_types] == ["CONFINED"]
    assert (out.watcher_name, out.watcher_evidence) == ("김철수", "화재감시자 김철수")


@pytest.mark.parametrize("bad", [
    m("O2", 17.0, "PCT", "낮게 나와서 17%"),        # 근거가 원문과 다름
    m("O2", 18.0, "PCT", "낮게 나와서 17.0%"),      # 값이 근거에 없음
    m("O2", 17.0, "PPM", "낮게 나와서 17.0%"),      # 항목이 받지 않는 단위
    m("O2", 17.0, "", "낮게 나와서 17.0%"),         # 단위 없음
    m("XX", 17.0, "PCT", "낮게 나와서 17.0%"),      # 모르는 항목
    m("O2", "열일곱", "PCT", "낮게 나와서 17.0%"),  # 숫자 아님
    m("O2", 170, "PCT", "낮게 나와서 17.0%"),       # % 인데 100 초과 (근거 검사 전에 걸림)
    m("O2", True, "PCT", "낮게 나와서 17.0%"),      # bool 은 숫자로 보지 않음
])
def test_drops_unsupported_value(bad):
    out = parse_rule_input({"measurements": [bad]}, TEXT, ITEMS, TYPES)
    assert out.measurements == []
    assert out.dropped


def test_drops_item_not_requested():
    out = parse_rule_input({"measurements": [m("O2", 17.0, "PCT", "낮게 나와서 17.0%")]}, TEXT, ["CO"], TYPES)
    assert out.measurements == []


def test_conflicting_values_drop_both_and_same_value_kept_once():
    text = "산소 17% 다시 재니 산소 19%"
    out = parse_rule_input({"measurements": [m("O2", 17, "PCT", "산소 17%"), m("O2", 19, "PCT", "산소 19%")]},
                           text, ITEMS, TYPES)
    assert out.measurements == []
    out = parse_rule_input({"measurements": [m("O2", 17, "PCT", "산소 17%"), m("O2", 17.0, "PCT", "산소 17%")]},
                           text, ITEMS, TYPES)
    assert len(out.measurements) == 1


def test_lel_accepts_percent_and_lel_unit():
    text = "가연성 가스 5%, 폭발하한 3%LEL"
    out = parse_rule_input({"measurements": [m("LEL", 5, "PCT", "가연성 가스 5%")]}, text, ITEMS, TYPES)
    assert out.measurements[0].unit == "PCT"
    out = parse_rule_input({"measurements": [m("LEL", 3, "PCT_LEL", "폭발하한 3%LEL")]}, text, ITEMS, TYPES)
    assert out.measurements[0].unit == "PCT_LEL"


def test_work_type_and_watcher_need_text_evidence():
    out = parse_rule_input({"workTypes": [{"type": "HOT", "evidence": "용접"}, {"type": "FLY", "evidence": "탱크"}],
                            "watcherName": "이영희", "watcherEvidence": "감시자 이영희"}, TEXT, ITEMS, TYPES)
    assert out.work_types == [] and out.watcher_name is None


def test_not_a_dict_gives_empty():
    assert parse_rule_input(None, TEXT, ITEMS, TYPES).measurements == []
    out = parse_rule_input({"measurements": "O2 17", "workTypes": [1, None]}, TEXT, ITEMS, TYPES)
    assert out.measurements == [] and out.work_types == []


# ---- 호출 (extractor) ----

def test_extractor_sends_prompt_and_parses_think_and_fence():
    raw = "<think>음</think>```json\n" + json.dumps(
        {"measurements": [m("O2", 17.0, "PCT", "낮게 나와서 17.0%")], "workTypes": [],
         "watcherName": None, "watcherEvidence": None}, ensure_ascii=False) + "\n```"
    llm = FakeLLMClient([text_response(raw)])
    out = extract_rule_input(llm, TEXT, ITEMS, TYPES, max_tokens=300)
    assert [x.item for x in out.measurements] == ["O2"]
    call = llm.calls[0]
    assert call["max_tokens"] == 300
    assert TEXT in call["messages"][0].content
    assert "evidence" in call["system"] and "O2" in call["system"]


def test_extractor_non_json_gives_empty():
    out = extract_rule_input(FakeLLMClient([text_response("모르겠습니다")]), TEXT, ITEMS, TYPES)
    assert out.measurements == [] and out.work_types == []


class SchemaClient(FakeLLMClient):
    """B님 GGUF 엔진처럼 json_schema 를 받는 클라이언트."""

    def chat(self, messages, *, system="", tools=None, max_tokens=None, json_schema=None):
        self.schema = json_schema
        return super().chat(messages, system=system, tools=tools, max_tokens=max_tokens)


def test_extractor_passes_json_schema_only_when_engine_accepts_it():
    llm = SchemaClient([text_response("{}")])
    extract_rule_input(llm, TEXT, ["O2"], ["HOT"])
    props = llm.schema["properties"]
    assert props["measurements"]["items"]["properties"]["item"]["enum"] == ["O2"]
    assert props["workTypes"]["items"]["properties"]["type"]["enum"] == ["HOT"]
    extract_rule_input(FakeLLMClient([text_response("{}")]), TEXT, ITEMS, TYPES)   # 받지 않는 엔진은 빼고 부름


def test_extractor_lets_llm_error_through():
    def boom(_messages):
        raise LLMError("timeout")

    with pytest.raises(LLMError):
        extract_rule_input(FakeLLMClient([boom]), TEXT, ITEMS, TYPES)


# ---- 서비스·경로 ----

def test_service_camel_case_response():
    body = RuleInputExtractRequest.model_validate({"text": TEXT, "measurementItems": ["O2"], "workTypes": ["CONFINED"]})
    llm = reply({"measurements": [m("O2", 17.0, "PCT", "낮게 나와서 17.0%")],
                 "workTypes": [{"type": "CONFINED", "evidence": "탱크 들어가는데"}],
                 "watcherName": "김철수", "watcherEvidence": "화재감시자 김철수"})
    dumped = run_rule_input_extract(llm, body).model_dump(mode="json", by_alias=True)
    assert dumped == {"success": True, "data": {
        "measurements": [{"item": "O2", "value": 17.0, "unit": "PCT", "evidence": "낮게 나와서 17.0%"}],
        "workTypes": [{"type": "CONFINED", "evidence": "탱크 들어가는데"}],
        "watcherName": "김철수", "watcherEvidence": "화재감시자 김철수"}}


class FakeProvider:
    def __init__(self, client):
        self.client, self.keys = client, []

    def get(self, key):
        self.keys.append(key)
        return self.client


class FakeSettings:
    ai_max_output_tokens = 512


def client_with(llm) -> tuple[TestClient, FakeProvider]:
    app = FastAPI()
    app.add_exception_handler(LLMError, llm_error_handler)
    app.include_router(router)
    provider = FakeProvider(llm)
    app.dependency_overrides[get_llm_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: FakeSettings()
    return TestClient(app), provider


def test_route_uses_scenario_b_and_returns_spring_shape():
    http, provider = client_with(reply({"measurements": [m("O2", 17.0, "PCT", "낮게 나와서 17.0%")],
                                         "workTypes": [], "watcherName": None, "watcherEvidence": None}))
    res = http.post("/api/v1/extract/rule-input",
                    json={"text": TEXT, "measurementItems": ITEMS, "workTypes": TYPES})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["measurements"][0] == {"item": "O2", "value": 17.0, "unit": "PCT",
                                               "evidence": "낮게 나와서 17.0%"}
    assert body["data"]["workTypes"] == [] and body["data"]["watcherName"] is None
    assert provider.keys == ["B"]
    assert provider.client.calls[0]["max_tokens"] == 512


def test_route_llm_error_is_failure_response():
    def boom(_messages):
        raise LLMError("timeout")

    http, _ = client_with(FakeLLMClient([boom]))
    res = http.post("/api/v1/extract/rule-input", json={"text": TEXT})
    assert res.status_code >= 500
    assert res.json()["success"] is False


def test_route_rejects_unknown_item_and_empty_text():
    http, _ = client_with(FakeLLMClient())
    assert http.post("/api/v1/extract/rule-input", json={"text": TEXT, "measurementItems": ["XX"]}).status_code == 422
    assert http.post("/api/v1/extract/rule-input", json={"text": ""}).status_code == 422
