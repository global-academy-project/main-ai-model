from app.near_miss.schemas import NearMissTagRequest
from app.near_miss.service import tag_near_miss
from app.llm.llm_fake import FakeLLMClient, text_response
from app.exceptions import LLMError
from app.llm.llm_base import LLMClient
import json


def test_near_miss_llm_receives_code_descriptions_and_missing_axes_only():
    from app.near_miss.llm import suggest_missing_tags
    from app.near_miss.schemas import NearMissTagSuggestion

    llm = FakeLLMClient([text_response(
        '{"hazard":["ELECTRICAL","CHEMICAL"],"workType":["ELECTRICAL"]}'
    )])
    existing = [
        NearMissTagSuggestion(axis="SEVERITY", value="NEAR_MISS", confidence=1.0),
        NearMissTagSuggestion(axis="ACCIDENT_TYPE", value="ELECTRIC", confidence=0.8),
    ]

    tags, model = suggest_missing_tags(
        llm,
        request_text="젖은 손으로 분전반 차단기를 조작하다 찌릿했다",
        existing=existing,
    )

    prompt = json.loads(llm.calls[0]["messages"][0].content)
    assert prompt["missing_axes"] == ["HAZARD", "WORK_TYPE", "SUBSTANCE"]
    assert "분전반" in prompt["allowed_codes"]["hazard"]["ELECTRICAL"]
    assert "workType" in prompt["allowed_codes"]
    assert "accidentType" not in prompt["allowed_codes"]
    assert "사다리" in prompt["allowed_codes"]["workType"]["HEIGHT"]
    assert model == "fake"
    hazard_tags = [tag for tag in tags if tag.axis == "HAZARD"]
    work_tags = [tag for tag in tags if tag.axis == "WORK_TYPE"]
    assert [tag.value for tag in hazard_tags] == ["ELECTRICAL"]
    assert [tag.value for tag in work_tags] == ["ELECTRICAL"]


def test_near_miss_llm_prompt_requires_direct_work_evidence():
    from app.near_miss.llm import suggest_missing_tags
    from app.near_miss.schemas import NearMissTagSuggestion

    llm = FakeLLMClient([text_response(
        '{"workType":["GENERAL"]}'
    )])
    existing = [
        NearMissTagSuggestion(axis="SEVERITY", value="NEAR_MISS", confidence=1.0),
        NearMissTagSuggestion(axis="ACCIDENT_TYPE", value="FALL", confidence=0.8),
        NearMissTagSuggestion(axis="HAZARD", value="ENVIRONMENTAL", confidence=0.8),
    ]

    suggest_missing_tags(
        llm,
        request_text="탱크 외부에서 누출 상태를 점검했지만 내부에는 진입하지 않았다",
        existing=existing,
    )

    system = llm.calls[0]["system"]
    assert "실제 작업행위가 없으면 해당 작업 유형으로 선택하지 않습니다" in system
    assert "내부 진입 없음" in system
    assert "위 작업이 실제로 확인되지 않으면 GENERAL" in system


def test_near_miss_rule_tagging_uses_scenario_c_shape():
    request = NearMissTagRequest(
        runId=51,
        siteId=3,
        nearMissId=9,
        title="비계 위 공구 낙하",
        description="망치가 떨어져 추락 위험이 발생했다",
        immediateAction="하부 통제",
        workType="HEIGHT",
    )

    response = tag_near_miss(request, FakeLLMClient())

    assert response.tagging_source == "near_miss_rules"
    assert response.tagging_version == "nm-3"
    assert {"axis": "SEVERITY", "value": "NEAR_MISS"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "ACCIDENT_TYPE", "value": "FALL"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "WORK_TYPE", "value": "HEIGHT"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]


class FailingNearMissLLM(LLMClient):
    name = "failing-near-miss"

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        raise LLMError("모델 호출 실패", engine=self.name)


def test_near_miss_llm_error_keeps_rule_tags():
    request = NearMissTagRequest(
        runId=51,
        siteId=3,
        nearMissId=9,
        title="비계 위 공구 낙하",
        description="망치가 떨어져 추락 위험이 발생했다",
        immediateAction="하부 통제",
        workType="HEIGHT",
    )

    response = tag_near_miss(request, FailingNearMissLLM())

    assert response.tagging_source == "near_miss_rules"
    assert response.tagging_version == "nm-3"
    assert response.model is None
    assert {"axis": "SEVERITY", "value": "NEAR_MISS"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "ACCIDENT_TYPE", "value": "FALL"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "WORK_TYPE", "value": "HEIGHT"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]


def test_near_miss_auxiliary_accident_rules_cover_field_expressions():
    request = NearMissTagRequest(
        runId=51,
        siteId=3,
        nearMissId=12,
        title="천장 조명 작업 중 추락 사고",
        description="작업자가 사다리를 이용해 천장 조명 교체 중 발을 헛디뎌 추락할 뻔했다",
        immediateAction="작업 중지와 안전대 체결",
        workType=None,
    )

    response = tag_near_miss(request, FakeLLMClient())
    values = {(tag.axis, tag.value) for tag in response.tags}

    assert ("ACCIDENT_TYPE", "FALL") in values
    assert ("HAZARD", "HUMAN") in values
    assert ("WORK_TYPE", "HEIGHT") in values


def test_near_miss_reactionary_place_name_does_not_force_confined():
    request = NearMissTagRequest(
        runId=51,
        siteId=3,
        nearMissId=36,
        title="탱크 주변 점검",
        description="탱크 외부에서 누출 상태를 점검했지만 내부에는 진입하지 않았다",
        immediateAction="누출 차단과 주변 통제",
        workType=None,
    )

    response = tag_near_miss(request, FakeLLMClient())
    work_types = {tag.value for tag in response.tags if tag.axis == "WORK_TYPE"}

    assert "CONFINED" not in work_types
