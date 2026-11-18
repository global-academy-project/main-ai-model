from app.near_miss.schemas import NearMissTagRequest
from app.near_miss.service import tag_near_miss
from app.llm.llm_fake import FakeLLMClient


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
    assert response.tagging_version == "nm-1"
    assert {"axis": "SEVERITY", "value": "NEAR_MISS"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "ACCIDENT_TYPE", "value": "FALL"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
    assert {"axis": "WORK_TYPE", "value": "HEIGHT"} in [
        {"axis": tag.axis, "value": tag.value} for tag in response.tags
    ]
