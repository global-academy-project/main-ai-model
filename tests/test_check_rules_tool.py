"""check_rules 실제 도구: B의 mock 입출력 그대로 Spring 규칙 엔진을 부르는지."""

import pytest

from app.agent.rule_tools import make_check_rules
from app.exceptions import ToolError
from app.integrations import InternalApiClient
from app.schemas.agent_work import RuleEvaluationDto

from .fakes import FakeSpring, check, error, ok, result, timeout


def tool(fake: FakeSpring):
    return make_check_rules(InternalApiClient("http://spring:8080", "t" * 32,
                                              connect_timeout=2, read_timeout=3, transport=fake))


CONFINED_BLOCK = result("BLOCK", [
    check("CONFINED_CO2_MAX", "measurement.CO2", "PASS", actual="0.8", expected="1.5 미만 (%)"),
    check("CONFINED_LEL_MAX", "measurement.LEL", "MISSING", expected="10 미만 (%LEL)", message="측정값이 없습니다."),
    check("CONFINED_O2_RANGE", "measurement.O2", "FAIL", actual="17.5", expected="18 이상 23.5 미만 (%)",
          message="산소농도가 적정공기 기준(18% 이상 23.5% 미만)을 벗어났습니다.",
          legal_basis="산업안전보건기준에 관한 규칙 제618조"),
])


def test_scenario_b_request_and_evaluations():
    fake = FakeSpring(ok(CONFINED_BLOCK))
    out = tool(fake)({
        "permit_types": ["CONFINED_SPACE_PERMIT"],
        "measurements": {"O2": 17.5, "CO2": 0.8},
        "assignments": {"FIRE_WATCHER": None, "ATTENDANT": "박감시"},
        "site_id": 1, "run_id": 345,
    })

    body = fake.requests[0]["body"]
    assert body == {"siteId": 1, "workType": "CONFINED", "agentRunId": 345,
                    "input": {"measurement": {"O2": 17.5, "CO2": 0.8}, "form": {"watcher_name": "박감시"}}}
    assert "target" not in body                          # permit_id 없으면 미리 확인 (저장 안 함)

    assert out["blocked"] is True
    by_code = {e["rule_code"]: e for e in out["evaluations"]}
    o2 = by_code["CONFINED_O2_RANGE"]
    assert (o2["status"], o2["item"], o2["unit"], o2["value"]) == ("FAIL", "O2", "PCT", 17.5)
    assert o2["criterion"] == "18 이상 23.5 미만 (%)"
    assert o2["message"].startswith("산소농도가")        # Spring 문구 그대로
    assert o2["basis"] == "산업안전보건기준에 관한 규칙 제618조"
    assert o2["title"] == "산소 농도"
    assert by_code["CONFINED_LEL_MAX"]["status"] == "MISSING"
    assert by_code["CONFINED_LEL_MAX"]["value"] is None
    co2 = by_code["CONFINED_CO2_MAX"]
    assert co2["status"] == "PASS" and "충족" in co2["message"]
    # B의 응답 스키마로 그대로 읽힌다
    for e in out["evaluations"]:
        RuleEvaluationDto.model_validate(e)


def test_hot_watcher_and_check_items():
    fake = FakeSpring(ok(result("BLOCK", [
        check("HOT_EXTINGUISHER", "form.extinguisher_ready", "PASS", expected="확인 필요", category="REQUIRED_FIELD"),
        check("HOT_FIRE_WATCHER", "form.watcher_name", "PASS", actual="김감시", expected="입력 필요", category="STAFFING"),
        check("HOT_LEL_MAX", "measurement.LEL", "PASS", actual="3", expected="10 미만 (%LEL)"),
        check("HOT_SPARK_PREVENTION", "form.spark_prevention", "MISSING", expected="확인 필요",
              message="불티 비산 방지 조치를 확인해야 합니다.", category="REQUIRED_FIELD"),
    ], evaluation_id=77)))
    out = tool(fake)({
        "permit_types": ["HOT_WORK_PERMIT"],
        "measurements": {"LEL": 3},
        "assignments": {"FIRE_WATCHER": " 김감시 ", "ATTENDANT": "무시됨"},
        "form": {"extinguisher_ready": True, "spark_prevention": None},
        "site_id": 1, "permit_id": 12,
    })

    body = fake.requests[0]["body"]
    assert body["workType"] == "HOT"
    assert body["target"] == {"type": "PERMIT", "id": 12}
    assert body["input"]["form"] == {"extinguisher_ready": True, "watcher_name": "김감시"}   # None 은 빼고 보냄 → MISSING

    by_code = {e["rule_code"]: e for e in out["evaluations"]}
    watcher = by_code["HOT_FIRE_WATCHER"]
    assert (watcher["role"], watcher["assignee"], watcher["message"]) == ("FIRE_WATCHER", "김감시", "화재감시자: 김감시")
    assert "item" not in by_code["HOT_EXTINGUISHER"] and "role" not in by_code["HOT_EXTINGUISHER"]
    assert by_code["HOT_EXTINGUISHER"]["message"] == "소화기 비치: 확인됨."
    assert by_code["HOT_SPARK_PREVENTION"]["message"] == "불티 비산 방지 조치를 확인해야 합니다."
    assert out["blocked"] is True
    assert out["evaluation_ids"] == [77]


def test_missing_watcher_has_no_assignee():
    fake = FakeSpring(ok(result("BLOCK", [
        check("HOT_FIRE_WATCHER", "form.watcher_name", "MISSING", expected="입력 필요", category="STAFFING",
              message="화재감시자를 지정해야 합니다."),
    ])))
    out = tool(fake)({"permit_types": ["HOT_WORK_PERMIT"], "assignments": {"FIRE_WATCHER": ""}, "site_id": 1})
    assert fake.requests[0]["body"]["input"]["form"] == {}
    e = out["evaluations"][0]
    assert (e["role"], e["assignee"], e["status"]) == ("FIRE_WATCHER", None, "MISSING")


def test_no_rules_yet_is_noted_not_blocked():
    fake = FakeSpring(ok(result("PASS", [])))
    out = tool(fake)({"permit_types": ["HEIGHT_WORK_PERMIT"], "site_id": 1})
    assert fake.requests[0]["body"]["workType"] == "HEIGHT"
    assert out["blocked"] is False
    assert out["evaluations"] == []
    assert any("HEIGHT_WORK_PERMIT" in n for n in out["notes"])


def test_two_permit_types_call_twice():
    fake = FakeSpring(ok(result("PASS", [check("HOT_LEL_MAX", "measurement.LEL", "PASS", actual="0")])),
                      ok(CONFINED_BLOCK))
    out = tool(fake)({"permit_types": ["HOT_WORK_PERMIT", "CONFINED_SPACE_PERMIT"], "measurements": {"LEL": 0},
                      "site_id": 1})
    assert [r["body"]["workType"] for r in fake.requests] == ["HOT", "CONFINED"]
    assert out["blocked"] is True
    assert len(out["evaluations"]) == 4


@pytest.mark.parametrize("arguments", [
    {"permit_types": ["HOT_WORK_PERMIT"]},                                         # site_id 없음
    {"permit_types": ["HOT_WORK_PERMIT"], "site_id": "1"},                         # 문자열
    {"permit_types": ["HOT_WORK_PERMIT"], "site_id": True},
    {"permit_types": ["LIFTING_PERMIT"], "site_id": 1},                            # 모르는 허가
    {"permit_types": "HOT_WORK_PERMIT", "site_id": 1},
    {"permit_types": ["HOT_WORK_PERMIT", "CONFINED_SPACE_PERMIT"], "site_id": 1, "permit_id": 3},
])
def test_input_invalid_before_calling_spring(arguments):
    fake = FakeSpring()
    with pytest.raises(ToolError) as exc:
        tool(fake)(arguments)
    assert exc.value.code == "TOOL_INPUT_INVALID"
    assert fake.requests == []


def test_saving_call_is_not_retried():
    fake = FakeSpring(timeout(), ok(CONFINED_BLOCK))
    with pytest.raises(ToolError) as exc:
        tool(fake)({"permit_types": ["CONFINED_SPACE_PERMIT"], "site_id": 1, "permit_id": 9})
    assert exc.value.code == "TOOL_UNAVAILABLE"
    assert len(fake.requests) == 1


def test_preview_call_is_retried_once():
    fake = FakeSpring(timeout(), ok(CONFINED_BLOCK))
    out = tool(fake)({"permit_types": ["CONFINED_SPACE_PERMIT"], "site_id": 1})
    assert out["blocked"] is True
    assert len(fake.requests) == 2


def test_spring_errors_pass_through_as_tool_errors():
    with pytest.raises(ToolError) as exc:
        tool(FakeSpring(error(404, "NOT_FOUND")))({"permit_types": ["HOT_WORK_PERMIT"], "site_id": 1, "permit_id": 999})
    assert exc.value.code == "TOOL_TARGET_NOT_FOUND"


def test_old_spring_without_checks_is_unavailable():
    data = result("PASS", [])
    del data["checks"]
    with pytest.raises(ToolError) as exc:
        tool(FakeSpring(ok(data)))({"permit_types": ["HOT_WORK_PERMIT"], "site_id": 1})
    assert exc.value.code == "TOOL_UNAVAILABLE"
