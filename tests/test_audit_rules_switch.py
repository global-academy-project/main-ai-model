"""구성기 라우터의 모델 설정 오류 대응과 기록 질문 규칙 스위치(RECORD_PLAN_RULES).

기준 app.zip(11/20 11:39)은 /audit-report를 구성기가 맡는다. 끄면(기본) 기록 질문은 D의 의존성 그대로.
`.env`·터미널 환경에 기대지 않는다 (필요한 값을 시험 안에서 정하고 설정 캐시를 비운다).
"""

import logging

import pytest
from fastapi.testclient import TestClient

from app.core import get_settings

REQUEST = {
    "reportId": 12,
    "siteId": 3,
    "dateFrom": "2026-10-01",
    "dateTo": "2026-10-31",
    "items": [
        {
            "queryType": "risk_assessment_rate",
            "title": "위험성평가 실시율",
            "stats": {"works": 40, "written": 38, "approved": 36, "missing": 2, "rate": 95.0, "approvedRate": 90.0},
        }
    ],
}


def make_app(monkeypatch, tmp_path, **env):
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않게
    base = {
        "CHECK_RULES_TOOL": "MOCK",
        "MSDS_TOOL": "MOCK",
        "NETWORK_ENV": "OPEN",
        "LLM_PRELOAD": "false",
        "RAG_ENABLED": "false",
        "SCENARIO_ENGINES": '{"default": {"engine": "FAKE"}}',
        "INTERNAL_API_BASE_URL": "http://internal.invalid",
        "INTERNAL_API_TOKEN": "test-token",
    }
    for key in ("RECORD_PLAN_RULES", "AUDIT_NARRATIVE_LLM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("KOSHA_MSDS_SERVICE_KEY", raising=False)
    for key, value in {**base, **env}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import create_app

    return create_app()


def endpoint_names(app, path):
    """경로를 맡은 처리 함수 이름들 (FastAPI 버전과 무관하게 openapi의 operationId에서 읽는다)."""
    operations = app.openapi()["paths"].get(path, {})
    return [op["operationId"].split("_api_v1_")[0] for op in operations.values()]


# ── /audit-report (구성기) ────────────────────────────────────


def test_audit_report_is_served_by_composer(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path)
    assert endpoint_names(app, "/api/v1/agent/audit-report") == ["audit_report_compose"]
    client = TestClient(app)
    for path in ("/api/v1/agent/audit-report",):  # `/compose` 호환 별칭은 없어짐
        response = client.post(path, json=REQUEST)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert response.json()["success"] is True
        assert data["formType"] == "AUDIT_REPORT"
        assert data["meta"]["reportId"] == 12
        assert data["groups"][0]["docNo"] == "AR-3-202610-01"


def test_accepts_d_style_body_with_data_snapshot(monkeypatch, tmp_path):
    client = TestClient(make_app(monkeypatch, tmp_path))
    body = {**REQUEST, "dataSnapshot": {"risk_assessment_rate": {"stats": {"works": 40}}}}
    response = client.post("/api/v1/agent/audit-report", json=body)
    assert response.status_code == 200, response.text
    assert "dataSnapshot" not in response.text


def test_bad_request_uses_common_error_shape(monkeypatch, tmp_path):
    client = TestClient(make_app(monkeypatch, tmp_path))
    response = client.post("/api/v1/agent/audit-report", json={**REQUEST, "reportId": "12"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_model_config_error_does_not_fail_the_report(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, AUDIT_NARRATIVE_LLM="true")

    class Broken:
        def get(self, key):
            raise RuntimeError("모델 설정 오류")

    app.state.llm_provider = Broken()
    response = TestClient(app).post("/api/v1/agent/audit-report", json=REQUEST)
    assert response.status_code == 200, response.text
    sentences = [
        s for g in response.json()["data"]["groups"] for i in g["items"] for s in i["sentences"]
    ]
    assert sentences and all(s["source"] == "TEMPLATE" for s in sentences)


# ── RECORD_PLAN_RULES ─────────────────────────────────────────


def test_record_plan_rules_default_keeps_d_dependency(monkeypatch, tmp_path):
    from app.api.v1.agent.record_query_deps import get_record_query_agent

    app = make_app(monkeypatch, tmp_path)
    assert get_settings().record_plan_rules is False
    assert get_record_query_agent not in app.dependency_overrides


def _fake_search(arguments):
    return {
        "queryType": arguments["query_type"],
        "conditions": {"siteId": arguments["site_id"]},
        "total": 0,
        "truncated": False,
        "records": [],
        "evidence": [],
    }


def test_record_plan_rules_on_uses_rules_and_falls_back(monkeypatch, tmp_path, caplog):
    from app.api.v1.agent import record_query_deps
    from app.api.v1.agent.record_query_deps import get_record_query_agent

    app = make_app(monkeypatch, tmp_path, RECORD_PLAN_RULES="true")
    assert get_record_query_agent in app.dependency_overrides
    seen = []
    monkeypatch.setattr(
        record_query_deps,
        "make_search_records_tool",
        lambda client: (lambda arguments: (seen.append(dict(arguments)), _fake_search(arguments))[1]),
    )
    client = TestClient(app)
    body = {"runId": 7, "siteId": 2, "mode": "VERIFY", "message": "2026-10-01부터 2026-10-31까지 위험성평가 실시 현황"}
    with caplog.at_level(logging.INFO):
        response = client.post("/api/v1/agent/record-query", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["runStatus"] == "DONE"
    assert seen[0]["query_type"] == "risk_assessment_rate"
    assert seen[0]["site_id"] == 2 and seen[0]["run_id"] == 7
    assert any("기록 질문 계획: 규칙" in r.getMessage() for r in caplog.records)

    # 규칙이 못 정하는 질문은 D의 모델 계획으로 넘어간다 (가짜 모델은 JSON을 안 줘서 되묻기가 된다)
    seen.clear()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        response = client.post("/api/v1/agent/record-query", json={**body, "message": "승인자별 처리 통계"})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["runStatus"] == "WAITING_INPUT"
    assert seen == []
    assert any("모델 계획으로 넘김" in r.getMessage() for r in caplog.records)
