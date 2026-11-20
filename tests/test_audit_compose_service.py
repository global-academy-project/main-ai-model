"""감사 대응 보고서 구성 서비스: FastAPI 없이 호출, 문장 모델 선택·실패 처리, 라우터가 얇아졌는지.

`.env`·터미널 환경에 기대지 않는다 (라우터 시험은 필요한 값을 안에서 정하고 설정 캐시를 비운다).
"""

import ast
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.audit_compose import compose_audit_report
from app.core import get_settings
from app.schemas.audit_compose import AuditComposeInput
from app.services import audit_compose_service as service

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
        },
        {
            "queryType": "blocked_records",
            "title": "차단 기록",
            "stats": {"count": 1},
        },
    ],
}


def body() -> AuditComposeInput:
    return AuditComposeInput.model_validate(REQUEST)


def dump(result) -> dict:
    data = result.model_dump(mode="json", by_alias=True)
    data["meta"]["generatedAt"] = None  # 만든 시각만 매번 다르다
    return data


def all_sources(data: dict) -> set[str]:
    return {s["source"] for g in data["groups"] for i in g["items"] for s in i["sentences"]}


# ── 서비스 (FastAPI 없이) ─────────────────────────────────────


def test_disabled_never_touches_the_model():
    asked = []  # 예외를 던지면 서비스가 삼키므로, 호출 여부를 따로 기록한다
    response = service.compose_report(body(), narrative_llm_enabled=False, get_llm=lambda: asked.append(1))
    assert asked == []
    assert dump(response.data) == dump(compose_audit_report(body()))
    assert all_sources(dump(response.data)) == {"TEMPLATE"}


def test_model_failure_falls_back_to_template_and_logs_only_the_kind(caplog):
    def broken():
        raise RuntimeError("비밀 설정값")

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        response = service.compose_report(body(), narrative_llm_enabled=True, get_llm=broken)
    assert dump(response.data) == dump(compose_audit_report(body()))
    messages = [r.getMessage() for r in caplog.records if r.name == service.logger.name]
    assert len(messages) == 1
    assert "RuntimeError" in messages[0] and "비밀 설정값" not in messages[0]


def test_enabled_builds_the_generator_from_the_model_once(monkeypatch):
    llm = object()
    asked = []
    seen = {}
    real = service.compose_audit_report

    def spy(request, *, narrative_generator=None):
        seen["generator"] = narrative_generator
        return real(request)

    monkeypatch.setattr(service, "compose_audit_report", spy)
    service.compose_report(body(), narrative_llm_enabled=True, get_llm=lambda: asked.append(1) or llm)

    generator = seen["generator"]
    assert callable(generator)
    got = []
    monkeypatch.setattr(service, "generate_sentence_candidates", lambda item, model: got.append((item, model)) or ["후보"])
    assert generator("항목 1") == ["후보"]
    assert generator("항목 2") == ["후보"]
    assert got == [("항목 1", llm), ("항목 2", llm)]
    assert asked == [1]  # 항목마다 모델을 다시 얻지 않는다


def test_build_generator_returns_none_when_the_model_is_missing():
    def missing():
        raise KeyError("audit_query")

    assert service.build_narrative_generator(missing) is None


# ── 구조: 서비스는 FastAPI를 모르고, 라우터는 얇다 ─────────────


def source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def imported_modules(module) -> set[str]:
    tree = ast.parse(source_of(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_service_does_not_know_fastapi():
    assert not {m for m in imported_modules(service) if m.split(".")[0] in {"fastapi", "starlette"}}


def test_router_only_wires_the_service():
    from app.api.v1.agent import audit_compose as router_module

    text = source_of(router_module)
    assert "from app.services.audit_compose_service import" in text
    for moved in ("compose_audit_report", "generate_sentence_candidates", "logging"):
        assert moved not in text, f"라우터에 {moved}가 남아 있음 (서비스로 옮긴 것)"


# ── 라우터 (모델 제공자가 없는 앱에서도) ───────────────────────


def router_client(monkeypatch, tmp_path, **env) -> TestClient:
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않게
    monkeypatch.delenv("AUDIT_NARRATIVE_LLM", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.api.v1.agent.audit_compose import router

    app = FastAPI()
    app.include_router(router)  # llm_provider를 일부러 붙이지 않는다
    return TestClient(app)


def test_router_default_needs_no_model_provider(monkeypatch, tmp_path):
    response = router_client(monkeypatch, tmp_path).post("/api/v1/agent/audit-report", json=REQUEST)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["formType"] == "AUDIT_REPORT"
    assert all_sources(data) == {"TEMPLATE"}


def test_router_with_llm_on_but_no_provider_still_builds_the_report(monkeypatch, tmp_path):
    client = router_client(monkeypatch, tmp_path, AUDIT_NARRATIVE_LLM="true")
    response = client.post("/api/v1/agent/audit-report", json=REQUEST)
    assert response.status_code == 200, response.text
    assert all_sources(response.json()["data"]) == {"TEMPLATE"}
