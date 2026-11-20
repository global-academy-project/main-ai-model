import importlib

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families


def test_ai_metrics_endpoint(monkeypatch):
    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")
    monkeypatch.setenv("MSDS_TOOL", "MOCK")
    monkeypatch.setenv("LLM_PRELOAD", "false")

    metrics_app = importlib.import_module("metrix.main")
    with TestClient(metrics_app.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/unknown/123").status_code == 404
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]
    request_samples = [sample for sample in samples if sample.name == "ai_http_requests_total"]
    assert any(
        sample.labels == {"method": "GET", "path": "/health", "status": "200"}
        and sample.value >= 1
        for sample in request_samples
    )
    assert any(
        sample.labels == {"method": "GET", "path": "unmatched", "status": "404"}
        and sample.value >= 1
        for sample in request_samples
    )
    assert not any(sample.labels["path"] == "/metrics" for sample in request_samples)
