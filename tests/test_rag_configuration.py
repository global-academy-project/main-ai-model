import json
from io import BytesIO
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from app.core.config import RagEnvironmentSettings, Settings
from app.exceptions import ToolError
from app.rag import http
from app.rag.config import RagSettings
from app.tools.adapters.rag_tools import register_rag_tools
from app.tools.rag_loader import load_search_backend
from app.tools.tool_registry import ToolRegistry
from app.tools.tool_setup import build_tool_registry


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in (
        "NETWORK_ENV", "RAG_BACKEND", "EMBEDDING_URL", "EMBEDDING_MODEL",
        "EMBEDDING_DIM", "EMBEDDING_API_KEY", "QDRANT_URL",
        "QDRANT_API_KEY", "QDRANT_READ_ONLY_API_KEY", "RAG_SEARCH_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_existing_settings_still_default_to_mock():
    settings = Settings(network_env="CLOSED")
    assert settings.rag_backend is None
    assert settings.embedding_url is None
    assert settings.embedding_dim == 1024
    assert settings.embedding_model == "BAAI/bge-m3"
    registry = build_tool_registry(rag_backend=settings.rag_backend)
    assert all(
        tool.mock for tool in registry.tools()
        if tool.name in {"search_accidents", "search_guides"}
    )


def test_core_and_backend_share_infra_environment(monkeypatch):
    variables = {
        "EMBEDDING_URL": "http://embedding:8000/v1/embeddings/",
        "EMBEDDING_MODEL": "deployment-model",
        "EMBEDDING_DIM": "8",
        "EMBEDDING_API_KEY": "test-embedding-key",
        "QDRANT_URL": "http://qdrant-server:6333/",
        "QDRANT_READ_ONLY_API_KEY": "test-reader-key",
        "QDRANT_API_KEY": "test-writer-key",
        "RAG_SEARCH_TIMEOUT_SECONDS": "7.5",
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    core = Settings(network_env="CLOSED")
    backend = RagSettings.from_env()
    assert backend == RagSettings.from_settings(core)
    assert backend.embedding_url == "http://embedding:8000/v1/embeddings"
    assert backend.qdrant_url == "http://qdrant-server:6333"
    assert backend.embedding_model == "deployment-model"
    assert backend.embedding_dim == 8
    assert backend.timeout_seconds == 7.5
    assert backend.embedding_api_key == "test-embedding-key"
    assert backend.qdrant_api_key == "test-reader-key"
    for representation in (repr(core), repr(backend)):
        assert "test-embedding-key" not in representation
        assert "test-reader-key" not in representation


def test_dotenv_is_shared_and_environment_has_priority(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "EMBEDDING_URL=http://embedding/v1/embeddings\nEMBEDDING_DIM=8\n",
        encoding="utf-8",
    )
    assert RagSettings.from_env().embedding_dim == 8
    monkeypatch.setenv("EMBEDDING_DIM", "16")
    assert RagSettings.from_env().embedding_dim == 16
    assert Settings(network_env="CLOSED").embedding_dim == 16


def test_embedding_endpoint_only_required_when_building_backend():
    assert RagEnvironmentSettings().embedding_url is None
    with pytest.raises(ValueError, match="EMBEDDING_URL"):
        RagSettings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_URL", ""),
        ("EMBEDDING_URL", "embedding:8000"),
        ("EMBEDDING_URL", "ftp://embedding"),
        ("QDRANT_URL", ""),
        ("QDRANT_URL", "file:///qdrant"),
        ("EMBEDDING_MODEL", " "),
        ("EMBEDDING_DIM", "0"),
        ("EMBEDDING_DIM", "-1"),
        ("EMBEDDING_DIM", "1.5"),
        ("RAG_SEARCH_TIMEOUT_SECONDS", "0"),
        ("RAG_SEARCH_TIMEOUT_SECONDS", "nan"),
        ("RAG_SEARCH_TIMEOUT_SECONDS", "inf"),
    ],
)
def test_invalid_infra_settings_fail_explicitly(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        RagEnvironmentSettings()


def test_tool_registration_keeps_default_collection_and_extra_filters():
    calls = []

    def search(query, *, k, filters):
        calls.append(filters)
        return []

    registry = ToolRegistry()
    register_rag_tools(
        registry, search,
        filters_by_tool={"search_guides": {"work_type": "HOT"}},
    )
    registry.call("search_accidents", {"query": "fire", "collection": "kb_guides"})
    registry.call("search_guides", {"query": "fire", "collection": "kb_accidents"})
    assert calls == [
        {"collection": "kb_accidents"},
        {"collection": "kb_guides", "work_type": "HOT"},
    ]


def test_registration_can_override_collection_explicitly():
    calls = []

    def search(query, *, k, filters):
        calls.append(filters)
        return []

    registry = ToolRegistry()
    register_rag_tools(
        registry, search,
        filters_by_tool={"search_guides": {"collection": "custom_guides"}},
    )
    registry.call("search_guides", {"query": "fire"})
    assert calls == [{"collection": "custom_guides"}]


def test_registered_backend_routes_and_preserves_contract(monkeypatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://embedding/v1/embeddings")
    monkeypatch.setenv("EMBEDDING_DIM", "2")
    calls = []

    def respond(request, timeout):
        body = json.loads(request.data)
        calls.append((request.full_url, body))
        if request.full_url == "http://embedding/v1/embeddings":
            result = {"data": [{"embedding": [0.1, 0.2]}]}
        else:
            is_guide = "/kb_guides/" in request.full_url
            result = {"result": {"points": [{
                "id": "00000000-0000-5000-8000-000000000001",
                "score": -0.2,
                "payload": {
                    "kb_source_id": 42,
                    "source_type": "LAW" if is_guide else "ACCIDENT_REPORT",
                    "title": "Safety",
                    "locator": "Section 1",
                    "text": "Fire prevention",
                },
            }]}}
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(http, "urlopen", respond)
    registry = build_tool_registry(
        rag_backend="app.rag.tool_backend:build_search_backend",
    )
    assert all(
        not tool.mock for tool in registry.tools()
        if tool.name in {"search_accidents", "search_guides"}
    )
    output = registry.call("search_accidents", {"query": " fire ", "k": 50})
    assert output["hits"][0]["doc_id"] == "ACCIDENT_REPORT:42"
    assert output["hits"][0]["score"] == -0.2
    assert output["hits"][0]["source"] == "Safety (Section 1)"
    assert "/kb_accidents/" in calls[1][0]
    assert calls[0][1]["input"] == "fire"
    assert calls[1][1]["limit"] == 50
    assert {"key": "flagged", "match": {"value": False}} in calls[1][1]["filter"]["must"]
    assert {"key": "site_id", "match": {"any": [0]}} in calls[1][1]["filter"]["must"]

    assert registry.call(
        "search_guides", {"query": "fire", "min_score": 0.0},
    ) == {"hits": []}
    assert "/kb_guides/" in calls[3][0]
    assert len(calls) == 4


def test_connection_error_still_maps_to_tool_unavailable(monkeypatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://embedding/v1/embeddings")

    def unavailable(request, timeout):
        raise URLError("test connection failure")

    monkeypatch.setattr(http, "urlopen", unavailable)
    backend = load_search_backend("app.rag.tool_backend:build_search_backend")
    assert backend is not None
    registry = ToolRegistry()
    register_rag_tools(registry, backend.search, backend_errors=backend.errors)
    with pytest.raises(ToolError) as error:
        registry.call("search_guides", {"query": "fire"})
    assert error.value.code == "TOOL_UNAVAILABLE"
