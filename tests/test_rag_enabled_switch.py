"""RAG 켜기·끄기 스위치 (Settings.rag_enabled, Settings.rag_backend_path, agent_deps.get_tool_registry).

실제 RAG·Qdrant 없이 돈다. 설정 객체는 .env 를 읽지 않게 만들고(_env_file=None), 환경변수는 monkeypatch 로 고정한다.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings

DEFAULT_PATH = "app.rag.tool_backend:build_search_backend"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # 개발 PC 의 환경변수가 결과를 바꾸지 않게 비우고, 필수 값만 채운다.
    monkeypatch.delenv("RAG_ENABLED", raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")


def make_settings() -> Settings:
    return Settings(_env_file=None)


def test_default_is_on_with_rag_path():
    s = make_settings()
    assert s.rag_enabled is True
    assert s.rag_backend == DEFAULT_PATH
    assert s.rag_backend_path == DEFAULT_PATH


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_disabled_gives_none(monkeypatch, value):
    monkeypatch.setenv("RAG_ENABLED", value)
    s = make_settings()
    assert s.rag_enabled is False
    assert s.rag_backend_path is None            # mock 으로 돌아간다
    assert s.rag_backend == DEFAULT_PATH         # 경로 자체는 그대로 남는다


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_backend_still_turns_off(monkeypatch, value):
    # 기존 안내(RAG_BACKEND= 빈 값으로 끄기)가 계속 통한다.
    monkeypatch.setenv("RAG_BACKEND", value)
    assert make_settings().rag_backend_path is None


def test_backend_path_can_be_overridden(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "mypkg.mod:build")
    assert make_settings().rag_backend_path == "mypkg.mod:build"


def test_disabled_wins_over_backend_override(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "mypkg.mod:build")
    monkeypatch.setenv("RAG_ENABLED", "false")
    assert make_settings().rag_backend_path is None


def test_module_level_rag_backend_is_gone():
    import app.core.config as config

    assert not hasattr(config, "rag_backend")    # 7행의 효과 없던 한 줄을 지웠다


# ── get_tool_registry 가 어떤 값을 build_tool_registry 에 넘기는지 ─────────────────

class _FakeSettings:
    def __init__(self, path):
        self.rag_backend_path = path
        self.rag_backend = "WRONG: agent_deps 는 rag_backend_path 를 써야 한다"


@pytest.mark.parametrize("path", [DEFAULT_PATH, None])
def test_get_tool_registry_passes_rag_backend_path(monkeypatch, path):
    import app.agent.agent_deps as deps

    seen = {}
    monkeypatch.setattr(deps, "get_settings", lambda: _FakeSettings(path))
    monkeypatch.setattr(deps, "build_tool_registry", lambda **kw: seen.update(kw) or "registry")

    assert deps.get_tool_registry.__wrapped__() == "registry"   # lru_cache 를 거치지 않고 직접 호출
    assert seen == {"rag_backend": path}
