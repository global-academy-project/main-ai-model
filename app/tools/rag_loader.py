"""설정 문자열 하나로 A의 검색 backend를 연결한다 (`RAG_BACKEND=모듈경로:함수이름`).

    RAG_BACKEND=app.rag.tool_backend:build_search_backend

지정한 함수는 인자 없이 불리고 검색 함수(backend)를 돌려준다. 그 모듈에 `BACKEND_ERRORS = (RagError, ...)`가 있으면
그 예외들을 `TOOL_UNAVAILABLE`로 바꾼다. 값이 비어 있으면 아무것도 연결하지 않고 mock을 그대로 쓴다.

서버 설정에서만 읽는다 (요청자가 모듈을 고를 수 없다). 잘못된 값은 `ValueError`로 알린다 (지침 4장: 설정 오류는 ValueError).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from .adapters.rag_search import SearchBackend


@dataclass(frozen=True)
class LoadedBackend:
    search: SearchBackend
    errors: tuple[type[Exception], ...]


def load_search_backend(spec: str | None) -> LoadedBackend | None:
    if not spec or not spec.strip():
        return None
    module_name, sep, factory_name = spec.strip().partition(":")
    if not sep or not module_name or not factory_name:
        raise ValueError("RAG_BACKEND는 '모듈경로:함수이름' 형식이어야 합니다.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"RAG_BACKEND의 모듈 '{module_name}'을 불러오지 못했습니다.") from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"RAG_BACKEND의 '{module_name}:{factory_name}'를 찾지 못했습니다.")
    search = factory()
    if not callable(search):
        raise ValueError("RAG_BACKEND 함수는 검색 함수(callable)를 돌려줘야 합니다.")
    errors = tuple(getattr(module, "BACKEND_ERRORS", ()))
    if not all(isinstance(e, type) and issubclass(e, Exception) for e in errors):
        raise ValueError("BACKEND_ERRORS는 예외 클래스의 튜플이어야 합니다.")
    return LoadedBackend(search=search, errors=errors)
