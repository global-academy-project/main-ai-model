"""외부 시스템 연결. 지금은 Spring 내부 도구 API(/internal/v1/tools/*) 하나."""

from .internal_api import InternalApiClient

__all__ = ["InternalApiClient"]
