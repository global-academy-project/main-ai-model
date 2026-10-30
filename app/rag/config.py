from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import RagEnvironmentSettings


@dataclass(frozen=True)
class RagSettings:
    embedding_url: str
    qdrant_url: str
    embedding_model: str
    embedding_dim: int
    timeout_seconds: float
    embedding_api_key: str | None = field(repr=False)
    qdrant_api_key: str | None = field(repr=False)

    @classmethod
    def from_env(cls) -> RagSettings:
        return cls.from_settings(RagEnvironmentSettings())

    @classmethod
    def from_settings(cls, settings: RagEnvironmentSettings) -> RagSettings:
        """Reuse the application's settings without reading the environment again."""
        if settings.embedding_url is None:
            raise ValueError("EMBEDDING_URL 환경 변수가 필요합니다.")
        return cls(
            embedding_url=settings.embedding_url,
            qdrant_url=settings.qdrant_url,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            timeout_seconds=settings.rag_search_timeout_seconds,
            embedding_api_key=(
                settings.embedding_api_key.get_secret_value()
                if settings.embedding_api_key is not None
                else None
            ),
            qdrant_api_key=(
                settings.qdrant_read_only_api_key.get_secret_value()
                if settings.qdrant_read_only_api_key is not None
                else None
            ),
        )
