from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagEnvironmentSettings(BaseSettings):
    """Infra-provided RAG settings, independent of application and LLM settings."""

    embedding_url: str | None = None
    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_api_key: SecretStr | None = None
    qdrant_url: str = "http://qdrant:6333"
    qdrant_read_only_api_key: SecretStr | None = None
    rag_search_timeout_seconds: float = Field(default=15.0, gt=0, allow_inf_nan=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("embedding_url", "qdrant_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("RAG URL은 http 또는 https URL이어야 합니다.")
        return value.rstrip("/")

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("EMBEDDING_MODEL은 비어 있을 수 없습니다.")
        return value


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
        """Convert validated RAG settings without reading the environment again."""
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


class EmbeddingServerSettings(RagEnvironmentSettings):
    """Model-serving options; no embedding endpoint or Qdrant access is needed."""

    embedding_model_path: str | None = Field(default=None, min_length=1)
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_local_files_only: bool = True
    embedding_max_tokens: int = Field(default=8192, ge=2, le=8192)
    embedding_max_batch_size: int = Field(default=8, ge=1, le=32)
