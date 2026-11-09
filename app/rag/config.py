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
    kb_db_host: str | None = None
    kb_db_port: int = Field(default=3306, gt=0, le=65535)
    kb_database: str = "agent_kb"
    kb_ingest_password: SecretStr | None = None

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
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("RAG URL은 http 또는 https URL이어야 합니다.")
        return value.rstrip("/")

    @field_validator(
        "embedding_api_key", "qdrant_read_only_api_key", "kb_ingest_password", mode="before"
    )
    @classmethod
    def strip_secret_whitespace(cls, value: SecretStr | str | None) -> SecretStr | str | None:
        if isinstance(value, SecretStr):
            return value.get_secret_value().strip()
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("EMBEDDING_MODEL은 비어 있을 수 없습니다.")
        return value.strip()


@dataclass(frozen=True)
class RagSettings:
    embedding_url: str
    qdrant_url: str
    embedding_model: str
    embedding_dim: int
    timeout_seconds: float
    embedding_api_key: str | None = field(repr=False)
    qdrant_api_key: str | None = field(repr=False)
    kb_db_host: str | None = None
    kb_db_port: int = 3306
    kb_database: str = "agent_kb"
    kb_ingest_password: str | None = field(default=None, repr=False)

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
            kb_db_host=settings.kb_db_host.strip() if settings.kb_db_host else None,
            kb_db_port=settings.kb_db_port,
            kb_database=settings.kb_database,
            kb_ingest_password=(
                settings.kb_ingest_password.get_secret_value()
                if settings.kb_ingest_password is not None
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
