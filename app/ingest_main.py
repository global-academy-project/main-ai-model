from __future__ import annotations

import os
import logging
from secrets import compare_digest
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.config import RagSettings
from app.rag.errors import RagBackendError
from app.rag.filters import COLLECTIONS
from app.rag.indexing import IndexableChunk, index_chunks

app = FastAPI(title="KB Ingest API")
bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


class IngestChunk(BaseModel):
    point_id: str | int
    text: str = Field(min_length=1, max_length=32768)
    payload: dict[str, Any]

    @field_validator("point_id", mode="before")
    @classmethod
    def validate_point_id(cls, value: object) -> str | int:
        if isinstance(value, bool):
            raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.")
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str):
            try:
                return str(UUID(value))
            except ValueError as exc:
                raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.") from exc
        raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 청크는 적재할 수 없습니다.")
        return value


class IngestRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    chunks: list[IngestChunk] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_payloads(self) -> IngestRequest:
        required_fields = {"kb_source_id", "kb_chunk_id", "source_type", "site_id", "flagged"}
        for chunk in self.chunks:
            payload = chunk.payload
            if not required_fields <= payload.keys():
                raise ValueError("payload에 RAG 필수 필드가 없습니다.")
            if (
                isinstance(payload["kb_source_id"], bool)
                or not isinstance(payload["kb_source_id"], int)
                or isinstance(payload["kb_chunk_id"], bool)
                or not isinstance(payload["kb_chunk_id"], int)
                or isinstance(payload["site_id"], bool)
                or not isinstance(payload["site_id"], int)
                or payload["site_id"] < 0
                or not isinstance(payload["flagged"], bool)
                or payload["source_type"] not in COLLECTIONS[self.collection]
            ):
                raise ValueError("payload 값이 collection의 RAG 스키마와 맞지 않습니다.")
        return self


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    token = os.getenv("KB_INGEST_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="적재 API 인증이 설정되지 않았습니다.")
    if credentials is None or not compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=401,
            detail="적재 API 인증에 실패했습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/ready")
def ready():
    return {"status": "ok"}


@app.post("/ingest/chunks")
def ingest(body: IngestRequest, _: Annotated[None, Depends(authenticate)]) -> dict[str, int]:
    write_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not write_api_key:
        raise HTTPException(status_code=503, detail="Qdrant 쓰기 설정이 없습니다.")
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    chunks = [
        IndexableChunk(point_id=chunk.point_id, text=chunk.text, payload=chunk.payload)
        for chunk in body.chunks
    ]
    try:
        indexed = index_chunks(settings, body.collection, chunks, write_api_key=write_api_key)
    except RagBackendError as exc:
        logger.exception("Embedding or Qdrant indexing failed")
        raise HTTPException(status_code=502, detail="임베딩 또는 Qdrant 적재 요청이 실패했습니다.") from exc
    return {"indexed": indexed}