from __future__ import annotations

import os
import logging
from datetime import datetime
from secrets import compare_digest
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.config import RagSettings
from app.rag.errors import RagBackendError
from app.rag.filters import COLLECTIONS
from app.rag.indexing import IndexableChunk, index_chunks
from app.rag.qdrant import delete_points as delete_qdrant_points

app = FastAPI(title="KB Ingest API")
bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def _validate_point_id(value: object) -> str | int:
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


class IngestChunk(BaseModel):
    point_id: str | int
    text: str = Field(min_length=1, max_length=32768)
    payload: dict[str, Any]

    @field_validator("point_id", mode="before")
    @classmethod
    def validate_point_id(cls, value: object) -> str | int:
        return _validate_point_id(value)

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
            if payload["source_type"] == "NEAR_MISS":
                self._validate_near_miss(chunk)
        return self

    @staticmethod
    def _validate_near_miss(chunk: IngestChunk) -> None:
        payload = chunk.payload
        required_fields = {
            "severity",
            "accident_type",
            "hazard",
            "work_type",
            "cas",
            "ghs",
            "year",
            "occurred_at",
            "zone_id",
            "title",
            "text",
        }
        if not required_fields <= payload.keys():
            raise ValueError("NEAR_MISS payload에 필수 분류·원문 필드가 없습니다.")

        source_id = payload["kb_source_id"]
        if source_id <= 0 or payload["kb_chunk_id"] != source_id:
            raise ValueError("NEAR_MISS의 kb_source_id와 kb_chunk_id는 같은 양의 업무 ID여야 합니다.")
        expected_point_id = str(uuid5(NAMESPACE_URL, f"NEAR_MISS:{source_id}:0"))
        if chunk.point_id != expected_point_id:
            raise ValueError("NEAR_MISS point_id가 UUIDv5 재적재 규칙과 맞지 않습니다.")
        if payload["site_id"] <= 0 or payload["flagged"] is not False:
            raise ValueError("NEAR_MISS는 양수 site_id와 flagged=false가 필요합니다.")

        if not isinstance(payload["severity"], str) or not payload["severity"].strip():
            raise ValueError("NEAR_MISS severity는 비어 있지 않은 문자열이어야 합니다.")
        for field in ("accident_type", "hazard", "work_type", "cas", "ghs"):
            values = payload[field]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"NEAR_MISS {field}는 문자열 배열이어야 합니다.")
        if payload["ghs"]:
            raise ValueError("현재 NEAR_MISS 계약에서 ghs는 빈 배열이어야 합니다.")

        occurred_at = payload["occurred_at"]
        if not isinstance(occurred_at, str):
            raise ValueError("NEAR_MISS occurred_at은 ISO 형식 문자열이어야 합니다.")
        try:
            occurred_year = datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).year
        except ValueError as exc:
            raise ValueError("NEAR_MISS occurred_at은 ISO 형식이어야 합니다.") from exc
        year = payload["year"]
        if isinstance(year, bool) or not isinstance(year, int) or year != occurred_year:
            raise ValueError("NEAR_MISS year는 occurred_at의 연도와 같은 정수여야 합니다.")

        zone_id = payload["zone_id"]
        if zone_id is not None and (
            isinstance(zone_id, bool) or not isinstance(zone_id, int) or zone_id < 0
        ):
            raise ValueError("NEAR_MISS zone_id는 음이 아닌 정수 또는 null이어야 합니다.")
        if not isinstance(payload["title"], str) or not payload["title"].strip():
            raise ValueError("NEAR_MISS title은 비어 있지 않은 문자열이어야 합니다.")
        if payload["text"] != chunk.text:
            raise ValueError("NEAR_MISS payload.text와 청크 text는 같아야 합니다.")


class DeletePointsRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    point_ids: list[str | int] = Field(min_length=1, max_length=32)

    @field_validator("point_ids", mode="before")
    @classmethod
    def validate_point_ids(cls, value: object) -> list[str | int]:
        if not isinstance(value, list):
            raise ValueError("point_ids는 UUID 문자열 또는 음이 아닌 정수의 배열이어야 합니다.")
        return [_validate_point_id(point_id) for point_id in value]


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


@app.post("/ingest/points/delete")
def delete_ingest_points(
    body: DeletePointsRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, int]:
    write_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not write_api_key:
        raise HTTPException(status_code=503, detail="Qdrant 쓰기 설정이 없습니다.")
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    try:
        delete_qdrant_points(
            settings,
            body.collection,
            body.point_ids,
            write_api_key=write_api_key,
        )
    except RagBackendError as exc:
        logger.exception("Qdrant point deletion failed")
        raise HTTPException(status_code=502, detail="Qdrant 삭제 요청이 실패했습니다.") from exc
    return {"deleted": len(body.point_ids)}