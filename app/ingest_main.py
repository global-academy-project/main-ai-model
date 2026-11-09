from __future__ import annotations

import os
import logging
from datetime import date, datetime
from secrets import compare_digest
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

from app.rag.catalog import (
    RegisteredChunk,
    delete_stale_chunks,
    mark_sources_embedded,
    mark_sources_failed,
    register_source_chunks,
    validate_chunks,
)
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


class SourceCompletion(BaseModel):
    source_id: StrictInt = Field(gt=0)
    chunk_count: StrictInt = Field(gt=0)


class IngestRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    chunks: list[IngestChunk] = Field(min_length=1, max_length=32)
    finalize_sources: list[SourceCompletion] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_payloads(self) -> IngestRequest:
        required_fields = {"kb_source_id", "kb_chunk_id", "source_type", "site_id", "flagged"}
        source_types: dict[int, str] = {}
        chunk_ids: set[int] = set()
        for chunk in self.chunks:
            payload = chunk.payload
            if not required_fields <= payload.keys():
                raise ValueError("payload에 RAG 필수 필드가 없습니다.")
            if (
                isinstance(payload["kb_source_id"], bool)
                or not isinstance(payload["kb_source_id"], int)
                or payload["kb_source_id"] <= 0
                or isinstance(payload["kb_chunk_id"], bool)
                or not isinstance(payload["kb_chunk_id"], int)
                or payload["kb_chunk_id"] <= 0
                or isinstance(payload["site_id"], bool)
                or not isinstance(payload["site_id"], int)
                or payload["site_id"] < 0
                or not isinstance(payload["flagged"], bool)
                or payload["source_type"] not in COLLECTIONS[self.collection]
            ):
                raise ValueError("payload 값이 collection의 RAG 스키마와 맞지 않습니다.")
            if payload["source_type"] != "NEAR_MISS" and (
                not isinstance(payload.get("text"), str) or not payload["text"].strip()
            ):
                raise ValueError("공공 지식 payload에 원문 text가 필요합니다.")
            if payload["kb_chunk_id"] in chunk_ids:
                raise ValueError("요청에 중복된 kb_chunk_id가 있습니다.")
            chunk_ids.add(payload["kb_chunk_id"])
            source_types[payload["kb_source_id"]] = payload["source_type"]
            if payload["source_type"] == "NEAR_MISS":
                self._validate_near_miss(chunk)
        completion_ids = [source.source_id for source in self.finalize_sources]
        if len(set(completion_ids)) != len(completion_ids):
            raise ValueError("finalize_sources에 중복된 kb_source_id가 있습니다.")
        for source_id in self.finalize_sources:
            source_type = source_types.get(source_id.source_id)
            if source_type is None:
                raise ValueError("완료 처리할 출처는 요청 청크에 포함되어야 합니다.")
            if source_type == "NEAR_MISS":
                raise ValueError("NEAR_MISS는 agent_kb 완료 처리 대상이 아닙니다.")
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


class PublicSource(BaseModel):
    source_type: Literal[
        "ACCIDENT_REPORT", "SIF", "KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"
    ]
    external_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    url: str | None = Field(default=None, max_length=500)
    license: str | None = Field(default=None, max_length=50)
    publisher: str | None = Field(default=None, max_length=100)
    published_on: date | None = None
    file_path: str | None = Field(default=None, max_length=500)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    flagged: bool = False

    @model_validator(mode="after")
    def validate_catalog_fields(self) -> PublicSource:
        if not self.external_id.strip() or not self.title.strip():
            raise ValueError("external_id와 title은 공백일 수 없습니다.")
        if not (self.url and self.url.strip()) and not (self.file_path and self.file_path.strip()):
            raise ValueError("공공 source에는 url 또는 file_path가 필요합니다.")
        return self


class PublicSourceChunk(BaseModel):
    chunk_index: StrictInt = Field(ge=0)
    text: str = Field(min_length=1, max_length=32768)
    embedding_text: str | None = Field(default=None, min_length=1, max_length=32768)
    locator: str | None = Field(default=None, max_length=100)
    token_count: StrictInt | None = Field(default=None, gt=0)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "embedding_text")
    @classmethod
    def validate_nonblank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("청크 본문은 공백일 수 없습니다.")
        return value


class PublicSourceIngestRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    source: PublicSource
    chunks: list[PublicSourceChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_source_collection(self) -> PublicSourceIngestRequest:
        if self.source.source_type not in COLLECTIONS[self.collection]:
            raise ValueError("source_type이 collection과 맞지 않습니다.")
        indexes = [chunk.chunk_index for chunk in self.chunks]
        if len(set(indexes)) != len(indexes):
            raise ValueError("요청에 중복된 chunk_index가 있습니다.")
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


def _build_public_ingest_request(
    body: PublicSourceIngestRequest,
    source_id: int,
    registered_chunks: list[RegisteredChunk],
) -> IngestRequest:
    ids_by_index = {chunk.chunk_index: chunk for chunk in registered_chunks}
    chunks = []
    for source_chunk in body.chunks:
        registered = ids_by_index[source_chunk.chunk_index]
        payload = dict(source_chunk.payload)
        payload.update(
            {
                "kb_source_id": source_id,
                "kb_chunk_id": registered.chunk_id,
                "source_type": body.source.source_type,
                "site_id": 0,
                "flagged": body.source.flagged,
                "title": body.source.title,
                "locator": source_chunk.locator,
                "url": body.source.url,
                "text": source_chunk.text,
            }
        )
        chunks.append(
            IngestChunk(
                point_id=registered.point_id,
                text=source_chunk.embedding_text or source_chunk.text,
                payload=payload,
            )
        )
    return IngestRequest(
        collection=body.collection,
        chunks=chunks,
        finalize_sources=[{"source_id": source_id, "chunk_count": len(chunks)}],
    )


def _mark_failed_after_ingest(settings: RagSettings, source_ids: list[int]) -> None:
    try:
        mark_sources_failed(settings, source_ids, "임베딩 또는 Qdrant 적재에 실패했습니다.")
    except RagBackendError:
        logger.exception("agent_kb failure status update failed")


@app.post("/ingest/document")
def ingest_document(
    body: PublicSourceIngestRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, Any]:
    write_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not write_api_key:
        raise HTTPException(status_code=503, detail="Qdrant 쓰기 설정이 없습니다.")
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    preflight_ids = [
        RegisteredChunk(
            chunk_index=chunk.chunk_index,
            chunk_id=index + 1,
            point_id=str(uuid5(
                NAMESPACE_URL,
                f"{body.source.source_type}:{body.source.external_id}:{chunk.chunk_index}",
            )),
        )
        for index, chunk in enumerate(body.chunks)
    ]
    try:
        _build_public_ingest_request(body, 1, preflight_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        registered = register_source_chunks(
            settings,
            body.collection,
            body.source.model_dump(),
            [chunk.model_dump() for chunk in body.chunks],
        )
        request = _build_public_ingest_request(body, registered.source_id, list(registered.chunks))
        validate_chunks(
            settings,
            request.collection,
            [
                IndexableChunk(point_id=chunk.point_id, text=chunk.text, payload=chunk.payload)
                for chunk in request.chunks
            ],
        )
    except RagBackendError as exc:
        logger.exception("agent_kb source registration or validation failed")
        raise HTTPException(status_code=503, detail="지식 DB 원본·청크 등록 또는 검증에 실패했습니다.") from exc

    indexable = [
        IndexableChunk(point_id=chunk.point_id, text=chunk.text, payload=chunk.payload)
        for chunk in request.chunks
    ]
    try:
        indexed = index_chunks(settings, request.collection, indexable, write_api_key=write_api_key)
        for offset in range(0, len(registered.stale_point_ids), 32):
            delete_qdrant_points(
                settings,
                request.collection,
                registered.stale_point_ids[offset : offset + 32],
                write_api_key=write_api_key,
            )
        delete_stale_chunks(settings, registered.source_id, registered.stale_chunk_ids)
        mark_sources_embedded(
            settings,
            {registered.source_id: len(request.chunks)},
            settings.embedding_model,
        )
    except RagBackendError as exc:
        logger.exception("Public source indexing failed")
        _mark_failed_after_ingest(settings, [registered.source_id])
        raise HTTPException(status_code=502, detail="임베딩, Qdrant 또는 장부 갱신이 실패했습니다.") from exc

    return {
        "indexed": indexed,
        "source_id": registered.source_id,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "point_id": chunk.point_id,
            }
            for chunk in registered.chunks
        ],
    }


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
        validate_chunks(settings, body.collection, chunks)
    except RagBackendError as exc:
        logger.exception("agent_kb ledger validation failed")
        raise HTTPException(status_code=503, detail="지식 DB 장부 확인에 실패했습니다.") from exc

    try:
        indexed = index_chunks(settings, body.collection, chunks, write_api_key=write_api_key)
    except RagBackendError as exc:
        logger.exception("Embedding or Qdrant indexing failed")
        source_ids = sorted({
            chunk.payload["kb_source_id"]
            for chunk in chunks
            if chunk.payload["source_type"] != "NEAR_MISS"
        })
        try:
            mark_sources_failed(settings, source_ids, "임베딩 또는 Qdrant 적재에 실패했습니다.")
        except RagBackendError:
            logger.exception("agent_kb failure status update failed")
        raise HTTPException(status_code=502, detail="임베딩 또는 Qdrant 적재 요청이 실패했습니다.") from exc

    try:
        mark_sources_embedded(
            settings,
            {item.source_id: item.chunk_count for item in body.finalize_sources},
            settings.embedding_model,
        )
    except RagBackendError as exc:
        logger.exception("agent_kb completion status update failed")
        raise HTTPException(
            status_code=502,
            detail="Qdrant 색인 후 지식 DB 완료 상태 저장에 실패했습니다.",
        ) from exc
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