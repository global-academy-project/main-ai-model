from __future__ import annotations

import os
import logging
import json
import math
import tempfile
from datetime import date, datetime
from pathlib import Path
from secrets import compare_digest
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator, model_validator

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
from app.rag.embedding import embed_texts
from app.rag.filters import COLLECTIONS
from app.rag.indexing import IndexableChunk, index_chunks
from app.rag.qdrant import delete_points as delete_qdrant_points, make_point, upsert_points

app = FastAPI(title="KB Ingest API")
bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)
LOCAL_VECTOR_BATCH_SIZE = 8
LOCAL_QDRANT_BATCH_SIZE = 32


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


class LocalVectorSaveRequest(IngestRequest):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    chunks: list[IngestChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_unique_point_ids(self) -> LocalVectorSaveRequest:
        point_ids = [chunk.point_id for chunk in self.chunks]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("요청에 중복된 point_id가 있습니다.")
        return self


class LocalChunkRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    natural_key: dict[str, Any]
    text: str = Field(min_length=1, max_length=32768)
    embed_text: str = Field(min_length=1, max_length=32768)
    collection: Literal["kb_accidents", "kb_guides"] = "kb_guides"

    @model_validator(mode="after")
    def validate_chunk(self) -> LocalChunkRecord:
        key = self.natural_key
        if self.collection == "kb_guides":
            part = key.get("part")
            if (
                not isinstance(key.get("law_key"), str)
                or not key["law_key"].strip()
                or not isinstance(key.get("article_key"), str)
                or not key["article_key"].strip()
                or isinstance(part, bool)
                or not isinstance(part, int)
                or part < 0
            ):
                raise ValueError("법령 natural_key에 law_key, article_key, part가 필요합니다.")
        else:
            extra = self.model_extra or {}
            external_id = extra.get("external_id")
            chunk_index = extra.get("chunk_index")
            if (
                key.get("dataset") != "sif"
                or not isinstance(key.get("record"), str)
                or not key["record"].strip()
                or not isinstance(key.get("section"), str)
                or not key["section"].strip()
                or isinstance(key.get("part"), bool)
                or not isinstance(key.get("part"), int)
                or key["part"] < 0
                or not isinstance(external_id, str)
                or external_id != key.get("record")
                or isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
                or extra.get("source_type") != "SIF"
            ):
                raise ValueError("SIF 청크 natural_key, external_id, chunk_index, source_type가 올바르지 않습니다.")
        if not self.text.strip() or not self.embed_text.strip():
            raise ValueError("청크 text와 embed_text는 비어 있을 수 없습니다.")
        return self


class LocalChunkFileRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    chunk_file: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jsonl$")
    data_type: Literal["accidents", "guides"] | None = None
    subdir: str = Field(default="law", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    validation_file: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.json$")


class LocalVectorLoadRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    collection: Literal["kb_accidents", "kb_guides"]


class LocalQdrantPoint(BaseModel):
    id: str | int
    vector: dict[str, Any]
    payload: dict[str, Any]

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: object) -> str | int:
        return _validate_point_id(value)

    @model_validator(mode="after")
    def validate_vector(self) -> LocalQdrantPoint:
        dense = self.vector.get("dense")
        if (
            not isinstance(dense, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in dense
            )
        ):
            raise ValueError("dense 벡터 형식이 올바르지 않습니다.")
        sparse = self.vector.get("sparse")
        if sparse is not None:
            if not isinstance(sparse, dict):
                raise ValueError("sparse 벡터 형식이 올바르지 않습니다.")
            indices, values = sparse.get("indices"), sparse.get("values")
            if (
                not isinstance(indices, list)
                or not isinstance(values, list)
                or len(indices) != len(values)
                or any(type(index) is not int or index < 0 for index in indices)
                or indices != sorted(set(indices))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError("sparse 벡터 형식이 올바르지 않습니다.")
        return self


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


@app.post("/ingest/vectors/local")
def save_local_vectors(
    body: LocalVectorSaveRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    output_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser()
    destination = output_dir / f"{body.dataset_id}.jsonl"
    temp_path: Path | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{body.dataset_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for offset in range(0, len(body.chunks), LOCAL_VECTOR_BATCH_SIZE):
                batch = body.chunks[offset : offset + LOCAL_VECTOR_BATCH_SIZE]
                embeddings = embed_texts(settings, [chunk.text for chunk in batch])
                if len(embeddings) != len(batch):
                    raise RagBackendError("임베딩 서버가 청크 수와 다른 결과를 반환했습니다.")
                for chunk, embedding in zip(batch, embeddings):
                    point = make_point(chunk.point_id, embedding, chunk.payload)
                    output.write(
                        json.dumps(
                            {"collection": body.collection, "model": settings.embedding_model, **point},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        os.replace(temp_path, destination)
    except RagBackendError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local vector embedding failed")
        raise HTTPException(status_code=502, detail="임베딩 요청이 실패했습니다.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local vector file save failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일 저장에 실패했습니다.") from exc

    return {
        "saved": len(body.chunks),
        "dataset_id": body.dataset_id,
        "file": destination.name,
        "model": settings.embedding_model,
    }


@app.post("/ingest/vectors/local/file")
def save_local_chunk_file_vectors(
    body: LocalChunkFileRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    if body.data_type is None:
        input_dir = Path(os.getenv("LOCAL_CHUNK_DIR", "out/law")).expanduser().resolve()
    else:
        input_root = Path(os.getenv("LOCAL_CHUNK_ROOT", "out")).expanduser().resolve()
        input_dir = (input_root / body.data_type / body.subdir).resolve()
        try:
            input_dir.relative_to(input_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="청크 디렉터리 경로가 올바르지 않습니다.") from exc
    chunk_path = (input_dir / body.chunk_file).resolve()
    validation_file = body.validation_file
    if validation_file is None and body.data_type != "accidents":
        validation_file = "06_validation.json"
    validation_path = (input_dir / validation_file).resolve() if validation_file else None
    if chunk_path.parent != input_dir or (
        validation_path is not None and validation_path.parent != input_dir
    ):
        raise HTTPException(status_code=422, detail="청크 파일 경로가 올바르지 않습니다.")
    if not chunk_path.is_file():
        raise HTTPException(status_code=404, detail="로컬 청크 파일을 찾을 수 없습니다.")
    if validation_path is not None and not validation_path.is_file():
        if body.data_type == "accidents":
            validation_path = None
            validation_file = None
        else:
            raise HTTPException(status_code=422, detail=f"검증 파일을 찾을 수 없습니다: {validation_file}")

    validation: dict[str, Any] | None = None
    try:
        if validation_path is not None:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if validation.get("ok") is not True:
                raise HTTPException(status_code=422, detail="청크 검증을 통과하지 못했습니다.")
        records: list[LocalChunkRecord] = []
        for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(LocalChunkRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"청크 파일 {line_number}번째 줄을 읽을 수 없습니다.",
                ) from exc
            if len(records) > 5000:
                raise HTTPException(status_code=422, detail="청크 파일은 최대 5,000건까지 처리할 수 있습니다.")
    except OSError as exc:
        logger.exception("Local chunk file read failed")
        raise HTTPException(status_code=503, detail="로컬 청크 파일을 읽지 못했습니다.") from exc
    except (json.JSONDecodeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="06_validation.json 형식이 올바르지 않습니다.") from exc

    if not records:
        raise HTTPException(status_code=422, detail="청크 파일이 비어 있습니다.")
    if validation is not None:
        summary = validation.get("summary")
        expected_count = summary.get("chunks") if isinstance(summary, dict) else None
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(records)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"청크 수가 검증 결과와 일치하지 않습니다 (파일 {len(records)}, 검증 {expected_count}).",
            )
    collections = {record.collection for record in records}
    if len(collections) != 1:
        raise HTTPException(status_code=422, detail="청크 파일의 collection이 일치하지 않습니다.")
    collection = next(iter(collections))
    expected_collection = {
        "accidents": "kb_accidents",
        "guides": "kb_guides",
    }.get(body.data_type)
    if expected_collection is not None and collection != expected_collection:
        raise HTTPException(status_code=422, detail="data_type과 청크 collection이 일치하지 않습니다.")
    natural_keys = [tuple(sorted(record.natural_key.items())) for record in records]
    if len(set(natural_keys)) != len(natural_keys):
        raise HTTPException(status_code=422, detail="청크 파일에 중복된 natural_key가 있습니다.")
    indexed_records: list[tuple[LocalChunkRecord, str, dict[str, Any]]] = []
    if collection == "kb_guides":
        law_keys = {record.natural_key.get("law_key") for record in records}
        source_types = {record.model_extra.get("source_type") if record.model_extra else None for record in records}
        if len(law_keys) != 1 or source_types != {"LAW"}:
            raise HTTPException(status_code=422, detail="법령 청크의 법령 ID 또는 source_type이 일치하지 않습니다.")
        external_id = next(iter(law_keys))

        index_map_path = input_dir / "chunk_index_map.json"
        if not index_map_path.is_file():
            raise HTTPException(status_code=422, detail="chunk_index_map.json이 필요합니다.")
        try:
            index_map_data = json.loads(index_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="chunk_index_map.json을 읽을 수 없습니다.") from exc
        indexes = index_map_data.get("indexes") if isinstance(index_map_data, dict) else None
        if (
            not isinstance(indexes, dict)
            or index_map_data.get("source_type") != "LAW"
            or index_map_data.get("external_id") != external_id
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in indexes.items()
            )
            or len(set(indexes.values())) != len(indexes)
        ):
            raise HTTPException(status_code=422, detail="chunk_index_map.json이 법령 또는 인덱스 형식과 맞지 않습니다.")

        for record in records:
            key = record.natural_key
            natural_key = f"{key['law_key']}|{key['article_key']}|{key['part']}"
            chunk_index = indexes.get(natural_key)
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
                raise HTTPException(status_code=422, detail="청크의 natural_key가 chunk_index_map.json에 없습니다.")
            point_id = str(uuid5(NAMESPACE_URL, f"LAW:{external_id}:{chunk_index}"))
            indexed_records.append((record, point_id, record.model_dump()))
    else:
        rows = [record.model_dump() for record in records]
        source_types = {row.get("source_type") for row in rows}
        datasets = {record.natural_key.get("dataset") for record in records}
        if source_types != {"SIF"} or datasets != {"sif"}:
            raise HTTPException(status_code=422, detail="현재 사고 파일 벡터화는 SIF 데이터셋만 지원합니다.")
        if validation is not None and validation.get("dataset") != "sif":
            raise HTTPException(status_code=422, detail="사고 validation 파일의 dataset이 sif가 아닙니다.")

        seen_chunk_indexes: set[tuple[str, int]] = set()
        payload_fields = (
            "source_type", "site_id", "flagged", "title", "locator", "url", "text",
            "severity", "accident_type", "hazard", "work_type", "cas", "ghs", "year",
            "external_id", "chunk_index", "natural_key", "license", "publisher",
            "content_sha256", "tier", "cate", "disaster_type", "situation",
        )
        for record, row in zip(records, rows):
            external_id = row.get("external_id")
            chunk_index = row.get("chunk_index")
            if (
                not isinstance(external_id, str)
                or not external_id.strip()
                or isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
            ):
                raise HTTPException(status_code=422, detail="SIF 청크 external_id 또는 chunk_index가 올바르지 않습니다.")
            key = (external_id, chunk_index)
            if key in seen_chunk_indexes:
                raise HTTPException(status_code=422, detail="SIF 청크 external_id/chunk_index가 중복됩니다.")
            seen_chunk_indexes.add(key)

            site_id = row.get("site_id")
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise HTTPException(status_code=422, detail="SIF payload.site_id가 올바르지 않습니다.")
            if not isinstance(row.get("flagged"), bool):
                raise HTTPException(status_code=422, detail="SIF payload.flagged가 올바르지 않습니다.")
            if not isinstance(row.get("title"), str) or not row["title"].strip():
                raise HTTPException(status_code=422, detail="SIF payload.title이 비어 있습니다.")
            if not isinstance(row.get("locator"), str) or not row["locator"].strip():
                raise HTTPException(status_code=422, detail="SIF payload.locator가 비어 있습니다.")
            if row.get("year") is not None and (
                isinstance(row["year"], bool) or not isinstance(row["year"], int)
            ):
                raise HTTPException(status_code=422, detail="SIF payload.year가 올바르지 않습니다.")
            if row.get("severity") is not None and not isinstance(row["severity"], str):
                raise HTTPException(status_code=422, detail="SIF payload.severity가 올바르지 않습니다.")
            for field in ("accident_type", "hazard", "work_type", "cas", "ghs"):
                values = row.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise HTTPException(status_code=422, detail=f"SIF payload.{field} 형식이 올바르지 않습니다.")

            point_id = str(uuid5(NAMESPACE_URL, f"SIF:{external_id}:{chunk_index}"))
            payload = {field: row[field] for field in payload_fields if field in row}
            payload["kb_source_id"] = row.get("kb_source_id")
            payload["kb_chunk_id"] = row.get("kb_chunk_id")
            indexed_records.append((record, point_id, payload))

    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    output_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser()
    destination = output_dir / f"{body.dataset_id}.jsonl"
    temp_path: Path | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{body.dataset_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for offset in range(0, len(records), LOCAL_VECTOR_BATCH_SIZE):
                batch = records[offset : offset + LOCAL_VECTOR_BATCH_SIZE]
                embeddings = embed_texts(settings, [record.embed_text for record in batch])
                if len(embeddings) != len(batch):
                    raise RagBackendError("임베딩 서버가 청크 수와 다른 결과를 반환했습니다.")
                for (record, point_id, payload), embedding in zip(indexed_records[offset : offset + len(batch)], embeddings):
                    point = make_point(point_id, embedding, payload)
                    output.write(
                        json.dumps(
                            point,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        os.replace(temp_path, destination)
    except RagBackendError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local file vectorization failed")
        raise HTTPException(status_code=502, detail="임베딩 요청이 실패했습니다.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local vector file save failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일 저장에 실패했습니다.") from exc

    return {
        "saved": len(records),
        "dataset_id": body.dataset_id,
        "file": destination.name,
        "collection": next(iter(collections)),
        "model": settings.embedding_model,
        "validation": "pipeline" if validation is not None else "structural",
    }


@app.post("/ingest/vectors/local/load")
def load_local_vectors_to_qdrant(
    body: LocalVectorLoadRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    write_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not write_api_key:
        raise HTTPException(status_code=503, detail="Qdrant 쓰기 설정이 없습니다.")
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    vector_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser().resolve()
    vector_path = (vector_dir / f"{body.dataset_id}.jsonl").resolve()
    if vector_path.parent != vector_dir:
        raise HTTPException(status_code=422, detail="벡터 파일 경로가 올바르지 않습니다.")
    if not vector_path.is_file():
        raise HTTPException(status_code=404, detail="로컬 벡터 파일을 찾을 수 없습니다.")

    points: list[dict[str, Any]] = []
    point_ids: set[str | int] = set()
    unlinked_payloads = 0
    try:
        for line_number, line in enumerate(vector_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            if len(points) >= 5000:
                raise HTTPException(status_code=422, detail="벡터 파일은 최대 5,000건까지 적재할 수 있습니다.")
            try:
                point = LocalQdrantPoint.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"벡터 파일 {line_number}번째 줄이 Qdrant 포인트 형식이 아닙니다.",
                ) from exc
            if point.id in point_ids:
                raise HTTPException(status_code=422, detail="벡터 파일에 중복된 point id가 있습니다.")
            point_ids.add(point.id)
            if len(point.vector["dense"]) != settings.embedding_dim:
                raise HTTPException(status_code=422, detail="dense 벡터 차원이 EMBEDDING_DIM과 다릅니다.")

            payload = point.payload
            if payload.get("collection", body.collection) != body.collection:
                raise HTTPException(status_code=422, detail="요청 collection과 payload collection이 다릅니다.")
            if payload.get("source_type") not in COLLECTIONS[body.collection]:
                raise HTTPException(status_code=422, detail="source_type이 collection과 맞지 않습니다.")
            site_id = payload.get("site_id")
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise HTTPException(status_code=422, detail="payload.site_id가 올바르지 않습니다.")
            if not isinstance(payload.get("flagged"), bool):
                raise HTTPException(status_code=422, detail="payload.flagged가 올바르지 않습니다.")
            if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                raise HTTPException(status_code=422, detail="payload.text가 비어 있습니다.")

            linked = True
            for id_field in ("kb_source_id", "kb_chunk_id"):
                value = payload.get(id_field)
                if value is None:
                    linked = False
                elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise HTTPException(status_code=422, detail=f"payload.{id_field}가 올바르지 않습니다.")
            if not linked:
                unlinked_payloads += 1
            points.append(point.model_dump())
    except OSError as exc:
        logger.exception("Local vector file read failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일을 읽지 못했습니다.") from exc

    if not points:
        raise HTTPException(status_code=422, detail="벡터 파일이 비어 있습니다.")

    try:
        for offset in range(0, len(points), LOCAL_QDRANT_BATCH_SIZE):
            upsert_points(
                settings,
                body.collection,
                points[offset : offset + LOCAL_QDRANT_BATCH_SIZE],
                write_api_key=write_api_key,
            )
    except RagBackendError as exc:
        logger.exception("Local Qdrant upsert failed")
        raise HTTPException(
            status_code=502,
            detail="Qdrant 적재 요청이 실패했습니다. 같은 요청으로 재시도할 수 있습니다.",
        ) from exc

    return {
        "indexed": len(points),
        "collection": body.collection,
        "dataset_id": body.dataset_id,
        "unlinked_payloads": unlinked_payloads,
    }


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