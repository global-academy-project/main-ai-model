"""사고·지침 문서를 장부에 등록하고 임베딩 및 Qdrant까지 적재한다."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, HTTPException

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
from app.rag.indexing import IndexableChunk, index_chunks
from app.rag.qdrant import delete_points as delete_qdrant_points
from .auth import authenticate
from .schemas import (
    IngestChunk,
    IngestRequest,
    PublicAccidentIngestRequest,
    PublicGuideIngestRequest,
    PublicSource,
    PublicSourceChunk,
    PublicSourceChunkBase,
    PublicSourceIngestRequest,
    PublicSourceMetadata,
)


logger = logging.getLogger(__name__)
router = APIRouter()


def _build_public_ingest_request(
    body: PublicSourceIngestRequest,
    source_id: int,
    registered_chunks: list[RegisteredChunk],
    *,
    canonical_payload: bool = True,
) -> IngestRequest:
    """장부가 발급한 청크 ID를 payload에 결합하고 결정적 point ID를 전달한다."""
    ids_by_index = {chunk.chunk_index: chunk for chunk in registered_chunks}
    chunks = []
    for source_chunk in body.chunks:
        registered = ids_by_index[source_chunk.chunk_index]
        if canonical_payload:
            payload = {
                "source_type": body.source.source_type,
                "site_id": 0,
                "flagged": body.source.flagged,
                "title": body.source.title,
                "locator": source_chunk.locator,
                "url": body.source.url,
                "text": source_chunk.text,
            }
            if body.collection == "kb_accidents":
                for field in (
                    "severity",
                    "accident_type",
                    "hazard",
                    "work_type",
                    "cas",
                    "ghs",
                    "year",
                ):
                    if field in source_chunk.payload:
                        payload[field] = source_chunk.payload[field]
                if body.source.source_type == "SIF":
                    payload.update(
                        {
                            "external_id": body.source.external_id,
                            "chunk_index": source_chunk.chunk_index,
                            "natural_key": source_chunk.payload["natural_key"],
                            "license": body.source.license,
                            "publisher": body.source.publisher,
                            "content_sha256": body.source.content_sha256,
                            "tier": source_chunk.payload["tier"],
                            "cate": source_chunk.payload["cate"],
                            "disaster_type": source_chunk.payload["disaster_type"],
                            "situation": source_chunk.payload["situation"],
                        }
                    )
            else:
                if body.source.source_type == "LAW":
                    law_payload = dict(source_chunk.payload)
                    payload.update(law_payload)
                    payload.update(
                        {
                            "embed_text": source_chunk.embedding_text or source_chunk.text,
                            "collection": "kb_guides",
                            "file_path": body.source.file_path,
                            "law_meta": law_payload["law_meta"],
                            "point_id": None,
                        }
                    )
                else:
                    for field in ("hazard", "work_type", "cas", "year", "guide_no"):
                        if field in source_chunk.payload and (
                            field != "guide_no" or source_chunk.payload[field] is not None
                        ):
                            payload[field] = source_chunk.payload[field]
            payload.update(
                {
                    "kb_source_id": source_id,
                    "kb_chunk_id": registered.chunk_id,
                }
            )
        else:
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


def _as_public_source_request(
    collection: Literal["kb_accidents", "kb_guides"],
    source: PublicSourceMetadata,
    chunks: Sequence[PublicSourceChunkBase],
) -> PublicSourceIngestRequest:
    source_data = source.model_dump()
    if source_data["source_type"] == "LAW":
        law_meta = chunks[0].model_dump()["payload"].get("law_meta")
        if not isinstance(law_meta, dict) or not isinstance(law_meta.get("source_sha256"), str):
            raise ValueError("LAW source hash가 첫 청크에 없습니다.")
        source_hash = law_meta["source_sha256"]
        if source_data["content_sha256"] is None:
            source_data["content_sha256"] = source_hash
    return PublicSourceIngestRequest(
        collection=collection,
        source=PublicSource.model_validate(source_data),
        chunks=[
            PublicSourceChunk.model_validate(
                chunk.model_dump(exclude_unset=source_data["source_type"] == "LAW")
            )
            for chunk in chunks
        ],
    )


def _ingest_public_document(
    body: PublicSourceIngestRequest,
    *,
    canonical_payload: bool = True,
) -> dict[str, Any]:
    """장부 등록·검증 후 색인하고 성공 시 완료 상태를 기록한다."""
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
        _build_public_ingest_request(
            body,
            1,
            preflight_ids,
            canonical_payload=canonical_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        registered = register_source_chunks(
            settings,
            body.collection,
            body.source.model_dump(),
            [chunk.model_dump() for chunk in body.chunks],
        )
        request = _build_public_ingest_request(
            body,
            registered.source_id,
            list(registered.chunks),
            canonical_payload=canonical_payload,
        )
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


@router.post("/ingest/accidents")
def ingest_accidents(
    body: PublicAccidentIngestRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, Any]:
    """사고 자료를 kb_accidents 컬렉션에 적재한다."""
    request = _as_public_source_request("kb_accidents", body.source, body.chunks)
    return _ingest_public_document(request)


@router.post("/ingest/guides")
def ingest_guides(
    body: PublicGuideIngestRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, Any]:
    """지침·법령 자료를 kb_guides 컬렉션에 적재한다."""
    request = _as_public_source_request("kb_guides", body.source, body.chunks)
    return _ingest_public_document(request)


@router.post("/ingest/document", deprecated=True)
def ingest_document(
    body: PublicSourceIngestRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, Any]:
    """기존 호출 호환을 위해 공통 문서 적재 API를 유지한다."""
    return _ingest_public_document(body, canonical_payload=False)
