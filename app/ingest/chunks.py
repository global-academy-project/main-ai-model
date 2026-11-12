"""장부에 이미 등록된 청크를 검증하고 임베딩·Qdrant 색인을 수행한다."""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.rag.catalog import mark_sources_embedded, mark_sources_failed, validate_chunks
from app.rag.config import RagSettings
from app.rag.errors import RagBackendError
from app.rag.indexing import IndexableChunk, index_chunks
from .auth import authenticate
from .schemas import IngestRequest


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest/chunks")
def ingest(body: IngestRequest, _: Annotated[None, Depends(authenticate)]) -> dict[str, int]:
    """청크를 장부와 대조한 뒤 색인하고 요청된 출처를 완료 처리한다."""
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
