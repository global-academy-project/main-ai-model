"""ASGI 진입점으로 ingest 라우터를 조립하고 기존 import 경로를 유지한다."""

from __future__ import annotations

from fastapi import FastAPI

from app.rag.catalog import RegisteredChunk
from app.rag.config import RagSettings
from .ingest.auth import authenticate
from .ingest.chunks import router as chunks_router
from .ingest.documents import router as documents_router
from .ingest.local_vectors import router as local_vectors_router
from .ingest.points import router as points_router
from .ingest.schemas import (
    DeletePointsRequest,
    IngestChunk,
    IngestRequest,
    LocalChunkFileRequest,
    LocalChunkRecord,
    LocalQdrantPoint,
    LocalVectorLoadRequest,
    LocalVectorSaveRequest,
    PublicAccidentChunk,
    PublicAccidentIngestRequest,
    PublicAccidentSource,
    PublicAccidentTags,
    PublicGuideChunk,
    PublicGuideIngestRequest,
    PublicGuideSource,
    PublicGuideTags,
    PublicSource,
    PublicSourceChunk,
    PublicSourceChunkBase,
    PublicSourceIngestRequest,
    PublicSourceMetadata,
    SourceCompletion,
    _validate_point_id,
)

app = FastAPI(title="KB Ingest API")


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ok"}


# 기존 uvicorn 진입점은 유지하고 기능별 라우터만 여기서 합친다.
app.include_router(documents_router)
app.include_router(chunks_router)
app.include_router(local_vectors_router)
app.include_router(points_router)
