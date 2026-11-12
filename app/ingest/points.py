"""ingest API가 소유한 Qdrant 포인트 삭제 경로를 제공한다."""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.rag.config import RagSettings
from app.rag.errors import RagBackendError
from app.rag.qdrant import delete_points as delete_qdrant_points
from .auth import authenticate
from .schemas import DeletePointsRequest


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest/points/delete")
def delete_ingest_points(
    body: DeletePointsRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, int]:
    """지정 컬렉션의 포인트를 Qdrant에서 삭제한다."""
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
