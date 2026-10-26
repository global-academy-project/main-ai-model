import logging
from fastapi import APIRouter
from app.schemas.base import DataResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

@router.get("/health")
def health_check() -> DataResponse[dict]:
    """현재 fast-api 서버 상태를 반환합니다."""
    logger.info("fast_api 서버 상태 확인")
    
    return DataResponse(data={"status" : "ok"})