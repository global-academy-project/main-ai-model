from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.llm_deps import scenario_llm
from app.llm.llm_base import LLMClient
from app.near_miss.schemas import NearMissTagRequest, NearMissTagResponse
from app.near_miss.service import tag_near_miss
from app.schemas import DataResponse

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])
NearMissWarningLLM = Depends(scenario_llm("near_miss_warning"))


@router.post("/near-miss-tag", response_model=DataResponse[NearMissTagResponse])
def near_miss_tag(
    body: NearMissTagRequest,
    llm: LLMClient = NearMissWarningLLM,
) -> DataResponse[NearMissTagResponse]:
    return DataResponse(data=tag_near_miss(body, llm))
