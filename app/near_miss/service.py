from __future__ import annotations

from app.exceptions import LLMError
from app.llm.llm_base import LLMClient

from .llm import suggest_missing_tags
from .schemas import NearMissTagRequest, NearMissTagResponse
from .tagging import TAGGING_VERSION, suggest_near_miss_tags


def tag_near_miss(request: NearMissTagRequest, llm: LLMClient) -> NearMissTagResponse:
    tags = suggest_near_miss_tags(
        title=request.title,
        description=request.description,
        immediate_action=request.immediate_action,
        work_type=request.work_type,
    )
    request_text = "\n".join(
        value for value in (request.title, request.description, request.immediate_action, request.zone_name)
        if value
    )

    try:
        tags, model = suggest_missing_tags(llm, request_text=request_text, existing=tags)
    except LLMError:
        raise

    source = "near_miss_rules+llm" if model else "near_miss_rules"
    return NearMissTagResponse(
        tags=tags,
        tagging_source=source,
        tagging_version=TAGGING_VERSION,
        model=model,
    )
