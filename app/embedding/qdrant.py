from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from .config import RagSettings
from .errors import RagBackendError
from .http import post_json

_PAYLOAD_FIELDS = [
    "kb_source_id",
    "source_type",
    "site_id",
    "title",
    "locator",
    "url",
    "text",
]


def query_points(
    settings: RagSettings,
    collection: str,
    vector: list[float],
    conditions: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    url = (
        f"{settings.qdrant_url}/collections/"
        f"{quote(collection, safe='')}/points/query"
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    response = post_json(
        url,
        {
            "query": vector,
            "using": "dense",
            "filter": {"must": conditions},
            "limit": limit,
            "with_payload": _PAYLOAD_FIELDS,
        },
        headers,
        settings.timeout_seconds,
        "Qdrant",
    )
    return _parse_hits(response)


def _parse_hits(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        points = response["result"]["points"]
    except (KeyError, TypeError) as exc:
        raise RagBackendError("Qdrant 응답 형식이 올바르지 않습니다.") from exc
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise RagBackendError("Qdrant 검색 결과 형식이 올바르지 않습니다.")

    hits: list[dict[str, Any]] = []
    for point in points:
        try:
            point_id = str(point["id"])
            score = float(point["score"])
            payload = point["payload"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RagBackendError("Qdrant 검색 결과에 필수 필드가 없습니다.") from exc
        if not math.isfinite(score) or not isinstance(payload, Mapping):
            raise RagBackendError("Qdrant 검색 결과 값이 올바르지 않습니다.")

        source_type = payload.get("source_type")
        source_id = payload.get("kb_source_id")
        title = payload.get("title")
        locator = payload.get("locator")
        source = f"{title} ({locator})" if title and locator else title
        hits.append(
            {
                "id": point_id,
                "doc_id": (
                    f"{source_type}:{source_id}"
                    if source_type is not None and source_id is not None
                    else None
                ),
                "source": source,
                "text": payload.get("text"),
                "score": score,
            }
        )
    return hits
