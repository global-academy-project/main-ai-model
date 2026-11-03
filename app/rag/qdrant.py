from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from app.embedding.config import RagSettings
from app.embedding.errors import RagBackendError
from app.embedding.filters import COLLECTIONS
from app.embedding.http import post_json, put_json
from app.embedding.types import HybridEmbedding

_PAYLOAD_FIELDS = [
    "kb_source_id",
    "kb_chunk_id",
    "source_type",
    "site_id",
    "title",
    "locator",
    "url",
    "text",
]


def make_point(
    point_id: str | int,
    embedding: HybridEmbedding,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": point_id,
        "vector": {
            "dense": embedding.dense,
            "sparse": {
                "indices": embedding.sparse.indices,
                "values": embedding.sparse.values,
            },
        },
        "payload": dict(payload),
    }


def upsert_points(
    settings: RagSettings,
    collection: str,
    points: Sequence[Mapping[str, Any]],
    *,
    write_api_key: str,
) -> None:
    if collection not in COLLECTIONS:
        raise ValueError("collection은 kb_accidents 또는 kb_guides여야 합니다.")
    if not points:
        return
    if not write_api_key.strip():
        raise ValueError("Qdrant 쓰기 키가 필요합니다.")

    url = (
        f"{settings.qdrant_url}/collections/"
        f"{quote(collection, safe='')}/points?wait=true"
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json", "api-key": write_api_key}
    response = put_json(
        url,
        {"points": list(points)},
        headers,
        settings.timeout_seconds,
        "Qdrant 적재",
    )
    if response.get("status") != "ok":
        raise RagBackendError("Qdrant 적재가 완료되지 않았습니다.")


def query_points(
    settings: RagSettings,
    collection: str,
    embedding: HybridEmbedding,
    conditions: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if collection not in COLLECTIONS:
        raise ValueError("collection은 kb_accidents 또는 kb_guides여야 합니다.")
    if limit <= 0:
        raise ValueError("검색 limit는 양수여야 합니다.")
    url = (
        f"{settings.qdrant_url}/collections/"
        f"{quote(collection, safe='')}/points/query"
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    qdrant_filter = {"must": conditions}
    response = post_json(
        url,
        {
            "prefetch": [
                {
                    "query": embedding.dense,
                    "using": "dense",
                    "filter": qdrant_filter,
                    "limit": limit,
                },
                {
                    "query": {
                        "indices": embedding.sparse.indices,
                        "values": embedding.sparse.values,
                    },
                    "using": "sparse",
                    "filter": qdrant_filter,
                    "limit": limit,
                },
            ],
            "query": {"fusion": "rrf"},
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