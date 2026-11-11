from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from .config import RagSettings
from .embedding import HybridEmbedding
from .errors import RagBackendError
from .filters import COLLECTIONS
from .http import post_json, put_json

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
    embedding: HybridEmbedding | list[float],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    dense = embedding.dense if isinstance(embedding, HybridEmbedding) else embedding
    vector: dict[str, Any] = {"dense": dense}
    if isinstance(embedding, HybridEmbedding) and embedding.sparse.indices:
        vector["sparse"] = {
            "indices": embedding.sparse.indices,
            "values": embedding.sparse.values,
        }
    return {
        "id": point_id,
        "vector": vector,
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

    url = f"{settings.qdrant_url}/collections/{quote(collection, safe='')}/points?wait=true"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "api-key": write_api_key,
    }
    response = put_json(
        url,
        {"points": list(points)},
        headers,
        settings.timeout_seconds,
        "Qdrant 적재",
    )
    if response.get("status") != "ok":
        raise RagBackendError("Qdrant 적재가 완료되지 않았습니다.")


def delete_points(
    settings: RagSettings,
    collection: str,
    point_ids: Sequence[str | int],
    *,
    write_api_key: str,
) -> None:
    if collection not in COLLECTIONS:
        raise ValueError("collection은 kb_accidents 또는 kb_guides여야 합니다.")
    if not point_ids:
        return
    if not write_api_key.strip():
        raise ValueError("Qdrant 쓰기 키가 필요합니다.")

    url = (
        f"{settings.qdrant_url}/collections/"
        f"{quote(collection, safe='')}/points/delete?wait=true"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "api-key": write_api_key,
    }
    response = post_json(
        url,
        {"points": list(point_ids)},
        headers,
        settings.timeout_seconds,
        "Qdrant 삭제",
    )
    if response.get("status") != "ok":
        raise RagBackendError("Qdrant 삭제가 완료되지 않았습니다.")


def query_points(
    settings: RagSettings,
    collection: str,
    vector: HybridEmbedding | list[float],
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

    if isinstance(vector, HybridEmbedding) and vector.sparse.indices:
        query = {
            "prefetch": [
                {
                    "query": vector.dense,
                    "using": "dense",
                    "filter": {"must": conditions},
                    "limit": limit,
                },
                {
                    "query": {
                        "indices": vector.sparse.indices,
                        "values": vector.sparse.values,
                    },
                    "using": "sparse",
                    "filter": {"must": conditions},
                    "limit": limit,
                },
            ],
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": _PAYLOAD_FIELDS,
        }
    else:
        dense = vector.dense if isinstance(vector, HybridEmbedding) else vector
        query = {
            "query": dense,
            "using": "dense",
            "filter": {"must": conditions},
            "limit": limit,
            "with_payload": _PAYLOAD_FIELDS,
        }

    response = post_json(
        url,
        query,
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
