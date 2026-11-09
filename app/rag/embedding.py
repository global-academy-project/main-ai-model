from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .config import RagSettings
from .errors import RagBackendError
from .http import post_json


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridEmbedding:
    dense: list[float]
    sparse: SparseVector


def embed_text(settings: RagSettings, text: str) -> HybridEmbedding:
    return embed_texts(settings, [text])[0]


def embed_texts(settings: RagSettings, texts: list[str]) -> list[HybridEmbedding]:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("임베딩할 문장이 필요합니다.")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    response = post_json(
        settings.embedding_url,
        {"model": settings.embedding_model, "input": texts[0] if len(texts) == 1 else texts},
        headers,
        settings.timeout_seconds,
        "임베딩",
    )
    try:
        data = response["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise TypeError
        embeddings = []
        for expected_index, item in enumerate(data):
            if not isinstance(item, Mapping) or item.get("index") != expected_index:
                raise TypeError
            dense = item["embedding"]
            sparse = item["sparse_embedding"]
            if not isinstance(sparse, Mapping):
                raise TypeError
            indices = sparse["indices"]
            values = sparse["values"]
            if (
                not isinstance(indices, list)
                or not isinstance(values, list)
                or len(indices) != len(values)
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in indices
                )
                or indices != sorted(set(indices))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in values
                )
            ):
                raise TypeError
            if (
                not isinstance(dense, list)
                or len(dense) != settings.embedding_dim
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in dense
                )
            ):
                raise TypeError
            embeddings.append(
                HybridEmbedding(
                    dense=[float(value) for value in dense],
                    sparse=SparseVector(
                        indices=indices,
                        values=[float(value) for value in values],
                    ),
                )
            )
    except (KeyError, IndexError, TypeError) as exc:
        raise RagBackendError("임베딩 서버 응답 형식이 올바르지 않습니다.") from exc

    return embeddings
