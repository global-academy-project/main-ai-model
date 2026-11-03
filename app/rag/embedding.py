from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from app.embedding.config import RagSettings
from app.embedding.errors import RagBackendError
from app.embedding.http import post_json
from app.embedding.types import HybridEmbedding, SparseVector

_MAX_BATCH_SIZE = 32


def embed_text(settings: RagSettings, text: str) -> HybridEmbedding:
    return _request_embeddings(settings, [text], input_value=text)[0]


def embed_texts(settings: RagSettings, texts: Sequence[str]) -> list[HybridEmbedding]:
    return _request_embeddings(settings, texts, input_value=list(texts))


def _request_embeddings(
    settings: RagSettings,
    texts: Sequence[str],
    *,
    input_value: str | list[str],
) -> list[HybridEmbedding]:
    if not 1 <= len(texts) <= _MAX_BATCH_SIZE:
        raise ValueError(f"임베딩 요청은 1~{_MAX_BATCH_SIZE}개 텍스트여야 합니다.")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    response = post_json(
        settings.embedding_url,
        {"model": settings.embedding_model, "input": input_value},
        headers,
        settings.timeout_seconds,
        "임베딩",
    )
    try:
        data = response["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise TypeError
        embeddings: list[HybridEmbedding] = []
        for expected_index, item in enumerate(data):
            if not isinstance(item, Mapping) or item.get("index") != expected_index:
                raise TypeError
            dense = item["embedding"]
            sparse = item["sparse_embedding"]
            if not isinstance(dense, list) or not isinstance(sparse, Mapping):
                raise TypeError
            indices = sparse["indices"]
            values = sparse["values"]
            if (
                len(dense) != settings.embedding_dim
                or not all(_is_finite_number(value) for value in dense)
                or not isinstance(indices, list)
                or not isinstance(values, list)
                or not indices
                or len(indices) != len(values)
                or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
                or indices != sorted(set(indices))
                or any(not _is_finite_number(value) or value <= 0 for value in values)
            ):
                raise ValueError
            embeddings.append(
                HybridEmbedding(
                    dense=[float(value) for value in dense],
                    sparse=SparseVector(indices=indices, values=[float(value) for value in values]),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise RagBackendError("하이브리드 임베딩 응답 형식이 올바르지 않습니다.") from exc
    return embeddings


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )