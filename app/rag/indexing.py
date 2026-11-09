from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import RagSettings
from .embedding import embed_texts
from .qdrant import make_point, upsert_points

MAX_INDEX_BATCH_SIZE = 32


@dataclass(frozen=True)
class IndexableChunk:
    point_id: str | int
    text: str
    payload: Mapping[str, Any]


def index_chunks(
    settings: RagSettings,
    collection: str,
    chunks: Sequence[IndexableChunk],
    *,
    write_api_key: str,
    batch_size: int = MAX_INDEX_BATCH_SIZE,
) -> int:
    if not 1 <= batch_size <= MAX_INDEX_BATCH_SIZE:
        raise ValueError(f"batch_size는 1~{MAX_INDEX_BATCH_SIZE}이어야 합니다.")

    indexed = 0
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        if any(not chunk.text.strip() for chunk in batch):
            raise ValueError("빈 청크는 적재할 수 없습니다.")
        embeddings = embed_texts(settings, [chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise RagBackendError("임베딩 서버가 청크 수와 다른 결과를 반환했습니다.")
        points = [
            make_point(chunk.point_id, embedding, chunk.payload)
            for chunk, embedding in zip(batch, embeddings)
        ]
        upsert_points(settings, collection, points, write_api_key=write_api_key)
        indexed += len(batch)
    return indexed