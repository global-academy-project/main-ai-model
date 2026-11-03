"""Qdrant-backed RAG search backend factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.embedding.config import RagSettings
from app.embedding.errors import RagBackendError
from app.embedding.filters import make_qdrant_conditions, read_filters

from .embedding import embed_text
from .qdrant import query_points

BACKEND_ERRORS = (RagBackendError,)


def build_search_backend():
    """Build the callable expected by the RAG search tool adapter."""
    settings = RagSettings.from_env()

    def search(
        query: str,
        *,
        k: int,
        filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        collections, payload_filters, site_ids = read_filters(filters)
        embedding = embed_text(settings, query)
        hits: list[dict[str, Any]] = []

        for collection in collections:
            conditions = make_qdrant_conditions(collection, payload_filters, site_ids)
            hits.extend(query_points(settings, collection, embedding, conditions, k))

        hits.sort(key=lambda hit: -hit["score"])
        return hits[:k]

    return search