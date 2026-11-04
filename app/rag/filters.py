from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COLLECTIONS: dict[str, tuple[str, ...]] = {
    "kb_accidents": ("ACCIDENT_REPORT", "SIF", "NEAR_MISS"),
    "kb_guides": ("KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"),
}

_FILTER_FIELDS = frozenset(
    {
        "severity",
        "accident_type",
        "hazard",
        "work_type",
        "cas",
        "ghs",
        "year",
        "guide_no",
    }
)


def read_filters(
    filters: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any], list[int]]:
    if not isinstance(filters, Mapping):
        raise ValueError("RAG filters는 mapping이어야 합니다.")

    requested_collection = filters.get("collection")
    if requested_collection is not None:
        if not isinstance(requested_collection, str) or requested_collection not in COLLECTIONS:
            raise ValueError("collection은 kb_accidents 또는 kb_guides여야 합니다.")
        collections = [requested_collection]
    else:
        collections = list(COLLECTIONS)

    payload_filters: dict[str, Any] = {}
    for key, value in filters.items():
        if key in {"collection", "site_id", "source_type"}:
            continue
        if key not in _FILTER_FIELDS:
            raise ValueError(f"지원하지 않는 RAG 필터입니다: {key}")
        if value is not None:
            payload_filters[key] = value

    source_types = _filter_values(filters.get("source_type"), "source_type")
    if source_types:
        known_types = {
            source_type
            for collection_types in COLLECTIONS.values()
            for source_type in collection_types
        }
        if any(not isinstance(value, str) or value not in known_types for value in source_types):
            raise ValueError("지원하지 않는 source_type 필터 값입니다.")

        matching = [
            name for name in collections if any(value in COLLECTIONS[name] for value in source_types)
        ]
        if not matching or any(
            not any(value in COLLECTIONS[name] for name in matching) for value in source_types
        ):
            raise ValueError("source_type과 collection 필터가 서로 맞지 않습니다.")
        collections = matching
        payload_filters["source_type"] = source_types

    return collections, payload_filters, _read_site_ids(filters.get("site_id"))


def make_qdrant_conditions(
    collection: str,
    payload_filters: Mapping[str, Any],
    site_ids: list[int],
) -> list[dict[str, Any]]:
    conditions = [
        {"key": "flagged", "match": {"value": False}},
        {"key": "site_id", "match": {"any": site_ids}},
        {"key": "source_type", "match": {"any": list(COLLECTIONS[collection])}},
    ]
    conditions.extend(
        _to_qdrant_condition(key, value) for key, value in payload_filters.items()
    )
    return conditions


def _read_site_ids(value: Any) -> list[int]:
    if value is None:
        return [0]
    values = value if isinstance(value, (list, tuple, set)) else [value]
    site_ids = [0]
    for site_id in values:
        if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id <= 0:
            raise ValueError("site_id 필터는 양의 정수여야 합니다.")
        if site_id not in site_ids:
            site_ids.append(site_id)
    return site_ids


def _filter_values(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    if not values:
        return []
    if any(item is None or isinstance(item, (dict, list, tuple, set)) for item in values):
        raise ValueError(f"{name} 필터 값이 올바르지 않습니다.")
    return values


def _to_qdrant_condition(key: str, value: Any) -> dict[str, Any]:
    values = _filter_values(value, key)
    if not values:
        raise ValueError(f"{key} 필터 값은 비어 있을 수 없습니다.")
    match = {"value": values[0]} if len(values) == 1 else {"any": values}
    return {"key": key, "match": match}
