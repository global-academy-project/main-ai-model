from __future__ import annotations

import copy
import html
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.exceptions import ToolError
from app.rag.tags import classify_regulations
from app.tools.tool_registry import ToolRegistry

_BASE_URL = "https://apis.data.go.kr/B552468/msdschem1"
_CAS_PATTERN = re.compile(r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")
_MAX_MATERIALS = 20
_CACHE_TTL_SECONDS = 6 * 60 * 60
_SOURCE = "한국산업안전보건공단 물질안전보건자료 조회 서비스"

MSDS_PARAMETERS = {
    "type": "object",
    "properties": {
        "materials": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _MAX_MATERIALS,
        }
    },
    "required": ["materials"],
}


class KoshaMsdsClient:
    def __init__(self, service_key: str, *, timeout_seconds: float = 10.0) -> None:
        service_key = service_key.strip()
        if not service_key:
            raise ValueError("공공데이터 API 키가 필요합니다.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds는 양수여야 합니다.")
        self._service_key = service_key
        self._timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[float, dict[str, object]]] = {}
        self._cache_lock = threading.Lock()

    def lookup(self, arguments: Mapping[str, object]) -> dict[str, object]:
        materials = arguments.get("materials")
        if not isinstance(materials, list) or not all(isinstance(item, str) for item in materials):
            raise ToolError(
                "TOOL_INPUT_INVALID",
                "lookup_msds에는 물질 이름 또는 CAS 번호 목록(materials)이 필요합니다.",
                tool="lookup_msds",
            )
        if len(materials) > _MAX_MATERIALS:
            raise ToolError(
                "TOOL_INPUT_INVALID",
                f"물질은 한 번에 {_MAX_MATERIALS}개까지 조회할 수 있습니다.",
                tool="lookup_msds",
            )

        return {"results": [self._lookup_one(query) for query in materials]}

    def _lookup_one(self, query: str) -> dict[str, object]:
        normalized_query = query.strip()
        if not normalized_query:
            return {"query": query, "found": False}

        cache_key = normalized_query.casefold()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None and cached[0] > time.monotonic():
                return copy.deepcopy(cached[1])

        is_cas = _CAS_PATTERN.fullmatch(normalized_query) is not None
        candidates = self._request_items(
            "getChemList001",
            {
                "searchWrd": normalized_query,
                "searchCnd": "1" if is_cas else "0",
                "numOfRows": "100",
                "pageNo": "1",
            },
        )
        chemical = self._select_exact_match(candidates, normalized_query, is_cas)
        if chemical is None or not chemical.get("chemId"):
            result: dict[str, object] = {"query": query, "found": False}
        else:
            result = self._build_result(query, chemical)

        with self._cache_lock:
            self._cache[cache_key] = (time.monotonic() + _CACHE_TTL_SECONDS, copy.deepcopy(result))
        return result

    def _build_result(self, query: str, chemical: dict[str, str]) -> dict[str, object]:
        chem_id = chemical["chemId"]
        hazard_items = self._request_items("getChemDetail021", {"chemId": chem_id})
        first_aid_items = self._request_items("getChemDetail041", {"chemId": chem_id})
        regulation_items = self._request_items("getChemDetail151", {"chemId": chem_id})

        hazards = _format_details(hazard_items)
        ghs_classes = [
            _clean_text(item.get("itemDetail", ""))
            for item in hazard_items
            if "분류" in item.get("msdsItemNameKor", "") and _clean_text(item.get("itemDetail", ""))
        ]
        regulations = _format_details(regulation_items)
        return {
            "query": query,
            "found": True,
            "name": chemical.get("chemNameKor") or query,
            "cas_no": chemical.get("casNo") or None,
            "ghs_classes": ghs_classes,
            "hazards": hazards,
            "first_aid": "\n".join(_format_details(first_aid_items)) or None,
            "regulations": regulations,
            "regulatory_tags": classify_regulations(regulations),
            "source": _SOURCE,
        }

    def _select_exact_match(
        self,
        candidates: list[dict[str, str]],
        query: str,
        is_cas: bool,
    ) -> dict[str, str] | None:
        if is_cas:
            matches = [item for item in candidates if item.get("casNo", "").strip() == query]
        else:
            normalized_name = _normalize_name(query)
            matches = [
                item for item in candidates
                if _normalize_name(item.get("chemNameKor", "")) == normalized_name
            ]
        if not matches:
            return None

        cas_numbers = {item.get("casNo", "").strip() for item in matches}
        if len(cas_numbers) > 1:
            return None
        return max(matches, key=lambda item: item.get("lastDate", ""))

    def _request_items(self, operation: str, parameters: Mapping[str, str]) -> list[dict[str, str]]:
        query = urlencode({"serviceKey": self._service_key, **parameters})
        request = Request(f"{_BASE_URL}/{operation}?{query}", headers={"Accept": "application/xml"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                root = ET.fromstring(response.read())
        except (URLError, TimeoutError, OSError, ET.ParseError, UnicodeDecodeError):
            raise ToolError(
                "MSDS_UPSTREAM_ERROR",
                "공단 MSDS API에 연결하거나 응답을 처리하지 못했습니다.",
                tool="lookup_msds",
            ) from None

        result_code = _find_text(root, "resultCode")
        if result_code and result_code not in {"00", "0000"}:
            raise ToolError(
                "MSDS_UPSTREAM_ERROR",
                "공단 MSDS API가 오류 응답을 반환했습니다.",
                tool="lookup_msds",
            )

        return [
            {
                _local_name(child.tag): "".join(child.itertext()).strip()
                for child in item
            }
            for item in root.iter()
            if _local_name(item.tag) == "item" and len(item)
        ]


def register_publicdata_msds(registry: ToolRegistry) -> None:
    service_key = os.getenv("KOSHA_MSDS_SERVICE_KEY", "").strip()
    if not service_key:
        return

    client = KoshaMsdsClient(service_key)
    registry.register(
        "lookup_msds",
        client.lookup,
        owner="A",
        description="공단 MSDS API에서 물질명 또는 CAS 번호로 유해성·응급조치 정보를 조회합니다.",
        parameters=MSDS_PARAMETERS,
        replace=True,
    )


def _format_details(items: list[dict[str, str]]) -> list[str]:
    details = []
    for item in items:
        value = _clean_text(item.get("itemDetail", ""))
        if not value:
            continue
        label = _clean_text(item.get("msdsItemNameKor", ""))
        details.append(f"{label}: {value}" if label else value)
    return details


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]*>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _find_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _local_name(element.tag) == name and element.text:
            return element.text.strip()
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_name(value: str) -> str:
    return "".join(value.split()).casefold()