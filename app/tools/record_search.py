"""Spring 내부 기록 조회 도구."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.exceptions import ToolError
from app.integrations import InternalApiClient

SEARCH_RECORDS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "run_id": {"type": "integer", "minimum": 1},
        "site_id": {"type": "integer", "minimum": 1},
        "query_type": {"type": "string"},
        "params": {"type": "object"},
    },
    "required": ["run_id", "site_id", "query_type", "params"],
}


def make_search_records_tool(client: InternalApiClient):
    def search_records(arguments: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("run_id", "site_id", "query_type"):
            if not arguments.get(key):
                raise ToolError("TOOL_INPUT_INVALID", f"{key}가 필요합니다.", tool="search_records")
        params = arguments.get("params", {})
        if not isinstance(params, Mapping):
            raise ToolError("TOOL_INPUT_INVALID", "params는 객체여야 합니다.", tool="search_records")
        return client.call_tool(
            "search_records",
            {
                "siteId": int(arguments["site_id"]),
                "agentRunId": int(arguments["run_id"]),
                "queryType": str(arguments["query_type"]),
                "params": dict(params),
            },
            retry=False,
        )

    return search_records
