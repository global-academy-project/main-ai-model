"""기록질문 에이전트가 사용하는 Spring ``search_records`` 어댑터."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.exceptions import ToolError
from app.integrations import InternalApiClient


def make_search_records_tool(client: InternalApiClient):
    def search_records(arguments: Mapping[str, Any]) -> dict[str, Any]:
        run_id = arguments.get("run_id")
        site_id = arguments.get("site_id")
        query_type = arguments.get("query_type")
        params = arguments.get("params")
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise ToolError("TOOL_INPUT_INVALID", "run_id가 필요합니다.", tool="search_records")
        if not isinstance(site_id, int) or isinstance(site_id, bool) or site_id <= 0:
            raise ToolError("TOOL_INPUT_INVALID", "site_id가 필요합니다.", tool="search_records")
        if not isinstance(query_type, str) or not query_type:
            raise ToolError("TOOL_INPUT_INVALID", "query_type이 필요합니다.", tool="search_records")
        if not isinstance(params, Mapping):
            raise ToolError("TOOL_INPUT_INVALID", "params는 객체여야 합니다.", tool="search_records")
        return client.call_tool(
            "search_records",
            {
                # 이 두 값은 LLM 계획에서 받지 않고 실행 요청 값만 사용한다.
                "siteId": site_id,
                "agentRunId": run_id,
                "queryType": query_type,
                "params": dict(params),
            },
            retry=False,
        )

    return search_records
