"""Spring 실행 요청을 받아 정형화된 기록 조회를 수행하는 E 에이전트."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.evidence_collector.textutil import extract_json_object
from app.exceptions import ToolError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.record_query import RecordQueryRequest, RecordQueryResult
from app.tools.tool_contract import ToolInvoker
from app.tools.tool_registry import to_plain

logger = logging.getLogger(__name__)

QUERY_TYPES = (
    "missing_measurements", "permit_approval_statistics", "pending_approvals",
    "rejected_revisions", "open_corrective_actions", "failed_inspections",
    "missing_tbm", "risk_assessment_rate", "approver_statistics",
    "record_change_history", "blocked_records",
)

_PLAN_SYSTEM = """너는 산업안전 기록 조회 계획을 만드는 보조자다.
SQL을 만들지 말고 허용된 queryType 하나와 params만 선택한다.
모든 조회에는 YYYY-MM-DD 형식의 from과 to가 필요하다.
JSON 객체 하나만 출력한다.
허용 queryType: """ + ", ".join(QUERY_TYPES) + "\n" + "예시: {\"queryType\": \"missing_tbm\", \"params\": {\"from\": \"2026-10-01\", \"to\": \"2026-10-31\"}}"
class RecordQueryAgent:
    def __init__(self, *, llm: LLMClient, tools: ToolInvoker) -> None:
        self._llm = llm
        self._tools = tools

    def run(self, request: RecordQueryRequest) -> RecordQueryResult:
        logger.info("record-query 계획 생성: scenario=E runId=%s siteId=%s", request.run_id, request.site_id)
        steps: list[dict[str, Any]] = []
        plan = self._plan(request)
        if plan is None:
            logger.info("record-query 추가 질문 필요: scenario=E runId=%s", request.run_id)
            return RecordQueryResult(
                run_status="WAITING_INPUT",
                ask={
                    "message": "조회 기간과 찾으려는 기록을 조금 더 구체적으로 알려 주세요.",
                    "missingFields": [],
                },
                context={"message": request.message},
                steps=steps,
            )

        steps.append(self._step("PHASE", "query.plan", {"message": request.message}, plan))
        logger.info(
            "record-query 계획 확정: scenario=E runId=%s queryType=%s from=%s to=%s",
            request.run_id,
            plan["queryType"],
            plan["params"].get("from"),
            plan["params"].get("to"),
        )
        try:
            logger.info("record-query 도구 호출: scenario=E runId=%s tool=search_records", request.run_id)
            raw = to_plain(self._tools.call("search_records", {
                "run_id": request.run_id,
                "site_id": request.site_id,
                "query_type": plan["queryType"],
                "params": plan["params"],
            }))
        except ToolError:
            logger.exception("record-query 도구 실패: scenario=E runId=%s tool=search_records", request.run_id)
            raise
        if not isinstance(raw, dict):
            raise ToolError("TOOL_OUTPUT_INVALID", "search_records 결과 형식이 올바르지 않습니다.", tool="search_records")

        query = {
            "queryType": raw.get("queryType", plan["queryType"]),
            "conditions": raw.get("conditions", plan["params"]),
            "total": raw.get("total", 0),
            "truncated": bool(raw.get("truncated", False)),
        }
        evidence = [
            {"type": item.get("type"), "id": item.get("id"), "label": item.get("label")}
            for item in raw.get("evidence", [])
            if isinstance(item, dict) and item.get("type") and item.get("id") is not None
        ]
        steps.append(self._step("TOOL", "search_records", {
            "queryType": plan["queryType"], "params": plan["params"]
        }, {"total": query["total"], "truncated": query["truncated"]}))
        # E 모델은 조회 계획을 고르는 데 한 번만 쓴다. 답변은 Spring이 돌려준 검증된
        # 결과를 그대로 요약해 4B 모델을 다시 호출하지 않는다.
        answer = self._answer(raw)
        logger.info(
            "record-query 답변 생성: scenario=E runId=%s source=tool-result total=%s evidenceCount=%s",
            request.run_id,
            query["total"],
            len(evidence),
        )
        steps.append(self._step("PHASE", "verification", {"evidence": len(evidence)}, {"answer": bool(answer)}))
        for seq, step in enumerate(steps, start=1):
            step["seq"] = seq
        return RecordQueryResult(
            run_status="DONE",
            answer=answer,
            queries=[query],
            evidence=evidence,
            context={"lastQuery": plan},
            steps=steps,
        )

    def _plan(self, request: RecordQueryRequest) -> dict[str, Any] | None:
        context = json.dumps(request.context, ensure_ascii=False) if request.context else "없음"
        user = f"질문:\n{request.message}\n이전 context:\n{context}"
        response = self._llm.chat([Message(role="user", content=user)], system=_PLAN_SYSTEM, max_tokens=350)
        raw = extract_json_object(response.text or "") if response.stop_reason != "max_tokens" else None
        if not raw:
            logger.info("기록질문 계획을 해석하지 못했습니다.")
            return None
        query_type = raw.get("queryType")
        params = raw.get("params")
        if query_type not in QUERY_TYPES or not isinstance(params, dict):
            return None
        if not isinstance(params.get("from"), str) or not isinstance(params.get("to"), str):
            return None
        return {"queryType": query_type, "params": params}

    @staticmethod
    def _answer(result: dict[str, Any]) -> str:
        """도구가 반환한 검증된 수치만 사용해 답한다. 추가 LLM 호출은 하지 않는다."""
        total = result.get("total", 0)
        total = total if isinstance(total, int) and total >= 0 else 0
        if total == 0:
            return "조회 조건에 맞는 기록은 0건입니다."

        summary = result.get("summary")
        if isinstance(summary, dict):
            parts = [f"{key}: {value}" for key, value in summary.items() if value is not None]
            if parts:
                return f"조회 결과는 {total}건입니다. " + ", ".join(parts[:5])
        return f"조회 결과는 {total}건입니다. 아래 근거 기록을 확인하세요."

    @staticmethod
    def _step(kind: str, name: str, input_data: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        return {
            "seq": 0,
            "kind": kind,
            "name": name,
            "toolName": "search_records" if kind == "TOOL" else None,
            "status": "OK",
            "input": input_data,
            "output": output,
            "elapsedMs": 0,
        }
