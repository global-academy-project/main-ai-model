"""검증된 Spring 기록만 사용해 답하는 기록질문 에이전트."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.evidence_collector.textutil import extract_json_object
from app.exceptions import AgentRunError, LLMError, ToolError
from app.exceptions.llm_error_info import to_error_info
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.record_query import (
    RecordEvidence,
    RecordQueryAsk,
    RecordQueryItem,
    RecordQueryRequest,
    RecordQueryResult,
    RecordQueryStep,
)

logger = logging.getLogger(__name__)

SearchRecords = Callable[[Mapping[str, Any]], dict[str, Any]]

QUERY_PARAMS: dict[str, frozenset[str]] = {
    "missing_measurements": frozenset({"permitType"}),
    "permit_approval_statistics": frozenset({"permitType"}),
    "pending_approvals": frozenset({"targetType", "stepNo"}),
    "rejected_revisions": frozenset({"targetType"}),
    "open_corrective_actions": frozenset({"status"}),
    "failed_inspections": frozenset({"zoneId", "templateCode"}),
    "missing_tbm": frozenset({"workType", "zoneId"}),
    "risk_assessment_rate": frozenset({"workType"}),
    "approver_statistics": frozenset({"approverId"}),
    "record_change_history": frozenset({"recordType", "recordId", "userId"}),
    "blocked_records": frozenset({"permitType"}),
}

QUERY_LABELS = {
    "missing_measurements": "가스측정값 누락 기록",
    "permit_approval_statistics": "작업허가 승인 통계",
    "pending_approvals": "승인 대기 기록",
    "rejected_revisions": "반려 후 수정 기록",
    "open_corrective_actions": "미완료 시정조치",
    "failed_inspections": "부적합 점검 기록",
    "missing_tbm": "TBM 누락 기록",
    "risk_assessment_rate": "위험성평가 실시 현황",
    "approver_statistics": "승인자 처리 통계",
    "record_change_history": "기록 변경 이력",
    "blocked_records": "차단 판정 기록",
}

_SYSTEM_PROMPT = """너는 산업안전 기록 질문을 정형 조회 계획으로 바꾸는 계획자다.
반드시 JSON 객체 하나만 출력한다. SQL, siteId, agentRunId, 총 건수, 기록 ID는 만들지 않는다.
출력 형식: {"queryType":"허용 유형", "params":{"from":"YYYY-MM-DD", "to":"YYYY-MM-DD"}}
from과 to는 양 끝을 포함하며 항상 필요하고 최대 366일이다. 상대 날짜는 제공된 오늘 날짜를 기준으로 계산한다.
질문이 이전 조회를 이어 묻는 경우 이전 context의 queryType과 params를 참고한다.
지원할 수 없거나 기간을 결정할 정보가 전혀 없으면 {"needsClarification":true}를 출력한다.

허용 유형과 추가 조건:
- missing_measurements: permitType
- permit_approval_statistics: permitType
- pending_approvals: targetType, stepNo
- rejected_revisions: targetType
- open_corrective_actions: status
- failed_inspections: zoneId, templateCode
- missing_tbm: workType, zoneId
- risk_assessment_rate: workType
- approver_statistics: approverId
- record_change_history: recordType, recordId, userId
- blocked_records: permitType
목록에 없는 조건은 넣지 않는다."""


class RecordQueryAgent:
    def __init__(self, *, llm: LLMClient, search_records: SearchRecords) -> None:
        self._llm = llm
        self._search_records = search_records

    def run(self, request: RecordQueryRequest) -> RecordQueryResult:
        steps: list[RecordQueryStep] = []
        plan_started = time.perf_counter()
        try:
            plan = self._plan(request)
        except (LLMError, ToolError) as exc:
            steps.append(self._error_step("PHASE", "query.plan", exc, plan_started))
            raise AgentRunError(exc, steps=steps) from exc

        if plan is None:
            return RecordQueryResult(
                run_status="WAITING_INPUT",
                ask=RecordQueryAsk(
                    message="조회할 기록 종류와 기간을 알려 주세요. 예: 지난달 가스측정값 누락 기록"
                ),
                context={"previousQuestion": request.message, **(request.context or {})},
            )

        steps.append(
            self._ok_step(
                "PHASE",
                "query.plan",
                {"message": request.message},
                plan,
                plan_started,
            )
        )
        tool_input = {
            "run_id": request.run_id,
            "site_id": request.site_id,
            "query_type": plan["queryType"],
            "params": plan["params"],
        }
        tool_started = time.perf_counter()
        try:
            raw = self._search_records(tool_input)
            validated = self._validate_result(raw, request=request, plan=plan)
        except ToolError as exc:
            steps.append(self._error_step("TOOL", "search_records", exc, tool_started, tool_input))
            self._renumber(steps)
            raise AgentRunError(exc, steps=steps) from exc

        query, evidence = self._response_items(validated)
        steps.append(
            self._ok_step(
                "TOOL",
                "search_records",
                {"queryType": plan["queryType"], "params": plan["params"]},
                {"total": query.total, "truncated": query.truncated},
                tool_started,
            )
        )
        verification_started = time.perf_counter()
        answer = self._answer(validated, query)
        steps.append(
            self._ok_step(
                "PHASE",
                "verification",
                {"evidenceCount": len(evidence)},
                {"answerCreated": True, "source": "search_records"},
                verification_started,
            )
        )
        self._renumber(steps)
        return RecordQueryResult(
            run_status="DONE",
            answer=answer,
            queries=[query],
            evidence=evidence,
            context={
                "lastQueryType": plan["queryType"],
                "lastParams": plan["params"],
                "lastQuestion": request.message,
            },
            steps=steps,
        )

    def _plan(self, request: RecordQueryRequest) -> dict[str, Any] | None:
        today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        previous = json.dumps(request.context, ensure_ascii=False) if request.context else "없음"
        response = self._llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        f"오늘: {today}\n현장 모드: {request.mode}\n"
                        f"이전 context: {previous}\n질문: {request.message}"
                    ),
                )
            ],
            system=_SYSTEM_PROMPT,
            max_tokens=350,
        )
        if response.stop_reason == "max_tokens":
            return None
        raw = extract_json_object(response.text or "")
        if not raw or raw.get("needsClarification") is True:
            return None
        query_type = raw.get("queryType")
        params = raw.get("params")
        if query_type not in QUERY_PARAMS or not isinstance(params, dict):
            return None
        if not self._valid_params(query_type, params):
            return None
        return {"queryType": query_type, "params": params}

    @staticmethod
    def _valid_params(query_type: str, params: dict[str, Any]) -> bool:
        allowed = QUERY_PARAMS[query_type] | {"from", "to"}
        if set(params) - allowed:
            return False
        try:
            start = date.fromisoformat(params["from"])
            end = date.fromisoformat(params["to"])
        except (KeyError, TypeError, ValueError):
            return False
        return start <= end and (end - start).days <= 365

    @staticmethod
    def _validate_result(
        raw: object, *, request: RecordQueryRequest, plan: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 결과 형식이 올바르지 않습니다.", tool="search_records"
            )
        if raw.get("queryType") != plan["queryType"]:
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 유형이 요청과 다릅니다.", tool="search_records"
            )
        conditions = raw.get("conditions")
        total = raw.get("total")
        truncated = raw.get("truncated")
        if not isinstance(conditions, dict) or conditions.get("siteId") != request.site_id:
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 조건이 올바르지 않습니다.", tool="search_records"
            )
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 건수가 올바르지 않습니다.", tool="search_records"
            )
        if not isinstance(truncated, bool):
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 잘림 정보가 올바르지 않습니다.", tool="search_records"
            )
        if not isinstance(raw.get("records", []), list) or not isinstance(raw.get("evidence", []), list):
            raise ToolError(
                "TOOL_OUTPUT_INVALID", "Spring 기록 조회 목록이 올바르지 않습니다.", tool="search_records"
            )
        return raw

    @classmethod
    def _response_items(
        cls, raw: dict[str, Any]
    ) -> tuple[RecordQueryItem, list[RecordEvidence]]:
        query = RecordQueryItem(
            query_type=raw["queryType"],
            conditions=raw["conditions"],
            total=raw["total"],
            truncated=raw["truncated"],
        )
        records = [item for item in raw.get("records", []) if isinstance(item, dict)]
        evidence: list[RecordEvidence] = []
        seen: set[tuple[str, int]] = set()
        for item in raw.get("evidence", []):
            if not isinstance(item, dict):
                continue
            evidence_type, evidence_id = item.get("type"), item.get("id")
            if not isinstance(evidence_type, str) or not isinstance(evidence_id, int) or isinstance(evidence_id, bool):
                raise ToolError(
                    "TOOL_OUTPUT_INVALID", "Spring 기록 근거 형식이 올바르지 않습니다.", tool="search_records"
                )
            key = (evidence_type, evidence_id)
            if key in seen:
                continue
            try:
                evidence.append(
                    RecordEvidence(
                        type=evidence_type,
                        id=evidence_id,
                        label=cls._evidence_label(item, records, evidence_type, evidence_id),
                    )
                )
            except (TypeError, ValueError):
                raise ToolError(
                    "TOOL_OUTPUT_INVALID", "Spring 기록 근거 형식이 올바르지 않습니다.", tool="search_records"
                ) from None
            seen.add(key)
        return query, evidence

    @staticmethod
    def _evidence_label(
        evidence: dict[str, Any], records: list[dict[str, Any]], evidence_type: object, evidence_id: object
    ) -> str:
        candidates = [evidence]
        candidates.extend(
            row
            for row in records
            if row.get("id") == evidence_id
            or row.get("recordId") == evidence_id
            or row.get("permitId") == evidence_id
        )
        for item in candidates:
            for key in ("label", "title", "name", "permitNo", "workName", "code"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:200]
        return f"{evidence_type} #{evidence_id}"

    @staticmethod
    def _answer(raw: dict[str, Any], query: RecordQueryItem) -> str:
        label = QUERY_LABELS[query.query_type]
        if query.total == 0:
            return f"조회 조건에 맞는 {label}은 없습니다."
        answer = f"조회 조건에 맞는 {label}은 총 {query.total}건입니다."
        summary = raw.get("summary")
        if isinstance(summary, dict):
            details = [
                f"{key} {value}"
                for key, value in summary.items()
                if key != "count" and isinstance(value, (str, int, float, bool))
            ]
            if details:
                answer += " " + ", ".join(details[:4]) + "."
        if query.truncated:
            answer += " 목록은 최대 100건까지만 표시했습니다."
        return answer

    @staticmethod
    def _ok_step(
        kind: str,
        name: str,
        input_data: dict[str, Any],
        output: dict[str, Any],
        started: float,
    ) -> RecordQueryStep:
        return RecordQueryStep(
            seq=1,
            kind=kind,
            name=name,
            tool_name="search_records" if kind == "TOOL" else None,
            status="OK",
            input=input_data,
            output=output,
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    @staticmethod
    def _error_step(
        kind: str,
        name: str,
        exc: LLMError | ToolError,
        started: float,
        input_data: dict[str, Any] | None = None,
    ) -> RecordQueryStep:
        return RecordQueryStep(
            seq=1,
            kind=kind,
            name=name,
            tool_name="search_records" if kind == "TOOL" else None,
            status="ERROR",
            input=input_data or {},
            output={},
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            error_code=exc.code if isinstance(exc, ToolError) else to_error_info(exc).code,
        )

    @staticmethod
    def _renumber(steps: list[RecordQueryStep]) -> None:
        for seq, step in enumerate(steps, start=1):
            step.seq = seq
