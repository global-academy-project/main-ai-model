"""검증된 Spring 기록만 사용해 답하는 기록질문 에이전트."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
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


@dataclass(frozen=True)
class PlanOutcome:
    decision: Literal["QUERY", "CLARIFY", "OUT_OF_SCOPE"]
    query_type: str | None = None
    params: dict[str, Any] | None = None
    reason: str | None = None

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

_OUT_OF_SCOPE_TERMS = (
    "날씨", "점심", "메뉴", "요리", "번역", "농담", "주식", "환율", "스포츠", "영화", "음악",
)

_SYSTEM_PROMPT = """너는 저장된 산업안전 공식 기록을 조회하기 위한 계획자다.
질문을 임의로 답하지 말고 반드시 JSON 객체 하나만 출력한다.

decision은 다음 중 하나다.
- QUERY: 지원하는 기록 조회로 변환할 수 있고 기간도 결정할 수 있음
- CLARIFY: 산업안전 기록 질문이지만 기록 종류나 기간 등 필수 정보가 부족함
- OUT_OF_SCOPE: 산업안전 공식 기록 조회와 관계없는 질문

출력 형식:
- QUERY: {"decision":"QUERY","queryType":"허용 유형","params":{"from":"YYYY-MM-DD","to":"YYYY-MM-DD"}}
- CLARIFY: {"decision":"CLARIFY","reason":"부족한 정보"}
- OUT_OF_SCOPE: {"decision":"OUT_OF_SCOPE","reason":"범위 밖인 이유"}

SQL, siteId, agentRunId, 총 건수, 기록 ID를 만들지 않는다.
from과 to는 양 끝을 포함하며 QUERY일 때 항상 필요하고 최대 366일이다.
상대 날짜는 제공된 오늘 날짜를 기준으로 계산한다. 질문이 이전 조회를 이어 묻는 경우 이전 context를 참고한다.

허용 유형과 의미·추가 조건:
- missing_measurements: 가스측정값이 누락된 작업허가, permitType
- permit_approval_statistics: 작업허가 승인 통계, permitType
- pending_approvals: 승인 대기 기록, targetType, stepNo
- rejected_revisions: 반려 후 수정 기록, targetType
- open_corrective_actions: 미완료 시정조치, status
- failed_inspections: 부적합 점검 기록, zoneId, templateCode
- missing_tbm: TBM 미실시·누락 기록, workType, zoneId
- risk_assessment_rate: 위험성평가 실시 현황, workType
- approver_statistics: 승인자 처리 통계, approverId
- record_change_history: 기록 변경 이력, recordType, recordId, userId
- blocked_records: 차단 판정 기록, permitType

목록에 없는 조건은 넣지 않는다. 질문 표현과 가장 가까운 유형이 명확할 때만 QUERY를 선택한다.
예시:
- "지난달 가스측정값이 빠진 화기허가" → missing_measurements, permitType HOT
- "이번 주 승인 대기 문서" → pending_approvals
- "TBM 안 한 작업"처럼 기간이 없고 이전 context에도 기간이 없으면 CLARIFY
- "오늘 날씨 알려줘" → OUT_OF_SCOPE
"""


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

        if plan.decision != "QUERY":
            if plan.decision == "OUT_OF_SCOPE":
                message = (
                    "산업안전 공식 기록에 관한 질문만 조회할 수 있습니다. "
                    "작업허가, 위험성평가, TBM, 점검, 승인 또는 시정조치 기록을 질문해 주세요."
                )
            elif plan.reason:
                message = f"조회 조건을 더 알려 주세요. {plan.reason}"
            else:
                message = "조회할 기록 종류와 기간을 알려 주세요. 예: 지난달 가스측정값 누락 기록"
            return RecordQueryResult(
                run_status="WAITING_INPUT",
                ask=RecordQueryAsk(message=message),
                context={"previousQuestion": request.message, **(request.context or {})},
            )

        plan_data = {"queryType": plan.query_type, "params": plan.params}

        steps.append(
            self._ok_step(
                "PHASE",
                "query.plan",
                {"message": request.message},
                plan_data,
                plan_started,
            )
        )
        tool_input = {
            "run_id": request.run_id,
            "site_id": request.site_id,
            "query_type": plan.query_type,
            "params": plan.params,
        }
        tool_started = time.perf_counter()
        try:
            raw = self._search_records(tool_input)
            validated = self._validate_result(raw, request=request, plan=plan_data)
        except ToolError as exc:
            steps.append(self._error_step("TOOL", "search_records", exc, tool_started, tool_input))
            self._renumber(steps)
            raise AgentRunError(exc, steps=steps) from exc

        query, evidence = self._response_items(validated)
        steps.append(
            self._ok_step(
                "TOOL",
                "search_records",
                {"queryType": plan.query_type, "params": plan.params},
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
                "lastQueryType": plan.query_type,
                "lastParams": plan.params,
                "lastQuestion": request.message,
            },
            steps=steps,
        )

    @staticmethod
    def _is_obviously_out_of_scope(message: str) -> bool:
        normalized = message.strip().lower()
        return any(term in normalized for term in _OUT_OF_SCOPE_TERMS)

    def _plan(self, request: RecordQueryRequest) -> PlanOutcome:
        if self._is_obviously_out_of_scope(request.message):
            return PlanOutcome(decision="OUT_OF_SCOPE", reason="산업안전 공식 기록 조회와 관련이 없습니다.")

        today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        previous = json.dumps(request.context, ensure_ascii=False) if request.context else "없음"
        response = self._llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        f"오늘: {today}\n현장 모드: {request.mode}\n"
                        f"<previous_context>{previous}</previous_context>\n"
                        f"<question>{request.message}</question>"
                    ),
                )
            ],
            system=_SYSTEM_PROMPT,
            max_tokens=180,
        )
        if response.stop_reason == "max_tokens":
            return PlanOutcome(decision="CLARIFY", reason="질문을 짧고 구체적으로 다시 입력해 주세요.")
        raw = extract_json_object(response.text or "")
        if not raw:
            return PlanOutcome(decision="CLARIFY", reason="기록 종류와 조회 기간을 확인할 수 없습니다.")

        # 이전 계획 응답과의 호환성을 유지한다.
        if raw.get("needsClarification") is True:
            return PlanOutcome(decision="CLARIFY", reason="기록 종류와 조회 기간을 알려 주세요.")
        decision = raw.get("decision")
        if decision == "OUT_OF_SCOPE":
            reason = raw.get("reason") if isinstance(raw.get("reason"), str) else None
            return PlanOutcome(decision="OUT_OF_SCOPE", reason=reason)
        if decision == "CLARIFY":
            reason = raw.get("reason") if isinstance(raw.get("reason"), str) else None
            return PlanOutcome(decision="CLARIFY", reason=reason)
        if decision not in (None, "QUERY"):
            return PlanOutcome(decision="CLARIFY", reason="지원하는 기록 조회인지 확인할 수 없습니다.")

        query_type = raw.get("queryType")
        params = raw.get("params")
        if query_type not in QUERY_PARAMS or not isinstance(params, dict):
            return PlanOutcome(decision="CLARIFY", reason="조회할 기록 종류와 기간을 알려 주세요.")
        if not self._valid_params(query_type, params):
            return PlanOutcome(decision="CLARIFY", reason="조회 기간을 시작일과 종료일로 알려 주세요.")
        return PlanOutcome(decision="QUERY", query_type=query_type, params=params)

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
