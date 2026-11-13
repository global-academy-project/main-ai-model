"""서버 집계값을 감사보고서 초안으로 요약하고 수치를 대조한다."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from app.agent.evidence_collector.textutil import extract_json_object
from app.exceptions import AgentRunError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.llm.engines.llm_audit_report import SYSTEM_PROMPT
from app.schemas.audit_report import (
    AuditReportDraftItem,
    AuditReportInput,
    AuditReportResult,
    AuditVerification,
)

_NUMBER = re.compile(r"(?<![A-Za-z가-힣])[-+]?\d+(?:[.,]\d+)?%?")


def _numbers(value: object) -> list[Decimal]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float, Decimal)):
        try:
            return [Decimal(str(value))]
        except InvalidOperation:
            return []
    if isinstance(value, dict):
        out: list[Decimal] = []
        for item in value.values():
            out.extend(_numbers(item))
        return out
    if isinstance(value, list):
        out: list[Decimal] = []
        for item in value:
            out.extend(_numbers(item))
        return out
    return []


def _text_numbers(text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for token in _NUMBER.findall(text):
        token = token.rstrip("%").replace(",", "")
        try:
            out.append(Decimal(token))
        except InvalidOperation:
            continue
    return out


def _fmt(values: list[Decimal]) -> list[str]:
    return [format(value.normalize(), "f") for value in values]


class AuditReportAgent:
    """LLM은 문장만 만들고, 집계는 입력 stats와 코드 검증기가 담당한다."""

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, data: AuditReportInput) -> AuditReportResult:
        prompt = json.dumps(data.model_dump(by_alias=True), ensure_ascii=False)
        response = self._llm.chat([Message(role="user", content=prompt)], system=SYSTEM_PROMPT)
        raw = extract_json_object(response.text or "")
        if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
            raise AgentRunError("감사보고서 모델 응답을 해석할 수 없습니다.")

        by_type = {item.query_type: item for item in data.items}
        drafts: list[AuditReportDraftItem] = []
        for output in raw["items"]:
            if not isinstance(output, dict) or output.get("queryType") not in by_type:
                raise AgentRunError("감사보고서 항목이 입력 집계와 일치하지 않습니다.")
            source = by_type[output["queryType"]]
            summary = output.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise AgentRunError("감사보고서 항목 요약이 비어 있습니다.")
            expected = _numbers(source.stats)
            actual = _text_numbers(summary)
            mismatch = Counter(expected) != Counter(actual)
            verification = AuditVerification(
                status="MISMATCH" if mismatch else "MATCH",
                expected_numbers=_fmt(expected),
                actual_numbers=_fmt(actual),
                mismatches=["요약문 수치와 stats가 일치하지 않습니다."] if mismatch else [],
            )
            drafts.append(AuditReportDraftItem(
                query_type=source.query_type,
                title=source.title,
                summary=summary.strip(),
                verification=verification,
                evidence=source.evidence,
            ))

        passed = len(drafts) == len(data.items) and all(
            item.verification.status == "MATCH" for item in drafts
        )
        return AuditReportResult(
            report_id=data.report_id,
            site_id=data.site_id,
            date_from=data.date_from,
            date_to=data.date_to,
            status="REVIEW_REQUIRED" if passed else "FAILED",
            items=drafts,
            overall_verification="PASS" if passed else "FAIL",
            data_snapshot=data.data_snapshot,
        )
