"""감사 대응 보고서 구성기 (B, LLM 없음).

Spring이 집계한 5개 항목(`/audit-report`와 같은 본문)을 받아 코드만으로
① 묶음 배치 ② 상태값 ③ 서술문(기준→현황→근거) ④ 종합 내역 ⑤ 문장 숫자 대조를 하고 JSON을 돌려준다.

원칙: 숫자는 입력 집계에서만 나오고 비율·합계는 코드가 계산한다. 값이 이상하거나 없으면 '확인 불가'로
두며 0건으로 바꾸지 않는다. 기존 `/audit-report`(D)를 부르거나 임포트하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.agent import audit_compose_criteria as criteria
from app.agent.audit_compose_check import (
    StatsInvalid,
    check_corrective,
    check_count,
    check_permit,
    check_risk,
    fmt,
    percent,
    numbers_in,
    rate_differs,
    unexpected_numbers,
)
from app.agent.audit_compose_text import (
    GROUPS,
    NOT_SUPPORTED_SENTENCE,
    SPECS,
    ItemSpec,
    action_source_ko,
    action_status_ko,
    basis_sentence,
    clip,
    date_part,
    join_parts,
    permit_type_ko,
    truncated_sentence,
    work_type_ko,
)
from app.schemas.audit_compose import (
    ActionStatus,
    AuditComposeInput,
    AuditComposeItem,
    AuditComposeResult,
    ComposeGroup,
    ComposeItem,
    ComposeMeta,
    ComposeOverall,
    ComposeOverview,
    ComposeScope,
    ComposeSentence,
    ComposeSummary,
    ComposeVerification,
    EvidenceRef,
    FactValue,
    ItemFacts,
    ItemRecords,
    OverallLine,
    OverviewGroup,
    OverviewPoint,
    ReviewNeeded,
    SummaryGroup,
    SummaryRow,
)

_KST = timezone(timedelta(hours=9))
_LABEL = criteria.STATE_LABELS


@dataclass
class _Built:
    """항목 하나를 만든 결과와, 종합 내역이 가져다 쓸 값."""

    item: ComposeItem
    row: SummaryRow
    review: ReviewNeeded | None = None
    reference: str | None = None
    permit_ids: list[int] | None = None  # 연결 id(허가). 목록이 잘렸으면 None
    scope: tuple[str, int] | None = None  # (ComposeScope 칸 이름, 값)


@dataclass
class _Run:
    """한 번의 구성에서 쓰는 상태."""

    verification: list[ComposeVerification] = field(default_factory=list)

    def note(self, code: str, level: str, message: str) -> None:
        self.verification.append(ComposeVerification(code=code, level=level, message=message))


# ---------------------------------------------------------------- 진입점


def _build_all(request: AuditComposeInput, run: _Run) -> dict[str, _Built]:
    """묶음 순서대로 세부 점검 7개를 만든다. 입력에 없는 항목은 '확인 불가'가 된다."""
    by_type = {item.query_type: item for item in request.items}
    for query_type in by_type:
        if query_type not in SPECS:
            run.note("ITEM_IGNORED", "INFO", "구성기가 다루지 않는 조회 항목은 건너뜀")

    builders: dict[str, Callable[[ItemSpec, AuditComposeItem | None, int, _Run], _Built]] = {
        "risk_assessment_rate": _build_risk,
        "permit_approval_statistics": _build_permit,
        "missing_measurements": _build_missing,
        "blocked_records": _build_blocked,
        "corrective_action_rate": _build_corrective,
    }

    built: dict[str, _Built] = {}
    seq = 0
    for _no, _area, query_types in GROUPS:
        for query_type in query_types:
            seq += 1
            spec = SPECS[query_type]
            builder = builders.get(query_type)
            if builder is None:
                built[query_type] = _build_not_supported(spec, seq, run)
            else:
                built[query_type] = builder(spec, by_type.get(query_type), seq, run)
    return built


def compose_audit_report(
    request: AuditComposeInput,
    *,
    now: datetime | None = None,
    narrative_generator: Callable[[ComposeItem], list[str]] | None = None,
) -> AuditComposeResult:
    run = _Run()
    built = _build_all(request, run)

    # P2는 선택 기능이다. 후보 생성 실패 시 P1 템플릿을 유지한다.
    if narrative_generator is not None:
        for entry in built.values():
            item = entry.item
            if item.state == "UNAVAILABLE":
                continue
            try:
                candidates = narrative_generator(item)
                if not candidates or any(not isinstance(text, str) or not text.strip() for text in candidates):
                    raise ValueError("LLM 문장 후보가 비어 있거나 올바르지 않습니다.")
                item_allowed = _allowed_item_numbers(item)
                if any(unexpected_numbers(text, item_allowed) for text in candidates):
                    run.note("NARRATIVE_REPLACED", "WARN", f"{item.area}: LLM 문장 수치가 해당 항목 facts와 달라 템플릿을 유지함")
                    continue
                item.sentences = [ComposeSentence(text=text.strip(), source="LLM") for text in candidates]
            except Exception as exc:
                run.note("NARRATIVE_FALLBACK", "WARN", f"{item.area}: LLM 문장 생성 실패, 템플릿 문장을 사용함 ({type(exc).__name__})")

    allowed = _allowed_numbers(built)
    _check_item_sentences(built, allowed, run)

    month = request.date_from[:4] + request.date_from[5:7]
    groups: list[ComposeGroup] = []
    overview_groups: list[OverviewGroup] = []
    summary_groups: list[SummaryGroup] = []
    for no, area, query_types in GROUPS:
        groups.append(
            ComposeGroup(
                no=f"AR-{request.site_id}-{month}-{no:02d}",
                seq=no,
                area=area,
                items=[built[q].item for q in query_types],
            )
        )
        overview_groups.append(
            OverviewGroup(
                no=no,
                area=area,
                items=[OverviewPoint(area=SPECS[q].area, point=SPECS[q].point, kind=SPECS[q].kind) for q in query_types],  # type: ignore[arg-type]
            )
        )
        summary_groups.append(SummaryGroup(no=no, area=area, rows=[built[q].row for q in query_types]))

    scope = ComposeScope()
    for b in built.values():
        if b.scope is not None:
            setattr(scope, b.scope[0], b.scope[1])

    overall = _build_overall(request, built, scope, allowed, run)

    unsupported_only = {b.item.query_type for b in built.values() if b.item.unavailable_reason == "NOT_SUPPORTED"}
    if unsupported_only:
        run.note("GROUP_NOT_SUPPORTED", "INFO", "아차사고 내역은 조회 항목이 아직 없어 확인 불가로 둠")
    incomplete = any(
        b.item.state == "UNAVAILABLE" and b.item.unavailable_reason != "NOT_SUPPORTED" for b in built.values()
    )

    stamp = (now or datetime.now(_KST)).isoformat(timespec="seconds")
    return AuditComposeResult(
        status="INCOMPLETE" if incomplete else "COMPLETE",
        meta=ComposeMeta(
            report_id=request.report_id,
            site_id=request.site_id,
            date_from=request.date_from,
            date_to=request.date_to,
            generated_at=stamp,
        ),
        overview=ComposeOverview(scope=scope, groups=overview_groups),
        summary=ComposeSummary(groups=summary_groups),
        groups=groups,
        overall=overall,
        verification=run.verification,
    )


# ---------------------------------------------------------------- 공통 부품


def _evidence_refs(item: AuditComposeItem) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    seen: set[tuple[str, int]] = set()
    for entry in item.evidence:
        kind, ident = entry.get("type"), entry.get("id")
        if not isinstance(kind, str) or isinstance(ident, bool) or not isinstance(ident, int):
            continue
        if (kind, ident) in seen:
            continue
        seen.add((kind, ident))
        refs.append(EvidenceRef(type=kind, id=ident))
    return refs


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _fact(label: str, value: float | int, unit: str = "건") -> FactValue:
    return FactValue(label=label, value=value, unit=unit)


def _unavailable(
    spec: ItemSpec, seq: int, reason: str, sentence: str, run: _Run | None = None, code: str | None = None,
    message: str | None = None,
) -> _Built:
    if run is not None and code is not None and message is not None:
        run.note(code, "WARN", f"{spec.area}: {message}")
    item = ComposeItem(
        seq=seq,
        query_type=spec.query_type,
        area=spec.area,
        kind=spec.kind,  # type: ignore[arg-type]
        state="UNAVAILABLE",
        state_label=_LABEL["UNAVAILABLE"],
        unavailable_reason=reason,  # type: ignore[arg-type]
        basis=list(spec.basis),
        sentences=[ComposeSentence(text=sentence)],
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, state="UNAVAILABLE", state_label=_LABEL["UNAVAILABLE"])
    return _Built(item=item, row=row)


def _build_not_supported(spec: ItemSpec, seq: int, run: _Run) -> _Built:
    return _unavailable(spec, seq, "NOT_SUPPORTED", f"{spec.area}: {NOT_SUPPORTED_SENTENCE}")


def _missing_input(spec: ItemSpec, seq: int, run: _Run, item: AuditComposeItem | None) -> _Built | None:
    """항목이 없거나 stats가 비면 '확인 불가'. 0건으로 바꾸지 않는다."""
    if item is None:
        return _unavailable(spec, seq, "NO_DATA", f"{spec.area}: 조회 결과를 받지 못해 확인하지 못함.", run, "ITEM_MISSING", "조회 결과가 없음")
    if not item.stats:
        return _unavailable(spec, seq, "NO_DATA", f"{spec.area}: 집계 값이 없어 확인하지 못함.", run, "ITEM_MISSING", "집계 값이 없음")
    return None


def _invalid(spec: ItemSpec, seq: int, run: _Run, exc: StatsInvalid) -> _Built:
    return _unavailable(
        spec, seq, "INVALID", f"{spec.area}: 집계 값을 확인할 수 없어 보고하지 못함.", run, exc.code, exc.message
    )


def _state_fields(state: str) -> dict[str, str]:
    return {"state": state, "state_label": _LABEL[state]}


def _sentences(spec: ItemSpec, texts: list[tuple[str, list[EvidenceRef]]]) -> list[ComposeSentence]:
    out = [ComposeSentence(text=basis_sentence(spec))]
    out.extend(ComposeSentence(text=text, evidence=refs) for text, refs in texts)
    return out


def _line_items(records: list[dict[str, Any]], make: Callable[[dict[str, Any]], str | None]) -> list[str]:
    lines: list[str] = []
    for row in records:
        text = make(row) if isinstance(row, dict) else None
        if text:
            lines.append(text)
    return lines


# ---------------------------------------------------------------- 항목별 구성


def _build_risk(spec: ItemSpec, item: AuditComposeItem | None, seq: int, run: _Run) -> _Built:
    early = _missing_input(spec, seq, run, item)
    if early:
        return early
    assert item is not None
    try:
        v = check_risk(item.stats)
    except StatsInvalid as exc:
        return _invalid(spec, seq, run, exc)
    works, written, approved, missing = v["works"], v["written"], v["approved"], v["missing"]
    rate_w = percent(written, works)
    rate_a = percent(approved, works)
    rate_m = percent(missing, works)
    if rate_differs(item.stats.get("rate", rate_w), rate_w):
        run.note("RATE_DIFFERS", "WARN", f"{spec.area}: 보낸 비율과 건수로 계산한 비율이 달라 계산한 값을 씀")
    if rate_differs(item.stats.get("approvedRate", rate_a), rate_a):
        run.note("RATE_DIFFERS", "WARN", f"{spec.area}: 보낸 승인 비율과 계산한 비율이 달라 계산한 값을 씀")
    state = criteria.risk_state(works, missing)
    texts: list[tuple[str, list[EvidenceRef]]]
    if works == 0:
        texts = [("해당 기간 확인 대상 작업은 0건임.", [])]
        values = [_fact("작업", 0)]
    else:
        assert rate_w is not None and rate_a is not None
        texts = [
            (
                f"해당 기간 작업 {works}건 중 위험성평가를 작성한 작업은 {written}건({fmt(rate_w)}%), "
                f"승인된 작업은 {approved}건({fmt(rate_a)}%)임.",
                [],
            ),
            (f"위험성평가 기록이 없는 작업은 {missing}건임.", []),
        ]
        values = [
            _fact("작업", works), _fact("평가 작성", written), _fact("작성 비율", rate_w, "%"),
            _fact("승인", approved), _fact("승인 비율", rate_a, "%"), _fact("평가 기록 없음", missing),
        ]
    lines = _line_items(
        item.records,
        lambda r: _risk_line(r),
    )
    built_item = ComposeItem(
        seq=seq, query_type=spec.query_type, area=spec.area, kind=spec.kind,  # type: ignore[arg-type]
        basis=list(spec.basis),
        facts=ItemFacts(denominator=works, numerator=missing, rate_pct=rate_m, total=works, shown=len(lines), values=values),
        sentences=_sentences(spec, texts),
        records=ItemRecords(lines=lines, shown=len(lines), total=len(lines)),
        **_state_fields(state),
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, checked=works, count=missing, rate_pct=rate_m, **_state_fields(state))
    review = ReviewNeeded(area=spec.area, count=missing, text=f"위험성평가 기록이 없는 작업 {missing}건") if state == "REVIEW" else None
    return _Built(item=built_item, row=row, review=review, scope=("works", works))


def _risk_line(r: dict[str, Any]) -> str | None:
    works, written = _int(r.get("works")), _int(r.get("written"))
    if works is None or written is None or written > works:
        return None
    rate = percent(written, works)
    tail = f"({fmt(rate)}%)" if rate is not None else ""
    return f"{work_type_ko(r.get('workType'))}: 작업 {works}건 중 평가 작성 {written}건{tail}"


def _build_permit(spec: ItemSpec, item: AuditComposeItem | None, seq: int, run: _Run) -> _Built:
    early = _missing_input(spec, seq, run, item)
    if early:
        return early
    assert item is not None
    try:
        v = check_permit(item.stats)
    except StatsInvalid as exc:
        return _invalid(spec, seq, run, exc)
    permits, requested = v["permits"], v["requested"]
    approved, rejected, canceled, blocked = v["approved"], v["rejected"], v["canceled"], v["blocked"]
    decided = approved + rejected
    rate = percent(approved, decided)
    if rate_differs(item.stats.get("approvalRate", rate), rate):
        run.note("RATE_DIFFERS", "WARN", f"{spec.area}: 보낸 승인률과 건수로 계산한 승인률이 달라 계산한 값을 씀")
    state = criteria.permit_state(permits)
    if permits == 0:
        texts: list[tuple[str, list[EvidenceRef]]] = [("해당 기간 작업허가는 0건임.", [])]
        values = [_fact("작업허가", 0)]
    else:
        rate_text = (
            f"승인률(승인÷(승인+반려))은 {fmt(rate)}%임." if rate is not None
            else "승인·반려 처리가 없어 승인률은 산정하지 않음."
        )
        texts = [
            (
                f"해당 기간 작업허가 {permits}건에 대한 승인 요청은 {requested}건이며, "
                f"승인 {approved}건, 반려 {rejected}건, 취소 {canceled}건임.",
                [],
            ),
            (rate_text, []),
            (f"규칙 판정으로 차단된 허가는 {blocked}건임.", []),
        ]
        values = [
            _fact("작업허가", permits), _fact("승인 요청", requested), _fact("승인", approved),
            _fact("반려", rejected), _fact("취소", canceled), _fact("차단", blocked),
        ]
        if rate is not None:
            values.append(_fact("승인률", rate, "%"))
    lines = _line_items(item.records, _permit_line)
    built_item = ComposeItem(
        seq=seq, query_type=spec.query_type, area=spec.area, kind=spec.kind,  # type: ignore[arg-type]
        basis=list(spec.basis),
        facts=ItemFacts(denominator=decided, numerator=approved, rate_pct=rate, total=permits, shown=len(lines), values=values),
        sentences=_sentences(spec, texts),
        records=ItemRecords(lines=lines, shown=len(lines), total=len(lines)),
        **_state_fields(state),
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, checked=decided, count=approved, rate_pct=rate, **_state_fields(state))
    reference = None
    if state == "REFERENCE":
        reference = f"작업허가 {permits}건 중 승인 {approved}건, 반려 {rejected}건, 차단 {blocked}건임."
    return _Built(item=built_item, row=row, reference=reference, scope=("permits", permits))


def _permit_line(r: dict[str, Any]) -> str | None:
    permits, approved = _int(r.get("permits")), _int(r.get("approved"))
    rejected, blocked = _int(r.get("rejected")), _int(r.get("blocked"))
    if None in (permits, approved, rejected, blocked):
        return None
    return f"{permit_type_ko(r.get('permitType'))}: 허가 {permits}건, 승인 {approved}건, 반려 {rejected}건, 차단 {blocked}건"


def _list_count(spec: ItemSpec, item: AuditComposeItem, run: _Run) -> tuple[int, int, bool] | None:
    """목록형 항목의 (전체 건수, 표시 건수, 잘림). 전체 건수는 stats.count, 잘림은 건수와 표시 개수로 계산한다."""
    count = check_count(item.stats)["count"]
    shown = len(item.records)
    if shown > count:
        raise StatsInvalid("STATS_INCONSISTENT", "표시한 기록이 전체 건수보다 많음")
    truncated = count > shown
    if item.total is not None and item.total != count:
        run.note("TOTAL_MISMATCH", "WARN", f"{spec.area}: 보낸 전체 건수와 집계 건수가 달라 집계 건수를 씀")
    if item.truncated is not None and item.truncated != truncated:
        run.note("TRUNCATED_MISMATCH", "WARN", f"{spec.area}: 보낸 잘림 표시와 건수로 계산한 값이 달라 계산한 값을 씀")
    if truncated:
        run.note("TRUNCATED", "INFO", f"{spec.area}: 전체 {count}건 중 {shown}건만 표시함")
    return count, shown, truncated


def _build_missing(spec: ItemSpec, item: AuditComposeItem | None, seq: int, run: _Run) -> _Built:
    early = _missing_input(spec, seq, run, item)
    if early:
        return early
    assert item is not None
    try:
        count, shown, truncated = _list_count(spec, item, run)
    except StatsInvalid as exc:
        return _invalid(spec, seq, run, exc)
    run.note("NO_DENOMINATOR", "INFO", f"{spec.area}: 모수 없이 건수만 있어 비율은 내지 않음")
    state = criteria.missing_measurement_state(count)
    refs = _evidence_refs(item)
    texts: list[tuple[str, list[EvidenceRef]]] = [(f"해당 기간 승인된 허가 중 작업 전 측정값이 없는 허가는 {count}건임.", refs)]
    if count > 0:
        texts.append(("승인 전 측정값 확인이 필요한 항목이므로 해당 기록의 보완 여부 확인이 필요함.", []))
    if truncated:
        texts.append((truncated_sentence(count, shown), []))
    lines = _line_items(item.records, _missing_line)
    permit_ids = _ids(item.records, "permitId")
    built_item = ComposeItem(
        seq=seq, query_type=spec.query_type, area=spec.area, kind=spec.kind,  # type: ignore[arg-type]
        basis=list(spec.basis),
        facts=ItemFacts(numerator=count, total=count, shown=shown, truncated=truncated, values=[_fact("측정값 없이 승인된 허가", count)]),
        sentences=_sentences(spec, texts),
        records=ItemRecords(lines=lines, shown=shown, total=count),
        **_state_fields(state),
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, count=count, **_state_fields(state))
    review = (
        ReviewNeeded(
            area=spec.area, count=count, record_ids=permit_ids, record_labels=_labels(item.records, "permitNo"),
            text=f"측정값 없이 승인된 허가 {count}건",
        )
        if state == "REVIEW" else None
    )
    return _Built(item=built_item, row=row, review=review, permit_ids=None if truncated else permit_ids)


def _missing_line(r: dict[str, Any]) -> str | None:
    items = r.get("missingItems")
    n = len(items) if isinstance(items, list) else None
    no = clip(r.get("permitNo"), 30)
    head = f"허가 {no}({permit_type_ko(r.get('permitType'))})" if no else permit_type_ko(r.get("permitType"))
    work = clip(r.get("workTitle"), 40)
    place = clip(r.get("location"), 20)
    day = date_part(r.get("approvedAt"))
    return join_parts([
        head,
        f"작업 {work}" if work else None,
        f"장소 {place}" if place else None,
        f"승인일 {day}" if day else None,
        f"누락 측정 항목 {n}개" if n else None,
    ])


def _ids(records: list[dict[str, Any]], key: str) -> list[int]:
    out: list[int] = []
    for r in records:
        value = r.get(key) if isinstance(r, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value not in out:
            out.append(value)
    return out


def _labels(records: list[dict[str, Any]], key: str) -> list[str]:
    out: list[str] = []
    for r in records:
        label = clip(r.get(key), 30) if isinstance(r, dict) else None
        if label and label not in out:
            out.append(label)
    return out


def _build_blocked(spec: ItemSpec, item: AuditComposeItem | None, seq: int, run: _Run) -> _Built:
    early = _missing_input(spec, seq, run, item)
    if early:
        return early
    assert item is not None
    try:
        count, shown, truncated = _list_count(spec, item, run)
    except StatsInvalid as exc:
        return _invalid(spec, seq, run, exc)
    run.note("NO_DENOMINATOR", "INFO", f"{spec.area}: 모수 없이 건수만 있어 비율은 내지 않음")
    state = criteria.blocked_state(count)
    refs = _evidence_refs(item)
    texts: list[tuple[str, list[EvidenceRef]]] = [(f"해당 기간 규칙 판정으로 차단된 허가 기록은 {count}건임.", refs)]
    if count > 0:
        texts.append(("차단은 허가 단계에서 규칙에 따라 승인을 막은 기록이며 현황 참고용임.", []))
    if truncated:
        texts.append((truncated_sentence(count, shown), []))
    lines = _line_items(item.records, _blocked_line)
    built_item = ComposeItem(
        seq=seq, query_type=spec.query_type, area=spec.area, kind=spec.kind,  # type: ignore[arg-type]
        basis=list(spec.basis),
        facts=ItemFacts(numerator=count, total=count, shown=shown, truncated=truncated, values=[_fact("차단 기록", count)]),
        sentences=_sentences(spec, texts),
        records=ItemRecords(lines=lines, shown=shown, total=count),
        **_state_fields(state),
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, count=count, **_state_fields(state))
    reference = f"규칙 판정으로 차단된 허가 기록 {count}건임." if state == "REFERENCE" else None
    return _Built(
        item=built_item, row=row, reference=reference,
        permit_ids=None if truncated else _ids(item.records, "permitId"),
    )


def _blocked_line(r: dict[str, Any]) -> str | None:
    no = clip(r.get("permitNo"), 30)
    head = f"허가 {no}({permit_type_ko(r.get('permitType'))})" if no else permit_type_ko(r.get("permitType"))
    day = date_part(r.get("blockedAt"))
    violations = r.get("violations")
    reasons = [v for v in violations if isinstance(v, dict)] if isinstance(violations, list) else []
    first = clip(reasons[0].get("message"), 80) if reasons else None
    return join_parts([
        head,
        f"차단일 {day}" if day else None,
        f"차단 사유 {len(reasons)}건" if reasons else None,
        first,
    ])


def _build_corrective(spec: ItemSpec, item: AuditComposeItem | None, seq: int, run: _Run) -> _Built:
    early = _missing_input(spec, seq, run, item)
    if early:
        return early
    assert item is not None
    try:
        v = check_corrective(item.stats)
    except StatsInvalid as exc:
        return _invalid(spec, seq, run, exc)
    actions, completed, verified = v["actions"], v["completed"], v["verified"]
    open_count, overdue = v["open"], v["overdue"]
    rate = percent(completed, actions)
    if rate_differs(item.stats.get("completionRate", rate), rate):
        run.note("RATE_DIFFERS", "WARN", f"{spec.area}: 보낸 조치율과 건수로 계산한 조치율이 달라 계산한 값을 씀")
    state = criteria.corrective_state(actions, open_count, overdue)
    refs = _evidence_refs(item)
    if actions == 0:
        texts: list[tuple[str, list[EvidenceRef]]] = [("해당 기간 등록된 개선조치는 0건임.", [])]
        values = [_fact("개선조치", 0)]
    else:
        assert rate is not None
        texts = [
            (
                f"해당 기간 등록된 개선조치(취소 제외) {actions}건 중 완료(검증 포함) {completed}건({fmt(rate)}%), "
                f"검증 완료 {verified}건임.",
                refs,
            ),
            (f"미완료 개선조치는 {open_count}건이며 이 중 기한이 지난 조치는 {overdue}건임.", []),
        ]
        values = [
            _fact("개선조치", actions), _fact("완료(검증 포함)", completed), _fact("조치율", rate, "%"),
            _fact("검증 완료", verified), _fact("미완료", open_count), _fact("기한 경과", overdue),
        ]
    lines = _line_items(item.records, _action_line)
    built_item = ComposeItem(
        seq=seq, query_type=spec.query_type, area=spec.area, kind=spec.kind,  # type: ignore[arg-type]
        basis=list(spec.basis),
        facts=ItemFacts(denominator=actions, numerator=completed, rate_pct=rate, total=actions, shown=len(lines), values=values),
        sentences=_sentences(spec, texts),
        records=ItemRecords(lines=lines, shown=len(lines), total=actions),
        action_status=ActionStatus(open=open_count, completed=completed, verified=verified, overdue=overdue),
        **_state_fields(state),
    )
    row = SummaryRow(area=spec.area, label=spec.summary_label, checked=actions, count=completed, rate_pct=rate, **_state_fields(state))
    review = (
        ReviewNeeded(area=spec.area, count=overdue, text=f"기한이 지난 개선조치 {overdue}건")
        if state == "REVIEW" else None
    )
    reference = f"미완료 개선조치 {open_count}건임." if state == "REFERENCE" else None
    return _Built(item=built_item, row=row, review=review, reference=reference, scope=("corrective_actions", actions))


def _action_line(r: dict[str, Any]) -> str | None:
    title = clip(r.get("title"), 40)
    due = r.get("dueDate") if isinstance(r.get("dueDate"), str) else None
    return join_parts([
        title or "개선조치",
        f"기한 {due}" if due else None,
        action_status_ko(r.get("status")),
        f"출처 {action_source_ko(r.get('sourceType'))}" if r.get("sourceType") else None,
    ])


# ---------------------------------------------------------------- 숫자 대조

def _allowed_numbers(built: dict[str, _Built]) -> set[str]:
    """종합 문장과 항목 문장이 써도 되는 숫자: 항목 `facts`의 값과 상태별 항목 수."""
    allowed: set[str] = set()
    for b in built.values():
        facts = b.item.facts
        for value in (facts.denominator, facts.numerator, facts.total, facts.shown):
            if value is not None:
                allowed.add(fmt(value))
        if facts.rate_pct is not None:
            allowed.add(fmt(facts.rate_pct))
        for fv in facts.values:
            allowed.add(fmt(fv.value))
        for basis in b.item.basis:
            if basis.legal_basis:
                allowed.update(numbers_in(basis.legal_basis))
        if b.item.action_status:
            a = b.item.action_status
            allowed.update(fmt(x) for x in (a.open, a.completed, a.verified, a.overdue))
        if b.item.records.total is not None:
            allowed.add(fmt(b.item.records.total))
    # 상태별 항목 수(전체와 묶음별). 항목 상태를 코드가 센 값이다.
    for state in ("REVIEW", "REFERENCE", "OK", "UNAVAILABLE"):
        allowed.add(str(sum(1 for b in built.values() if b.item.state == state)))
        for _no, _area, query_types in GROUPS:
            allowed.add(str(sum(1 for q in query_types if built[q].item.state == state)))
    return allowed


def _allowed_item_numbers(item: ComposeItem) -> set[str]:
    """LLM 항목 문장이 해당 항목 facts·법적 근거에 포함된 숫자만 쓰도록 제한한다."""
    facts = item.facts
    allowed = {
        fmt(value)
        for value in (facts.denominator, facts.numerator, facts.total, facts.shown, facts.rate_pct)
        if value is not None
    }
    allowed.update(fmt(value.value) for value in facts.values)
    for basis in item.basis:
        if basis.legal_basis:
            allowed.update(numbers_in(basis.legal_basis))
    if item.action_status:
        allowed.update(fmt(value) for value in (
            item.action_status.open,
            item.action_status.completed,
            item.action_status.verified,
            item.action_status.overdue,
        ))
    if item.records.total is not None:
        allowed.add(fmt(item.records.total))
    return allowed


def _fallback_sentence(item: ComposeItem) -> str:
    """대조에 실패한 문장 대신 쓰는 문장: `facts` 값만 그대로 나열한다."""
    parts = [f"{fv.label} {fmt(fv.value)}{fv.unit}" for fv in item.facts.values]
    return (", ".join(parts) + "임.") if parts else "집계 값은 항목 표를 따름."


def _check_item_sentences(built: dict[str, _Built], allowed: set[str], run: _Run) -> None:
    for b in built.values():
        for sentence in b.item.sentences:
            if unexpected_numbers(sentence.text, allowed):
                run.note("NARRATIVE_REPLACED", "WARN", f"{b.item.area}: 문장 속 숫자가 집계 값과 달라 값 나열로 바꿈")
                sentence.text = _fallback_sentence(b.item)
                sentence.source = "TEMPLATE"


# ---------------------------------------------------------------- 종합 내역


def _build_overall(
    request: AuditComposeInput, built: dict[str, _Built], scope: ComposeScope, allowed: set[str], run: _Run
) -> ComposeOverall:
    lines: list[str] = []

    def count_text(label: str, value: int | None) -> str:
        return f"{label} {value}건" if value is not None else f"{label} 확인 불가"

    lines.append(
        f"점검 기간은 {request.date_from}부터 {request.date_to}까지이며, 조회한 기록은 "
        f"{count_text('작업', scope.works)}, {count_text('작업허가', scope.permits)}, "
        f"{count_text('개선조치', scope.corrective_actions)}임. 아차사고는 확인하지 못함."
    )
    for no, area, query_types in GROUPS:
        states = [built[q].item.state for q in query_types]
        review, reference = states.count("REVIEW"), states.count("REFERENCE")
        unavailable = states.count("UNAVAILABLE")
        lines.append(f"{area}: 확인 필요 {review}개 항목, 현황 참고 {reference}개 항목, 확인 불가 {unavailable}개 항목임.")

    review_needed = [b.review for b in built.values() if b.review is not None]
    reference_lines = [OverallLine(text=b.reference) for b in built.values() if b.reference]
    links = _links(built, allowed, run)

    lines = _drop_mismatched(lines, allowed, run)
    for review in review_needed:
        if _bad(review.text, allowed, run):
            review.text = review.area
    reference_lines = [o for o in reference_lines if not _bad(o.text, allowed, run)]
    links = [o for o in links if not _bad(o.text, allowed, run)]

    unavailable_areas = [b.item.area for b in built.values() if b.item.state == "UNAVAILABLE"]
    tasks = [f"{area} 묶음의 작성자 의견과 조치할 사항 작성" for _no, area, _q in GROUPS]
    if unavailable_areas:
        tasks.append("확인 불가 항목(" + ", ".join(unavailable_areas) + ")은 기록을 직접 확인해 작성")
    tasks.append("작성자·확인자 서명")

    return ComposeOverall(
        counts=scope, lines=lines, review_needed=review_needed, links=links, reference=reference_lines, human_tasks=tasks
    )


def _bad(text: str, allowed: set[str], run: _Run) -> bool:
    if unexpected_numbers(text, allowed):
        run.note("NARRATIVE_REPLACED", "WARN", "종합 문장의 숫자가 항목과 달라 뺌")
        return True
    return False


def _drop_mismatched(lines: list[str], allowed: set[str], run: _Run) -> list[str]:
    kept: list[str] = []
    for line in lines:
        if _bad(line, allowed, run):
            continue
        kept.append(line)
    return kept


def _links(built: dict[str, _Built], allowed: set[str], run: _Run) -> list[OverallLine]:
    """항목 간 연결은 조회 응답이 연결 id를 줄 때만 만든다(추정으로 잇지 않는다)."""
    missing, blocked = built["missing_measurements"], built["blocked_records"]
    if missing.item.state == "UNAVAILABLE" or blocked.item.state == "UNAVAILABLE":
        return []
    if missing.permit_ids is None or blocked.permit_ids is None:
        run.note("LINK_SKIPPED_TRUNCATED", "INFO", "기록 목록이 잘려 항목 간 연결을 만들지 않음")
        return []
    both = [i for i in missing.permit_ids if i in set(blocked.permit_ids)]
    if not both:
        return []
    n = len(both)
    allowed.add(fmt(n))  # 연결 건수는 두 목록의 허가 id 교집합이라 코드가 센 값이다
    total = missing.item.facts.numerator
    refs = [EvidenceRef(type="PERMIT", id=i) for i in both]
    return [
        OverallLine(
            text=f"측정값 없이 승인된 허가 {total}건 중 같은 기간 규칙 차단 기록이 있는 허가는 {n}건임.",
            evidence=refs,
        )
    ]
