"""작업 등록 에이전트 (시나리오 A·F 골격): 발화 → 되묻기 → 도구 호출 → 아침 등록 항목 초안.

    ① 되묻기     slots 병합 → classify_work → 필수 항목 체크리스트. 빠지면 WAITING_INPUT으로 돌려준다
    ② 근거 수집  EvidenceCollector (계획 → search_accidents/search_guides → 충분성 → 재질의)
    ③ 물질 조회  lookup_msds
    ④ 판단      LLM JSON (실패 시 규칙 템플릿)
    ⑤ 규칙 판정  check_rules (허가 종류마다)
    ⑥ 자기검증  근거 인용 · 필수 위험요인 · 빠진 근거
    ⑦ 출력      fill_form으로 위험성평가 · 작업허가서 · 점검표 · TBM 검증 → DONE

이 클래스는 엔진이 아니다. LLMClient와 ToolInvoker(도구 레지스트리)를 생성자로 받아서 쓴다.
도구는 이름으로만 부르므로 A·C의 mock을 실제 모듈로 바꿔도 이 파일은 그대로다.
모든 도구 호출과 단계는 steps(agent_steps 기록용)에 남는다.

실행 값 (Spring이 정해 요청에 실어 보낸다. LLM이 정하지 않는다)
  - run_id · site_id: 모든 steps와 check_rules 인자에 그대로 넣는다.
  - permit_id: 허가 종류가 하나일 때만 check_rules에 넣는다 (넣으면 Spring이 그 허가서의 판정으로 저장한다).
  - 개발 확인용 경로(/api/v1/test)는 id가 없다. 이때는 키를 넣지 않는다.

실패 정책
  - LLMError · ToolError로 실행이 멈추면 AgentRunError(원인, 그때까지의 steps)로 감싸 올린다.
    Spring이 실패한 실행의 steps도 agent_steps로 저장할 수 있게 (계획서 7-2 "실패" 행).
    그 밖의 예외(프로그래밍 오류)는 감싸지 않는다.
  - classify_work · fill_form의 ToolError: B 자신의 도구라 실행을 멈춘다.
  - 검색이 모두 실패: EvidenceCollector가 ToolError(EVIDENCE_SEARCH_FAILED)를 올린다.
  - lookup_msds · check_rules의 ToolError: 실행은 계속하고 verification에 남긴다.
    규칙 판정을 못 하면 승인 요청을 막는다 (모름을 통과로 바꾸지 않는다).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from app.llm.llm_base import LLMClient
from app.schemas.agent_work import (
    AgentStepDto,
    ApprovalGate,
    Classification,
    Drafts,
    EvidenceItemDto,
    EvidenceSummary,
    LlmSources,
    MsdsDto,
    RequirementDto,
    RuleCheckDto,
    RuleEvaluationDto,
    SearchFailureDto,
    VerificationIssue,
    WorkRegisterRequest,
    WorkRegisterResult,
    WorkSlots,
)

from . import drafting
from .done_message import build_done_message
from .evidence_collector import MAX_SUMMARY_CHARS, EvidenceCollector, EvidencePack, StepRecord, WorkContext
from .plan_brief import MSDS_FIRST, PLAN_MODE, StagedCollector
from .judgment import JudgeInput, judge, verify
from .judgment_pick import JUDGMENT_MODE, PICK_MAX_TOKENS, judge_pick
from .judgment_code import JUDGMENT_CODE, judge_code
from .slots import ask_message, full_text, merge_slots, missing_fields, with_asked
from .text_rules import sentences
from ..tools.tool_contract import ToolInvoker
from ..tools.tool_registry import RecordingToolInvoker, ToolCallRecord, to_plain
from app.exceptions import AgentRunError, LLMError, ToolError

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
SNIPPET_CHARS = 300


class _StepLog:
    def __init__(self, *, run_id: int | None = None, site_id: int | None = None) -> None:
        self.steps: list[AgentStepDto] = []
        self._run_id = run_id
        self._site_id = site_id

    def _add(self, **kwargs: Any) -> None:
        self.steps.append(
            AgentStepDto(seq=len(self.steps) + 1, run_id=self._run_id, site_id=self._site_id, **kwargs)
        )

    def tool(self, record: ToolCallRecord) -> None:
        self._add(kind="TOOL", name=record.tool_name, tool_name=record.tool_name, status="OK" if record.ok else "ERROR",
                  input=record.input, output=record.output, elapsed_ms=record.elapsed_ms, error_code=record.error_code)

    def evidence(self, record: StepRecord) -> None:
        self._add(kind="PHASE", name=f"evidence.{record.name}", tool_name=None, status="OK",
                  input=record.input, output=record.output, elapsed_ms=record.elapsed_ms)

    def phase(self, name: str, input: dict[str, Any], output: dict[str, Any], started: float) -> None:
        self._add(kind="PHASE", name=name, tool_name=None, status="OK", input=input, output=output,
                  elapsed_ms=int((time.perf_counter() - started) * 1000))


class WorkRegisterAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolInvoker,
        k: int = 5,
        max_requery_rounds: int = 3,
        plan_max_tokens: int = 800,
        judge_max_tokens: int = 1200,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._k = k
        self._max_requery_rounds = max_requery_rounds
        self._plan_max_tokens = plan_max_tokens
        self._judge_max_tokens = judge_max_tokens

    def run(self, request: WorkRegisterRequest) -> WorkRegisterResult:
        log = _StepLog(run_id=request.run_id, site_id=request.site_id)
        tools = RecordingToolInvoker(self._tools, log.tool)
        try:
            return self._run(request, log, tools)
        except (LLMError, ToolError) as exc:
            raise AgentRunError(exc, steps=log.steps) from exc

    def _run(self, request: WorkRegisterRequest, log: _StepLog, tools: ToolInvoker) -> WorkRegisterResult:
        # ① 되묻기
        slots = merge_slots(request.slots, request.message)
        text = full_text(slots)[-MAX_CONTEXT_CHARS:]
        classification = Classification.model_validate(tools.call("classify_work", {"text": text}))
        if missing := missing_fields(slots, classification.work_types):
            return WorkRegisterResult(
                run_status="WAITING_INPUT",
                slots=with_asked(slots, missing),
                ask={"message": ask_message(missing), "missing_fields": missing},
                classification=classification,
                steps=log.steps,
            )

        # ② 근거 수집
        msds_issues: list[VerificationIssue] = []  # MSDS_FIRST일 때 근거 검색 앞의 물질 조회에서 생긴 확인 항목
        collector = StagedCollector(
            llm=self._llm, tools=tools, k=self._k, max_requery_rounds=self._max_requery_rounds,
            max_tokens=self._plan_max_tokens, on_step=log.evidence,
            brief=PLAN_MODE == "brief",  # 계획 LLM은 작업·공구·물질만, 검색 질의는 코드 (plan_brief.py). "full"이면 이전 방식
            lookup_msds=(lambda materials: self._lookup_msds(tools, materials, msds_issues)) if MSDS_FIRST else None,
        )
        pack = collector.collect(
            WorkContext(
                text=text,
                location=slots.location,
                hot_work_kind=slots.hot_work_kind,
                nearby_materials=slots.nearby_materials,
                headcount=slots.headcount,
                end_time=slots.end_time,
                equipment=slots.equipment,
                work_types=classification.work_types,
            )
        )
        issues: list[VerificationIssue] = []
        extra_types = [w for w in pack.work_types if w not in classification.work_types and w != "일반"]
        if extra_types:
            issues.append(VerificationIssue(
                code="PLAN_EXTRA_WORK_TYPE",
                message=f"검색 계획 단계가 {', '.join(extra_types)} 작업 요소를 추가로 봤습니다. 허가 종류는 classify_work 결과만 따르니 확인하세요.",
            ))

        issues += msds_issues
        # ③ 물질 조회 (MSDS_FIRST면 계획 뒤 근거 검색 앞에서 이미 했다)
        msds = collector.msds if collector.msds is not None else self._lookup_msds(tools, pack.materials, issues)

        # ④ 판단
        started = time.perf_counter()
        summary = pack.summary if pack.plan_source == "llm" else _rule_summary(slots)
        inp = JudgeInput(
            text=text, summary=summary, work_types=classification.work_types, location=slots.location,
            hot_work_kind=slots.hot_work_kind, materials=pack.materials, equipment=slots.equipment, pack=pack, msds=msds,
        )
        if JUDGMENT_CODE:  # 위험요인마다 다시 검색해 코드가 근거를 고른다. LLM 없음 (judgment_code.py). False면 아래 방식
            hazards, judgment_source, dropped = judge_code(tools, inp)
            judged_by = {"mode": "code"}
        elif JUDGMENT_MODE == "pick":  # 위험요인은 코드, LLM은 근거 번호만 (judgment_pick.py). "full"이면 이전 방식
            hazards, judgment_source, dropped = judge_pick(self._llm, inp, max_tokens=PICK_MAX_TOKENS)
            judged_by = {"mode": "pick"}
        else:
            hazards, judgment_source, dropped = judge(self._llm, inp, max_tokens=self._judge_max_tokens)
            judged_by = {}
        log.phase("judgment", {"evidence_items": len(pack.items), "msds": len(msds)},
                  {"source": judgment_source, "hazards": [h.hazard for h in hazards], **judged_by}, started)

        # ⑤ 규칙 판정
        rule_check, raw_by_permit = self._check_rules(
            tools, classification, slots, issues, _run_arguments(request, classification.required_permits)
        )
        issues += _conflict_issues(rule_check)

        # ⑥ 자기검증
        started = time.perf_counter()
        hazards, verify_issues = verify(hazards, inp, dropped_refs=dropped)
        issues += verify_issues
        log.phase("verification", {"hazards": len(hazards)}, {"issues": [i.model_dump() for i in verify_issues]}, started)

        # ⑦ 출력 (fill_form으로 검증)
        work_types = classification.work_types
        risk = self._fill(tools, "RISK_ASSESSMENT", drafting.risk_assessment_data(summary, slots, work_types, hazards))
        permits = [
            self._fill(tools, form_type, drafting.permit_data(
                form_type, slots, hazards, rule_check if form_type in raw_by_permit else None, raw_by_permit.get(form_type, []), msds, issues=issues,
            ))
            for form_type in classification.required_permits
        ]
        block_reasons = [r for p in permits for r in p["block_reasons"]]
        block_titles = _block_titles(rule_check, block_reasons)
        checklist = self._fill(tools, "INSPECTION", drafting.checklist_data(slots, work_types))
        tbm = self._fill(tools, "TBM", drafting.tbm_data(hazards, block_titles, msds=msds))
        drafts = Drafts.model_validate(
            {"risk_assessment": risk, "permits": permits, "inspection_checklist": checklist, "tbm": tbm}
        )

        if classification.out_of_scope:
            issues.append(VerificationIssue(
                code="PERMIT_OUT_OF_SCOPE",
                message=f"{', '.join(classification.out_of_scope)} 작업은 고위험이지만 허가서 양식이 아직 없습니다 (1개월 범위 밖). 현장 절차로 확인하세요.",
            ))
        return WorkRegisterResult(
            run_status="DONE",
            slots=slots,
            ask={"message": build_done_message(drafts, request.mode), "missing_fields": []},
            classification=classification,
            evidence=_evidence_summary(pack),
            msds=msds,
            rule_check=rule_check,
            drafts=drafts,
            citations=drafting.collect_citations(hazards),
            approval_gate=ApprovalGate(can_request=not block_reasons, reasons=block_reasons),
            verification=issues,
            guidance=drafting.guidance_notes(classification, pack.missing) if request.mode == "GUIDE" else [],
            llm_sources=LlmSources(plan=pack.plan_source, judgment=judgment_source),
            steps=log.steps,
        )

    # ── 도구 호출 묶음 ──

    @staticmethod
    def _lookup_msds(tools: ToolInvoker, materials: Sequence[str], issues: list[VerificationIssue]) -> list[MsdsDto]:
        if not materials:
            return []
        try:
            raw = to_plain(tools.call("lookup_msds", {"materials": list(materials)}))
        except ToolError as exc:
            logger.warning("MSDS 조회 실패: code=%s", exc.code, exc_info=exc)
            issues.append(VerificationIssue(code="MSDS_UNAVAILABLE", message=f"MSDS 조회에 실패했습니다 ({exc.code}): {exc.message}"))
            return []
        results = [MsdsDto.model_validate(r) for r in raw.get("results", [])]
        if unknown := [r.query for r in results if not r.found]:
            issues.append(VerificationIssue(
                code="MSDS_NOT_FOUND",
                message=f"MSDS를 찾지 못한 물질이 있습니다: {', '.join(unknown)}. 허가서의 구분 칸이 비어 있으니 담당자가 확인하세요.",
            ))
        return results

    @staticmethod
    def _check_rules(
        tools: ToolInvoker,
        classification: Classification,
        slots: WorkSlots,
        issues: list[VerificationIssue],
        run_arguments: dict[str, int] | None = None,
    ) -> tuple[RuleCheckDto | None, dict[str, list[dict[str, Any]]]]:
        """run_arguments: 코드가 채우는 실행 값 (`_run_arguments`). LLM이 만든 값이 아니다."""
        if not classification.required_permits:
            return None, {}
        evaluations: list[RuleEvaluationDto] = []
        raw_by_permit: dict[str, list[dict[str, Any]]] = {}
        conflicts: list[str] = []
        notes: list[str] = []
        failed = False
        for form_type in classification.required_permits:
            arguments = {
                "permit_types": [form_type],
                "measurements": dict(slots.measurements),
                "assignments": {"FIRE_WATCHER": slots.fire_watcher, "ATTENDANT": slots.attendant},
                "location": slots.location,
                "period": {"start": slots.start_time, "end": slots.end_time},
                **(run_arguments or {}),
            }
            try:
                raw = to_plain(tools.call("check_rules", arguments))
            except ToolError as exc:
                logger.warning("규칙 판정 실패: permit=%s code=%s", form_type, exc.code, exc_info=exc)
                issues.append(VerificationIssue(code="RULE_CHECK_UNAVAILABLE",
                                                message=f"{form_type} 규칙 판정에 실패했습니다 ({exc.code}). 판정 전에는 승인 요청을 할 수 없습니다."))
                failed = True
                continue
            raw_by_permit[form_type] = list(raw.get("evaluations", []))
            evaluations += [RuleEvaluationDto.model_validate(e) for e in raw_by_permit[form_type]]
            conflicts += [c for c in raw.get("conflicts", []) if c not in conflicts]
            notes += [n for n in raw.get("notes", []) if n not in notes]
        blocked = failed or any(e.severity == "BLOCK" and e.status != "PASS" for e in evaluations)
        return RuleCheckDto(blocked=blocked, evaluations=evaluations, conflicts=conflicts, notes=notes), raw_by_permit

    @staticmethod
    def _fill(tools: ToolInvoker, form_type: str, data: dict[str, Any]) -> dict[str, Any]:
        return to_plain(tools.call("fill_form", {"form_type": form_type, "data": data}))["data"]


def _run_arguments(request: WorkRegisterRequest, permit_types: Sequence[str]) -> dict[str, int]:
    """check_rules에 덧붙일 실행 값. 요청에 없는 값은 키를 넣지 않는다.

    site_id가 없으면 실제 check_rules가 TOOL_INPUT_INVALID를 내고, 에이전트는 판정 실패로 기록해
    승인 요청을 막는다 (안전하게 실패). permit_id는 허가 종류가 하나일 때만 넣는다: 여럿이면 어느
    허가서의 판정인지 정할 수 없어 저장 없이 미리 확인만 한다.
    """
    arguments: dict[str, int] = {}
    if request.site_id is not None:
        arguments["site_id"] = request.site_id
    if request.run_id is not None:
        arguments["run_id"] = request.run_id
    if request.permit_id is not None and len(set(permit_types)) == 1:
        arguments["permit_id"] = request.permit_id
    return arguments


def _rule_summary(slots: WorkSlots) -> str:
    """검색 계획을 LLM으로 못 만들었을 때의 작업 요약: 장소가 나온 문장, 없으면 마지막 발화의 첫 문장."""
    all_sentences = [s for u in slots.utterances for s in sentences(u)]
    if slots.location:
        for sentence in all_sentences:
            if slots.location.split()[0] in sentence:
                return sentence[:MAX_SUMMARY_CHARS]
    last = sentences(slots.utterances[-1]) if slots.utterances else []
    return (last[0] if last else "작업")[:MAX_SUMMARY_CHARS]


def _conflict_issues(rule_check: RuleCheckDto | None) -> list[VerificationIssue]:
    """check_rules가 알려 준 동시작업 충돌을 확인 항목(verification)으로 올린다.

    승인 요청 차단 여부는 여기서 정하지 않는다. 차단은 Spring 규칙 엔진의 몫이다(계획서 7-0장).
    충돌 문구는 규칙 엔진이 준 그대로 쓴다.
    """
    if rule_check is None:
        return []
    return [
        VerificationIssue(code="CONCURRENT_WORK_CONFLICT", message=f"같은 구역·시간대의 다른 작업과 겹칩니다: {c} 승인 전에 확인하세요.")
        for c in rule_check.conflicts
    ]


def _block_titles(rule_check: RuleCheckDto | None, block_reasons: Sequence[str]) -> list[str]:
    if not block_reasons:
        return []
    titles = [e.title for e in rule_check.evaluations if e.severity == "BLOCK" and e.status != "PASS"] if rule_check else []
    return titles or ["규칙 판정"]


def _evidence_summary(pack: EvidencePack) -> EvidenceSummary:
    return EvidenceSummary(
        plan_source=pack.plan_source,
        rounds=pack.rounds,
        items=[
            EvidenceItemDto(
                id=i.id, doc_id=i.doc_id, corpus=i.corpus, source=i.source,
                snippet=i.text[:SNIPPET_CHARS], score=i.score, round=i.round, matched_by=i.matched_by,
            )
            for i in pack.items
        ],
        requirements=[RequirementDto(**r.model_dump()) for r in pack.requirements],
        missing=pack.missing,
        search_failures=[SearchFailureDto(**f.model_dump()) for f in pack.search_failures],
    )
