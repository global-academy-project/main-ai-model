"""계획 단계(plan_brief) 모델 비교 측정 — 순수 부분.

모델을 읽지 않는다 (torch 없이 시험 가능). 모델 호출은 `GenFn`(system, user → GenResult)으로 받는다.
실제 모델 호출은 model_runner.py, 실행은 run_quality.py·run_speed.py.

서버와 같은 것을 쓴다 (따로 베끼지 않는다 — 프롬프트나 검증 규칙을 고치면 이 측정도 같이 바뀐다):
  - 시스템 프롬프트  plan_brief.build_brief_prompt()
  - 사용자 메시지    evidence_collector.planner.build_planner_input(WorkContext)
  - 후보 검증        plan_brief.clean_equipment / clean_materials / clean_summary
  - 슬롯 추출        text_rules.find_terms (slots.py가 쓰는 것과 같은 호출)

재시도는 흉내 내지 않는다: 첫 시도의 JSON 파싱 성공률만 잰다 (실패하면 서버는 1회 재시도라 시간이 약 2배).
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datetime import date

from app.agent import plan_brief as pb
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, build_adapter_input
from app.agent.evidence_collector.models import WorkContext
from app.agent.evidence_collector.planner import build_planner_input
from app.agent.evidence_collector.textutil import extract_json_object, merge, one_line, str_list
from app.agent.slots import merge_slots
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms

KINDS = ("plain", "negation", "conditional", "out_of_dict", "multi", "none", "distractor")
EXPECTED_KEYS = frozenset({"summary", "equipment", "materials"})
DEFAULT_EVAL_SET = Path(__file__).with_name("eval_set.jsonl")


# ── 평가 문장 ──


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    text: str
    location: str | None
    hot_work_kind: str | None
    headcount: int | None
    end_time: str | None
    gold_equipment: tuple[str, ...]
    gold_materials: tuple[str, ...]
    excluded: tuple[str, ...]  # 원문에 있지만 쓰지 않는다고 한 것 (정답 목록에 없어야 한다)


def load_cases(path: Path | str = DEFAULT_EVAL_SET) -> list[Case]:
    """eval_set.jsonl을 읽고 자기모순(정답이 원문에 없거나 부정된 정답 등)을 검사한다. 문제가 있으면 ValueError."""
    cases: list[Case] = []
    problems: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        gold = row.get("gold") or {}
        case = Case(
            id=str(row["id"]), kind=str(row["kind"]), text=str(row["text"]),
            location=row.get("location"), hot_work_kind=row.get("hot_work_kind"),
            headcount=row.get("headcount"), end_time=row.get("end_time"),
            gold_equipment=tuple(gold.get("equipment", [])), gold_materials=tuple(gold.get("materials", [])),
            excluded=tuple(gold.get("excluded", [])),
        )
        if case.id in seen:
            problems.append(f"{n}행 id 중복: {case.id}")
        seen.add(case.id)
        if case.kind not in KINDS:
            problems.append(f"{case.id}: 모르는 kind {case.kind}")
        if not case.text.strip():
            problems.append(f"{case.id}: 문장이 비었습니다")
        for item in (*case.gold_equipment, *case.gold_materials):
            if not pb.mentioned(case.text, item):
                problems.append(f"{case.id}: 정답 '{item}'이 원문에 없거나 부정돼 있습니다")
        for item in case.excluded:
            if item not in case.text or pb.mentioned(case.text, item):
                problems.append(f"{case.id}: 제외 '{item}'이 원문에서 부정돼 있지 않습니다")
        cases.append(case)
    if problems:
        raise ValueError("평가 문장 검사 실패:\n" + "\n".join(problems))
    return cases


SLOT_DATE = date(2026, 1, 15)  # 여러 줄 결합에 쓰는 고정 날짜 (장비·물질에는 영향 없음, 시각 해석만 쓴다)


def slot_terms(text: str) -> tuple[list[str], list[str]]:
    """슬롯 추출(slots.py)과 같은 호출: (장비, 물질). 사전에 있는 말만 표준 이름으로, 부정된 것은 뺀다.

    여러 줄(발화가 줄바꿈으로 이어진 문장)은 서버처럼 줄마다 merge_slots 로 이어 붙인다. 뒤 줄의 "말고·대신"이 앞 줄에서 잡힌 것을
    빼므로, 문장 전체에 한 번에 find_terms 를 부르는 것과 다를 수 있다. 한 줄 문장은 지금까지와 같다.
    """
    if "\n" in text:
        slots = None
        for line in text.split("\n"):
            if line.strip():
                slots = merge_slots(slots, line, today=SLOT_DATE)
        return (list(slots.equipment), list(slots.nearby_materials)) if slots else ([], [])
    equipment, _ = find_terms(text, EQUIPMENT)
    materials, _ = find_terms(text, MATERIALS)
    return equipment, materials


def work_context(case: Case) -> WorkContext:
    equipment, materials = slot_terms(case.text)
    return WorkContext(
        text=case.text, location=case.location, hot_work_kind=case.hot_work_kind,
        nearby_materials=materials, headcount=case.headcount, end_time=case.end_time, equipment=equipment,
    )


PROMPTS = ("long", "short")


def build_messages(case: Case, prompt: str = "long") -> tuple[str, str]:
    """(system, user). long = 서버가 어댑터 없이 계획 LLM에 보내는 것, short = 어댑터를 쓸 때(plan_adapter.py)와 같은 것."""
    if prompt == "short":
        return ADAPTER_SYSTEM_PROMPT, build_adapter_input(work_context(case))
    if prompt != "long":
        raise ValueError(f"prompt는 {', '.join(PROMPTS)} 중 하나여야 합니다: {prompt}")
    return pb.build_brief_prompt(), build_planner_input(work_context(case))


# ── 모델 호출 결과 ──


@dataclass(frozen=True)
class GenResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    seconds: float
    hit_max: bool  # max_new_tokens에서 끊겼나


GenFn = Callable[[str, str], GenResult]


# ── 한 건 채점 ──

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _norm(value: str) -> str:
    return pb._compact(value).lower()  # noqa: SLF001 - 서버의 비교 규칙(공백 무시·소문자)을 그대로 쓴다


def _canon(items: Iterable[str], keys: dict[str, str]) -> list[str]:
    return pb._canonical(list(items), keys)  # noqa: SLF001 - 사전 표기를 표준 이름으로 (서버와 같은 함수)


def summary_reason(value: object, text: str) -> str | None:
    """clean_summary가 None을 낸 이유: empty / short / number. 쓸 수 있으면 None. (clean_summary와 같은 순서·조건)"""
    if not isinstance(value, str) or len(value.split()) == 0:
        return "empty"
    summary = one_line(value, pb.BRIEF_SUMMARY_CHARS)
    if len(summary) < 4:
        return "short"
    compact_text = pb._compact(text)  # noqa: SLF001
    if any(n not in compact_text for n in pb._NUMBER.findall(summary)):  # noqa: SLF001
        return "number"
    return None


def summary_support(summary: str, text: str) -> tuple[float, list[str]]:
    """요약의 내용어가 원문에 근거가 있는지 보는 거친 지표 (임시 휴리스틱, 사람 확인을 대신하지 않는다).

    두 글자 이상 토큰마다 '끝 한 글자를 뗀 앞부분'이 원문에 있으면 근거 있음으로 본다 (교체함 → 교체).
    돌려주는 값: (근거 있는 토큰 비율, 근거 없는 토큰 목록). 토큰이 없으면 (1.0, []).
    """
    haystack = pb._compact(text).lower()  # noqa: SLF001
    tokens = [t for t in _TOKEN_RE.findall(summary) if len(t) >= 2]
    if not tokens:
        return 1.0, []
    missing = [t for t in tokens if t.lower()[: max(2, len(t) - 1)] not in haystack]
    return 1 - len(missing) / len(tokens), missing


def match_counts(pred: Sequence[str], gold: Sequence[str]) -> dict[str, int]:
    """예측과 정답 비교용 개수. strict는 정확히 같은 이름, lenient는 한쪽이 다른 쪽에 포함돼도 맞는 것으로 본다."""
    p = [_norm(x) for x in pred]
    g = [_norm(x) for x in gold]
    strict = len(set(p) & set(g))

    def close(a: str, b: str) -> bool:
        return a == b or (len(a) >= 2 and len(b) >= 2 and (a in b or b in a))

    return {
        "n_pred": len(p), "n_gold": len(g), "strict_tp": strict,
        "lenient_pred_tp": sum(any(close(x, y) for y in g) for x in p),
        "lenient_gold_tp": sum(any(close(x, y) for x in p) for y in g),
    }


def score_case(case: Case, raw_text: str, gen: GenResult) -> dict[str, Any]:
    """모델 출력 한 건을 서버와 같은 검증으로 채점한다. JSONL 한 줄이 되는 dict."""
    raw = extract_json_object(raw_text)
    slot_eq, slot_mat = slot_terms(case.text)
    gold_eq = _canon(case.gold_equipment, pb._EQUIPMENT_KEYS)  # noqa: SLF001
    gold_mat = _canon(case.gold_materials, pb._MATERIAL_KEYS)  # noqa: SLF001

    rec: dict[str, Any] = {
        "id": case.id, "kind": case.kind, "text": case.text, "raw_output": raw_text,
        "prompt_tokens": gen.prompt_tokens, "gen_tokens": gen.gen_tokens, "seconds": round(gen.seconds, 3),
        "hit_max": gen.hit_max, "parsed": raw is not None,
        "gold_equipment": gold_eq, "gold_materials": gold_mat,
        "rule_equipment": slot_eq, "rule_materials": slot_mat,
        # 슬롯 사전(모델 없음)이 정답에 없는 이름을 잡은 것: 예) '메탄올'에서 '메탄'이 같이 잡힘 (부분 문자열 일치)
        "rule_extra": [x for x in (*slot_eq, *slot_mat) if _norm(x) not in {_norm(g) for g in (*gold_eq, *gold_mat)}],
    }
    if raw is None:
        rec.update(
            schema_ok=False, extra_keys=[], raw_eq=0, raw_mat=0, eq_dropped=0, mat_dropped=0,
            llm_equipment=[], llm_materials=[], final_equipment=merge(slot_eq), final_materials=merge(slot_mat),
            summary_raw=None, summary_ok=False, summary_reason="no_json", summary_truncated=False,
            summary_coverage=None, summary_unsupported=[],
        )
        return rec

    equipment, eq_dropped = pb.clean_equipment(raw.get("equipment"), case.text)
    materials, mat_dropped = pb.clean_materials(raw.get("materials"), case.text)
    summary_raw = raw.get("summary")
    reason = summary_reason(summary_raw, case.text)
    summary_ok = reason is None
    # clean_summary와 이유 판정이 어긋나지 않게 한 번 더 맞춰 본다 (서버 쪽 규칙이 바뀌면 여기서 드러난다)
    if summary_ok != (pb.clean_summary(summary_raw, case.text) is not None):
        raise AssertionError("summary_reason이 plan_brief.clean_summary와 어긋났습니다. harness.summary_reason을 고치세요.")
    summary_text = " ".join(summary_raw.split()) if isinstance(summary_raw, str) else ""
    coverage, unsupported = summary_support(summary_text, case.text) if summary_ok else (None, [])
    schema_ok = (
        set(raw) == EXPECTED_KEYS and isinstance(raw.get("summary"), str)
        and isinstance(raw.get("equipment"), list) and isinstance(raw.get("materials"), list)
    )
    rec.update(
        schema_ok=schema_ok, extra_keys=sorted(set(raw) - EXPECTED_KEYS),
        raw_eq=len(str_list(raw.get("equipment"))[: pb.MAX_BRIEF_ITEMS * 2]),
        raw_mat=len(str_list(raw.get("materials"))[: pb.MAX_BRIEF_ITEMS * 2]),
        eq_dropped=eq_dropped, mat_dropped=mat_dropped,
        llm_equipment=equipment, llm_materials=materials,
        final_equipment=merge(slot_eq, equipment), final_materials=merge(slot_mat, materials),
        summary_raw=summary_raw if isinstance(summary_raw, str) else None, summary_ok=summary_ok,
        summary_reason=reason, summary_truncated=len(summary_text) > pb.BRIEF_SUMMARY_CHARS,
        summary_coverage=coverage, summary_unsupported=unsupported,
    )
    return rec


# ── 집계 ──


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def prf(tp_pred: int, n_pred: int, tp_gold: int, n_gold: int) -> dict[str, float | None]:
    precision, recall = _div(tp_pred, n_pred), _div(tp_gold, n_gold)
    f1 = _div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {"precision": precision, "recall": recall, "f1": f1}


def _item_scores(records: Sequence[dict[str, Any]], view: str, domains: Sequence[str]) -> dict[str, dict]:
    """view: llm(모델이 낸 것 중 검증 통과분) / final(슬롯 + llm, 서버가 실제로 쓰는 값) / rule(슬롯만, 모델 없음)."""
    total = {k: 0 for k in ("n_pred", "n_gold", "strict_tp", "lenient_pred_tp", "lenient_gold_tp")}
    for r in records:
        for domain in domains:  # "equipment" | "materials"
            counts = match_counts(r[f"{view}_{domain}"], r[f"gold_{domain}"])
            for k in total:
                total[k] += counts[k]
    return {
        "strict": prf(total["strict_tp"], total["n_pred"], total["strict_tp"], total["n_gold"]),
        "lenient": prf(total["lenient_pred_tp"], total["n_pred"], total["lenient_gold_tp"], total["n_gold"]),
    }


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    records: Sequence[dict[str, Any]], *, model: str = "", dtype: str = "", revision: str = "", prompt: str = "long", adapter: str = "",
) -> dict[str, Any]:
    n = len(records)
    parsed = [r for r in records if r["parsed"]]
    summaries_ok = [r for r in parsed if r["summary_ok"]]
    out: dict[str, Any] = {
        "model": model, "dtype": dtype, "revision": revision, "prompt": prompt, "adapter": adapter, "n": n,
        "format": {
            "json_parse_rate": _div(len(parsed), n),
            "schema_rate": _div(sum(r["schema_ok"] for r in records), n),
            "max_tokens_cut_rate": _div(sum(r["hit_max"] for r in records), n),
            "extra_key_cases": sum(bool(r["extra_keys"]) for r in records),
        },
        "tokens": {
            "prompt_mean": _mean([r["prompt_tokens"] for r in records]),
            "gen_mean": _mean([r["gen_tokens"] for r in records]),
            "gen_p95": _p95([r["gen_tokens"] for r in records]),
            "gen_max": max((r["gen_tokens"] for r in records), default=None),
            "seconds_mean": _mean([r["seconds"] for r in records]),
        },
        "candidates": {
            "equipment_raw": sum(r["raw_eq"] for r in records), "equipment_dropped": sum(r["eq_dropped"] for r in records),
            "materials_raw": sum(r["raw_mat"] for r in records), "materials_dropped": sum(r["mat_dropped"] for r in records),
        },
        "summary": {
            "usable_rate": _div(len(summaries_ok), n),
            "reasons": {k: sum(r["summary_reason"] == k for r in records) for k in ("no_json", "empty", "short", "number")},
            "truncated_rate": _div(sum(r["summary_truncated"] for r in parsed), len(parsed)),
            "mean_support": _mean([r["summary_coverage"] for r in summaries_ok]),
            "unsupported_case_rate": _div(sum(bool(r["summary_unsupported"]) for r in summaries_ok), len(summaries_ok)),
        },
        "items": {
            domain: {view: _item_scores(records, view, [domain]) for view in ("llm", "final", "rule")}
            for domain in ("equipment", "materials")
        },
        "items_all": {view: _item_scores(records, view, ["equipment", "materials"]) for view in ("llm", "final", "rule")},
        "by_kind": {},
        "rule_extra_cases": [{"id": r["id"], "items": r["rule_extra"]} for r in records if r["rule_extra"]],
    }
    cand = out["candidates"]
    cand["equipment_drop_rate"] = _div(cand["equipment_dropped"], cand["equipment_raw"])
    cand["materials_drop_rate"] = _div(cand["materials_dropped"], cand["materials_raw"])
    for kind in KINDS:
        rows = [r for r in records if r["kind"] == kind]
        if not rows:
            continue
        out["by_kind"][kind] = {
            "n": len(rows),
            "parse_rate": _div(sum(r["parsed"] for r in rows), len(rows)),
            "items_all": {view: _item_scores(rows, view, ["equipment", "materials"]) for view in ("llm", "final", "rule")},
            # 뽑을 것이 없는 문장에서 모델이 후보를 냈나 (검증 전 / 검증 뒤 남은 것)
            "raw_candidate_cases": sum((r["raw_eq"] + r["raw_mat"]) > 0 for r in rows),
            "kept_candidate_cases": sum(bool(r["llm_equipment"] or r["llm_materials"]) for r in rows),
        }
    return out


# ── 실행 ──


def evaluate(
    cases: Sequence[Case], gen_fn: GenFn, *, model: str = "", dtype: str = "", revision: str = "",
    progress: Callable[[int, int, Case], None] | None = None, prompt: str = "long", adapter: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        system, user = build_messages(case, prompt)
        gen = gen_fn(system, user)
        records.append(score_case(case, gen.text, gen))
        if progress:
            progress(i, len(cases), case)
    return records, summarize(records, model=model, dtype=dtype, revision=revision, prompt=prompt, adapter=adapter)


# ── 저장 ──

REVIEW_COLUMNS = ("id", "kind", "text", "summary", "unsupported_tokens", "summary_usable", "최종_장비", "최종_물질", "요약_사실일치(O/X)", "메모")


def write_outputs(out_dir: Path | str, tag: str, records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"records": out / f"{tag}.records.jsonl", "summary": out / f"{tag}.summary.json", "review": out / f"{tag}.review.csv"}
    with paths["records"].open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    # 요약 사실 일치는 사람이 본다: 엑셀에서 열리게 BOM을 붙인다
    with paths["review"].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REVIEW_COLUMNS)
        for r in records:
            writer.writerow([
                r["id"], r["kind"], r["text"], r["summary_raw"] or "", " ".join(r["summary_unsupported"]),
                "O" if r["summary_ok"] else f"X({r['summary_reason']})",
                ", ".join(r["final_equipment"]), ", ".join(r["final_materials"]), "", "",
            ])
    return paths
