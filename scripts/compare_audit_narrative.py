"""감사 보고서 예시 문서를 LLM 끔/켬(가드 있음·없음)으로 나란히 만들어 비교하는 B용 스크립트 (제품 코드 아님).

한 번 실행하면 출력 폴더에 이런 파일이 생긴다 (모두 UTF-8).
  off.md          LLM 끔 (템플릿 문장만)           ← AUDIT_NARRATIVE_LLM=false 와 같은 결과
  on_guard.md     LLM 켬 + 가드(지금 패치)          ← AUDIT_NARRATIVE_LLM=true, NARRATIVE_GUARD=True
  on_noguard.md   LLM 켬 + 가드 없음(패치 전 방식)  ← NARRATIVE_GUARD=False
  compare.md      세 가지를 항목별로 비교한 표와 문장 차이, 검증 코드(NARRATIVE_*), 걸린 시간

사용 (저장소 폴더에서, PowerShell):
  # 모델 없이 시연: 가짜 생성기 (good = 문장만 다듬음, bad = 평가어·지어낸 날짜·빠진 숫자를 일부러 섞음)
  python -m scripts.compare_audit_narrative tests/fixtures/audit_compose/normal.json --llm fake-good --out-dir out_good
  python -m scripts.compare_audit_narrative tests/fixtures/audit_compose/normal.json --llm fake-bad  --out-dir out_bad
  # 실제 모델: 서버와 같은 설정(.env, SCENARIO_MODELS의 audit_query)으로 모델을 올려 문장 후보를 만든다 (CPU면 항목당 수 초~)
  python -m scripts.compare_audit_narrative tests/fixtures/audit_compose/normal.json --llm real --out-dir out_real

입력은 `/audit-report` 본문 JSON이거나 `data_snapshot`(이때는 --site-id --from --to 를 준다)이다. DB·서버는 쓰지 않는다.
`--llm real`은 가드가 켜진 쪽과 꺼진 쪽에 **같은 후보를 한 번만 만들어 재사용**한다(모델 호출 횟수·시간을 공정하게 보려면 --calls 로 호출 기록을 본다).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from app.agent import audit_narrative_guard as guard
from app.agent.audit_compose import compose_audit_report
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult, ComposeItem
from scripts.compose_from_snapshot import SnapshotError, parse_json_text, read_text, to_compose_input
from scripts.preview_audit_compose import render_markdown

Generator = Callable[[ComposeItem], list[str]]


# ── 후보 생성기 ───────────────────────────────────────────────


def _replaceable(item: ComposeItem) -> list[str]:
    """가드가 교체 대상으로 보는(= LLM이 대신 쓸 수 있는) 템플릿 문장."""
    return [s.text for s in guard.split_sentences(item.sentences)[1]]


def fake_good(item: ComposeItem) -> list[str]:
    """숫자·사실은 그대로 두고 어투만 바꾸는 이상적인 후보."""
    texts = _replaceable(item)
    return [f"점검 결과, {t}" for t in texts] if texts else []


def fake_bad(item: ComposeItem) -> list[str]:
    """항목 순서대로 문제 유형을 돌려 가며 섞는다 (가드가 무엇을 막는지 보여 주기 위한 시연용)."""
    texts = _replaceable(item)
    if not texts:
        return []
    kind = _BAD_KINDS[_bad_counter.setdefault(item.query_type, len(_bad_counter) % len(_BAD_KINDS))]
    if kind == "평가어":
        return [texts[0].rstrip(".") + "으로 기준에 부적정하며 위반 사례가 있음."] + texts[1:]
    if kind == "지어낸 날짜":
        return [texts[0].rstrip(".") + ". 2025-03-15에 점검함."] + texts[1:]
    if kind == "숫자 누락":
        return [re.sub(r"\d+(?:\.\d+)?", "여러", texts[0])]
    return [f"점검 결과, {t}" for t in texts]  # 정상 후보도 하나 섞는다


_BAD_KINDS = ("평가어", "지어낸 날짜", "숫자 누락", "정상")
_bad_counter: dict[str, int] = {}


def real_generator(calls: list[dict[str, Any]]) -> Generator:
    """서버와 같은 설정으로 audit_query 모델을 올려 D의 후보 생성기를 부른다."""
    from app.agent.audit_narrative import generate_sentence_candidates
    from app.core import get_settings
    from app.llm.llm_provider import LLMClientProvider

    provider = LLMClientProvider.from_settings(get_settings())
    llm = provider.get("audit_query")

    def generate(item: ComposeItem) -> list[str]:
        started = time.monotonic()
        try:
            return generate_sentence_candidates(item, llm)
        finally:
            calls.append({"item": item.query_type, "state": item.state, "seconds": round(time.monotonic() - started, 2)})

    return generate


def recording(generator: Generator, calls: list[dict[str, Any]], cache: dict[str, Any]) -> Generator:
    """같은 항목에는 같은 후보를 돌려주고(첫 호출 때만 진짜 생성), 호출 기록을 남긴다."""

    def generate(item: ComposeItem) -> list[str]:
        key = item.query_type
        if key not in cache:
            try:
                cache[key] = ("ok", generator(item))
            except Exception as exc:  # 가드 쪽의 오류 처리를 비교하려고 그대로 다시 던진다
                cache[key] = ("error", exc)
        kind, value = cache[key]
        if kind == "error":
            raise value
        return list(value)

    return generate


# ── 실행·비교 ─────────────────────────────────────────────────


def run(body: AuditComposeInput, generator: Generator | None, *, guard_on: bool) -> tuple[AuditComposeResult, float, list[str]]:
    seen: list[str] = []
    wrapped: Generator | None = None
    if generator is not None:

        def wrapped(item: ComposeItem) -> list[str]:  # type: ignore[misc]
            seen.append(item.query_type)
            return generator(item)

    old = guard.NARRATIVE_GUARD
    guard.NARRATIVE_GUARD = guard_on
    started = time.monotonic()
    try:
        result = compose_audit_report(body, narrative_generator=wrapped)
    finally:
        guard.NARRATIVE_GUARD = old
    return result, time.monotonic() - started, seen


def item_map(result: AuditComposeResult) -> dict[str, Any]:
    return {item.query_type: item for group in result.groups for item in group.items}


def narrative_codes(result: AuditComposeResult) -> list[tuple[str, str]]:
    return [(v.code, v.message) for v in result.verification if v.code.startswith("NARRATIVE_")]


def sources(item: Any) -> str:
    mark = {"TEMPLATE": "T", "LLM": "L", "MOCK": "M"}
    return "".join(mark.get(s.source, "?") for s in item.sentences)


def render_compare(results: dict[str, tuple[AuditComposeResult, float, list[str]]], *, mode: str, calls: list[dict[str, Any]]) -> str:
    names = {"off": "LLM 끔", "on_guard": "LLM 켬 + 가드", "on_noguard": "LLM 켬 + 가드 없음"}
    maps = {k: item_map(v[0]) for k, v in results.items()}
    base = maps["off"]
    out: list[str] = ["# 감사 보고서 문장: LLM 끔/켬 비교", ""]
    out.append(f"후보 생성기: `{mode}` · 문장 출처 표기: T=템플릿(규칙) L=LLM M=mock")
    out.append("")
    out.append("## 1. 한눈에 보기")
    out.append("")
    out.append("| 항목 | 상태 | " + " | ".join(names[k] for k in results) + " |")
    out.append("| --- | --- | " + " | ".join("---" for _ in results) + " |")
    for qt, item in base.items():
        cells = []
        for key in results:
            it = maps[key][qt]
            changed = [s.text for s in it.sentences] != [s.text for s in item.sentences]
            cells.append(f"{sources(it)}{' (바뀜)' if changed else ''}")
        out.append(f"| {item.area} | {item.state_label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## 2. 실행 결과")
    out.append("")
    out.append("| 구분 | 모델에 물은 항목 수 | 걸린 시간(초) | 채택 거절·대체(NARRATIVE_*) |")
    out.append("| --- | --- | --- | --- |")
    for key, (res, seconds, seen) in results.items():
        codes = narrative_codes(res)
        out.append(f"| {names[key]} | {len(seen)} | {seconds:.2f} | {len(codes)} |")
    out.append("")
    if calls:
        out.append("모델 호출 기록(실제 생성 때만):")
        for c in calls:
            out.append(f"- {c['item']} ({c['state']}): {c['seconds']}초")
        out.append("")
    out.append("## 3. 항목별 문장 비교")
    for qt, item in base.items():
        out.append("")
        out.append(f"### {item.area} [{item.state_label}]")
        for key, (res, _, _) in results.items():
            it = maps[key][qt]
            out.append(f"**{names[key]}** ({sources(it)})")
            for s in it.sentences:
                out.append(f"- {s.text}")
            note = [m for c, m in narrative_codes(res) if it.area in m]
            for m in note:
                out.append(f"  - ⚠ {m}")
            out.append("")
    out.append("## 4. 가드가 한 일 (LLM 켬 + 가드)")
    out.append("")
    codes = narrative_codes(results["on_guard"][0])
    if codes:
        for code, message in codes:
            out.append(f"- `{code}`: {message}")
    else:
        out.append("- 거절·대체 없음 (후보가 전부 가드를 통과했거나 호출 대상이 없었음)")
    out.append("")
    out.append("## 5. 가드 없음에서 새는 것 (평가어·입력에 없는 날짜)")
    out.append("")
    leaks = []
    for qt, it in maps["on_noguard"].items():
        for s in it.sentences:
            if s.source == "LLM":
                bad_words = guard.banned_words_in(s.text)
                bad_dates = guard.date_problems(s.text, {base_input_dates[0], base_input_dates[1]})
                if bad_words or bad_dates:
                    leaks.append(f"- {it.area}: {s.text}  (평가어 {bad_words or '없음'}, 날짜 {bad_dates or '없음'})")
    out.extend(leaks or ["- 새는 문장 없음"])
    out.append("")
    return "\n".join(out) + "\n"


base_input_dates: tuple[str, str] = ("", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="감사 보고서 문장 LLM 끔/켬 비교")
    parser.add_argument("input", help="/audit-report 본문 JSON 또는 data_snapshot JSON")
    parser.add_argument("--llm", choices=("fake-good", "fake-bad", "real"), default="fake-good", help="후보 생성기")
    parser.add_argument("--out-dir", default="compare_out", help="결과 폴더(없으면 만든다)")
    parser.add_argument("--site-id", type=int)
    parser.add_argument("--from", dest="date_from")
    parser.add_argument("--to", dest="date_to")
    parser.add_argument("--report-id", type=int)
    args = parser.parse_args(argv)

    try:
        data = parse_json_text(read_text(Path(args.input)))
        payload = to_compose_input(data, report_id=args.report_id, site_id=args.site_id, date_from=args.date_from, date_to=args.date_to)
    except SnapshotError as exc:
        print(f"입력 오류: {exc}")
        return 2
    body = AuditComposeInput.model_validate(payload)
    global base_input_dates
    base_input_dates = (body.date_from, body.date_to)

    calls: list[dict[str, Any]] = []
    if args.llm == "real":
        source: Generator = real_generator(calls)
    elif args.llm == "fake-bad":
        _bad_counter.clear()
        source = fake_bad
    else:
        source = fake_good
    shared = recording(source, calls, {})

    results = {
        "off": run(body, None, guard_on=True),
        "on_guard": run(body, shared, guard_on=True),
        "on_noguard": run(body, shared, guard_on=False),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, (res, _, _) in results.items():
        (out_dir / f"{key}.md").write_text(render_markdown(res), encoding="utf-8", newline="\n")
    (out_dir / "compare.md").write_text(render_compare(results, mode=args.llm, calls=calls), encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"저장함: {out_dir.resolve()} (off.md, on_guard.md, on_noguard.md, compare.md)")
    for key, (res, seconds, seen) in results.items():
        print(f"- {key}: 물은 항목 {len(seen)}개, {seconds:.2f}초, NARRATIVE_* {len(narrative_codes(res))}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
