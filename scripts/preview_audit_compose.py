"""구성기 결과를 마크다운으로 찍어 눈으로 확인하는 B용 스크립트(제품 화면 아님).

    python -m scripts.preview_audit_compose tests/fixtures/audit_compose/normal.json
    python -m scripts.preview_audit_compose tests/fixtures/audit_compose/normal.json --json
    python -m scripts.preview_audit_compose tests/fixtures/audit_compose/normal.json --out preview.md

PowerShell 5.1에서 `>`로 돌리면 파일이 UTF-16이 되어 한글이 깨질 수 있으니, 파일로 저장할 때는 `--out`을 쓴다(UTF-8).

입력은 `/audit-report`와 같은 모양의 JSON 파일이다. 서버·DB·LLM 없이 돈다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agent.audit_compose import compose_audit_report
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult

_ROMAN = {1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ", 4: "Ⅳ", 5: "Ⅴ"}


def _num(value: object, unit: str = "") -> str:
    return "확인 불가" if value is None else f"{value}{unit}"


def render_markdown(result: AuditComposeResult) -> str:
    m = result.meta
    out: list[str] = []
    add = out.append
    add("# 안전관리 자체 점검(감사 대응) 결과보고서")
    add(f"현장 {m.site_id}번 · 점검 기간 {m.date_from} ~ {m.date_to} · 작성 {m.generated_at}")
    add("")
    add("> 이 출력은 B의 확인용 미리보기이며 제품 화면·인쇄 서식이 아닙니다.")
    if result.status == "INCOMPLETE":
        add("> 확인 불가 항목이 있어 보고서가 완전하지 않습니다.")
    add("")

    add(f"## {_ROMAN[1]}. 점검 개요")
    s = result.overview.scope
    add(
        f"- 점검 범위: 작업 {_num(s.works, '건')}, 작업허가 {_num(s.permits, '건')}, "
        f"아차사고 {_num(s.near_misses, '건')}, 개선조치 {_num(s.corrective_actions, '건')}"
    )
    for g in result.overview.groups:
        add(f"- 묶음 {g.no}. {g.area}")
        for p in g.items:
            kind = "법정 의무" if p.kind == "LEGAL" else "사내 절차"
            add(f"  - {p.area}: {p.point} ({kind})")
    add("")

    add(f"## {_ROMAN[2]}. 점검 결과 요약")
    add("| 묶음 | 점검 항목 | 기준 | 확인 건수 | 해당 건수 | 비율 | 상태 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for g in result.summary.groups:
        for r in g.rows:
            rate = "-" if r.rate_pct is None else f"{r.rate_pct:.1f}%"
            checked = "-" if r.checked is None else str(r.checked)
            count = "-" if r.count is None else str(r.count)
            add(f"| {g.area} | {r.area} | {r.label} | {checked} | {count} | {rate} | {r.state_label} |")
    add("")

    add(f"## {_ROMAN[3]}. 항목별 점검 결과")
    for g in result.groups:
        add(f"### {g.area} ({g.doc_no})")
        for item in g.items:
            add(f"○ {item.area} [{item.state_label}]")
            for sentence in item.sentences:
                add(f"  - {sentence.text}")
            if item.records.lines:
                add(f"  - 근거 기록 ({item.records.shown}건 표시)")
                for line in item.records.lines:
                    add(f"    - {line}")
            if item.action_status:
                a = item.action_status
                add(f"  - 조치 현황: 미완료 {a.open}건, 완료 {a.completed}건, 검증 완료 {a.verified}건, 기한 경과 {a.overdue}건")
        add(f"작성자 의견: {g.responder_opinion or '(작성 전)'}")
        add(f"조치할 사항: {g.requested_action or '(작성 전)'}")
        add("")

    o = result.overall
    add(f"## {_ROMAN[4]}. 종합 내역")
    add("### 1. 점검 결과 종합")
    for line in o.lines:
        add(f"- {line}")
    add("### 2. 기록 보완 여부 확인이 필요한 사항")
    if o.review_needed:
        for r in o.review_needed:
            ids = ""
            if r.record_labels:
                shown = r.record_labels[:10]
                more = len(r.record_labels) - len(shown)
                ids = f" (허가 번호 {', '.join(shown)}{f' 외 {more}건' if more else ''})"
            add(f"- {r.text}{ids}")
    else:
        add("- 해당 없음")
    add("### 3. 항목 간 연결")
    for link in o.links or []:
        add(f"- {link.text}")
    if not o.links:
        add("- 조회 응답의 연결 정보로 만든 항목이 없음")
    add("### 4. 현황 참고 사항")
    for ref in o.reference:
        add(f"- {ref.text}")
    if not o.reference:
        add("- 해당 없음")
    add("### 5. 사람이 작성·확인할 사항")
    for task in o.human_tasks:
        add(f"- {task}")
    add("")

    add(f"## {_ROMAN[5]}. 작성·확인")
    add("작성자: (작성 전)  확인자: (작성 전)  서명: (작성 전)")
    add("")

    if result.verification:
        add("---")
        add("검증 메모 (개발용)")
        for v in result.verification:
            add(f"- [{v.level}] {v.code}: {v.message}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="감사 대응 보고서 구성기 미리보기")
    parser.add_argument("input", help="구성기 입력 JSON 파일(/audit-report 본문과 같은 모양)")
    parser.add_argument("--json", action="store_true", help="마크다운 대신 응답 JSON을 찍는다")
    parser.add_argument("--out", help="화면 대신 이 파일에 UTF-8로 저장한다")
    args = parser.parse_args(argv)

    body = AuditComposeInput.model_validate(json.loads(Path(args.input).read_text(encoding="utf-8")))
    result = compose_audit_report(body)
    if args.json:
        text = json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2) + "\n"
    else:
        text = render_markdown(result)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"저장함: {args.out}")
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # 콘솔 코드 페이지와 무관하게 UTF-8로 내보낸다
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
