"""Spring 감사보고서의 `audit_reports.data_snapshot`으로 구성기(LLM 없음)를 돌려 종합 내역까지 찍는 B용 스크립트.

DB를 조회하지 않는다. 사용자가 DB에서 꺼내 둔 파일만 읽는다(제품 코드 아님, 확인용).

    # 1) docker MySQL에서 한 건을 꺼낸다 (예: --raw로 이스케이프 없이)
    #    SELECT JSON_OBJECT('id', id, 'siteId', site_id, 'dateFrom', date_from, 'dateTo', date_to,
    #                       'dataSnapshot', data_snapshot) FROM audit_reports WHERE id = 7;
    # 2) 구성기를 돌린다
    python -m scripts.compose_from_snapshot row.json --md out.md
    python -m scripts.compose_from_snapshot snapshot.json --site-id 1 --from 2026-10-01 --to 2026-10-31 --md out.md

받는 파일 모양(아래 중 아무거나):
  - `data_snapshot` 그대로: {"risk_assessment_rate": {...}, "permit_approval_statistics": {...}, ...}
  - `items_json` 그대로: [{"queryType": ...}, ...]
  - `/audit-report` 본문: {"reportId": .., "siteId": .., "dateFrom": .., "dateTo": .., "items": [...]}
  - `audit_reports` 한 행: {"id", "siteId"(site_id), "dateFrom"(date_from), "dateTo"(date_to), "dataSnapshot"(data_snapshot)}
`data_snapshot`만 줄 때는 현장 번호와 기간을 `--site-id --from --to`로 준다. 명령줄 값이 파일 값보다 앞선다.

PowerShell 5.1의 `>`는 UTF-16 파일을 만들 수 있다. 이 스크립트는 UTF-16·BOM도 읽고, 결과는 `--md`·`--input-out`(UTF-8)으로 저장한다.
서버·DB·LLM 없이 돈다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.agent.audit_compose import compose_audit_report
from app.schemas.audit_compose import AuditComposeInput
from scripts.preview_audit_compose import render_markdown

# 구성기가 읽는 항목 필드. summary·verification 같은 나머지는 구성기에 필요 없어 버린다.
ITEM_FIELDS = ("queryType", "title", "stats", "total", "truncated", "records", "evidence")
_ROW_KEYS = {
    "report_id": ("reportId", "report_id", "id"),
    "site_id": ("siteId", "site_id"),
    "date_from": ("dateFrom", "date_from"),
    "date_to": ("dateTo", "date_to"),
}
_SNAPSHOT_KEYS = ("dataSnapshot", "data_snapshot")


class SnapshotError(ValueError):
    """읽을 수 없는 파일이거나 필요한 값이 빠짐."""


def read_text(path: Path) -> str:
    """UTF-8·BOM·UTF-16(PowerShell 5.1 `>`)·CP949 순으로 읽는다."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SnapshotError("파일 인코딩을 알 수 없음(UTF-8·UTF-16·CP949 아님)")


def parse_json_text(text: str) -> Any:
    """JSON으로 읽는다. mysql 배치 출력(`\\\\`·`\\n` 이스케이프)이면 한 번 풀어서 다시 시도한다."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        unescaped = text.replace("\\\\", "\\")
        try:
            return json.loads(unescaped)
        except json.JSONDecodeError:
            raise SnapshotError(
                f"JSON으로 읽을 수 없음({first.msg}, {first.lineno}행). "
                "mysql로 꺼냈다면 -N -B 대신 --raw(-r)로 이스케이프 없이 꺼내 보세요"
            ) from first


def _pick(obj: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if obj.get(name) is not None:
            return obj[name]
    return None


def _items_from(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = [x for x in data if isinstance(x, dict) and x.get("queryType")]
    elif isinstance(data, dict):
        items = [v for v in data.values() if isinstance(v, dict) and v.get("queryType")]
    else:
        items = []
    if not items:
        raise SnapshotError("항목(queryType이 있는 객체)을 찾지 못함")
    return items


def extract(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """파일 내용에서 (항목 목록, 파일에 들어 있던 보고서 값)을 뽑는다."""
    meta: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, names in _ROW_KEYS.items():
            value = _pick(data, names)
            if value is not None:
                meta[key] = value
        snap = _pick(data, _SNAPSHOT_KEYS)
        if snap is not None:
            if isinstance(snap, str):
                snap = parse_json_text(snap)
            return _items_from(snap), meta
        if isinstance(data.get("items"), list):
            return _items_from(data["items"]), meta
        # data_snapshot 자체. 키 이름이 우연히 id·siteId여도 값이 dict면 항목이므로 meta로 보지 않는다.
        meta = {}
    return _items_from(data), meta


def to_compose_input(
    data: Any,
    *,
    report_id: int | None = None,
    site_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    items, meta = extract(data)
    given = {"report_id": report_id, "site_id": site_id, "date_from": date_from, "date_to": date_to}
    merged = {k: (given[k] if given[k] is not None else meta.get(k)) for k in given}
    missing = [name for name in ("site_id", "date_from", "date_to") if merged[name] is None]
    if missing:
        flags = {"site_id": "--site-id", "date_from": "--from", "date_to": "--to"}
        raise SnapshotError("현장 번호·기간이 없음. " + ", ".join(flags[m] for m in missing) + " 를 주세요")
    try:
        report = int(merged["report_id"]) if merged["report_id"] is not None else 1
        site = int(merged["site_id"])
    except (TypeError, ValueError) as exc:
        raise SnapshotError("보고서 번호·현장 번호는 정수여야 함") from exc
    return {
        "reportId": report,
        "siteId": site,
        "dateFrom": str(merged["date_from"])[:10],
        "dateTo": str(merged["date_to"])[:10],
        "items": [{k: it[k] for k in ITEM_FIELDS if k in it} for it in items],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="data_snapshot으로 감사 대응 보고서 구성기를 돌린다(LLM 없음)")
    parser.add_argument("input", help="data_snapshot·items_json·/audit-report 본문·audit_reports 한 행 JSON 파일")
    parser.add_argument("--report-id", type=int, help="보고서 번호(없으면 파일 값, 그것도 없으면 1)")
    parser.add_argument("--site-id", type=int, help="현장 번호")
    parser.add_argument("--from", dest="date_from", help="기간 시작 YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="기간 끝 YYYY-MM-DD")
    parser.add_argument("--input-out", help="바꾼 구성기 입력 JSON을 이 파일에 UTF-8로 저장")
    parser.add_argument("--md", help="종합 내역까지 든 마크다운을 이 파일에 UTF-8로 저장(없으면 화면에 찍음)")
    parser.add_argument("--json", action="store_true", help="마크다운 대신 응답 JSON을 찍는다")
    args = parser.parse_args(argv)

    try:
        data = parse_json_text(read_text(Path(args.input)))
        body = to_compose_input(
            data,
            report_id=args.report_id,
            site_id=args.site_id,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        parsed = AuditComposeInput.model_validate(body)
    except (SnapshotError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:  # pydantic ValidationError도 ValueError
        print(f"오류: 구성기 입력이 올바르지 않음\n{exc}", file=sys.stderr)
        return 2

    if args.input_out:
        Path(args.input_out).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"구성기 입력 저장함: {args.input_out}")

    result = compose_audit_report(parsed)
    if args.json:
        text = json.dumps(result.model_dump(by_alias=True), ensure_ascii=False, indent=2) + "\n"
    else:
        text = render_markdown(result)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8", newline="\n")
        print(f"저장함: {args.md}")
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
