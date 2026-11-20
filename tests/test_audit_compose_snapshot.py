"""data_snapshot → 구성기 입력 변환 스크립트(scripts/compose_from_snapshot.py) 시험."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import compose_from_snapshot as cfs

FIXTURES = Path(__file__).parent / "fixtures" / "audit_compose"
SNAPSHOT = FIXTURES / "snapshot_sample.json"


def load_snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def row_of(snapshot: dict, as_text: bool = False) -> dict:
    return {
        "id": 7,
        "siteId": 3,
        "dateFrom": "2026-10-01",
        "dateTo": "2026-10-31",
        "dataSnapshot": json.dumps(snapshot, ensure_ascii=False) if as_text else snapshot,
    }


def test_keyed_snapshot_becomes_compose_input_with_only_needed_fields() -> None:
    body = cfs.to_compose_input(load_snapshot(), site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    assert body["reportId"] == 1
    assert body["siteId"] == 1
    assert [i["queryType"] for i in body["items"]] == list(load_snapshot())
    for item in body["items"]:
        assert set(item) <= set(cfs.ITEM_FIELDS)
        assert "summary" not in item and "verification" not in item
    risk = body["items"][0]
    assert risk["stats"]["works"] == 12
    assert risk["total"] == 12 and risk["truncated"] is False


def test_items_json_list_and_request_body_are_accepted() -> None:
    items = list(load_snapshot().values())
    from_list = cfs.to_compose_input(items, site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    body = {"reportId": 9, "siteId": 2, "dateFrom": "2026-09-01", "dateTo": "2026-09-30", "items": items}
    from_body = cfs.to_compose_input(body)
    assert from_list["items"] == from_body["items"]
    assert (from_body["reportId"], from_body["siteId"], from_body["dateFrom"]) == (9, 2, "2026-09-01")


@pytest.mark.parametrize("as_text", [False, True])
def test_audit_reports_row_fills_report_site_and_period(as_text: bool) -> None:
    body = cfs.to_compose_input(row_of(load_snapshot(), as_text))
    assert (body["reportId"], body["siteId"], body["dateFrom"], body["dateTo"]) == (7, 3, "2026-10-01", "2026-10-31")
    assert len(body["items"]) == 5


def test_command_line_values_win_over_file_values_and_datetime_is_cut_to_date() -> None:
    row = row_of(load_snapshot())
    row["dateFrom"] = "2026-10-01 00:00:00"
    body = cfs.to_compose_input(row, report_id=70, site_id=4, date_to="2026-10-15")
    assert (body["reportId"], body["siteId"], body["dateFrom"], body["dateTo"]) == (70, 4, "2026-10-01", "2026-10-15")


def test_datetime_values_in_row_are_cut_to_dates() -> None:
    row = row_of(load_snapshot())
    row["dateFrom"] = "2026-10-01 00:00:00"
    row["dateTo"] = "2026-10-31T23:59:59"
    body = cfs.to_compose_input(row)
    assert (body["dateFrom"], body["dateTo"]) == ("2026-10-01", "2026-10-31")


def test_missing_site_or_period_names_the_flags() -> None:
    with pytest.raises(cfs.SnapshotError) as exc:
        cfs.to_compose_input(load_snapshot(), date_from="2026-10-01")
    message = str(exc.value)
    assert "--site-id" in message and "--to" in message and "--from" not in message


def test_no_items_is_an_error() -> None:
    with pytest.raises(cfs.SnapshotError):
        cfs.to_compose_input({"foo": 1}, site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    with pytest.raises(cfs.SnapshotError):
        cfs.to_compose_input([], site_id=1, date_from="2026-10-01", date_to="2026-10-31")


def test_non_item_values_in_snapshot_are_ignored() -> None:
    snap = load_snapshot()
    snap["note"] = "메모"
    snap["empty"] = {}
    body = cfs.to_compose_input(snap, site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    assert len(body["items"]) == 5


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp949"])
def test_read_text_handles_powershell_and_bom_encodings(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "s.json"
    path.write_bytes('{"한글": "위험성평가"}'.encode(encoding))
    assert json.loads(cfs.read_text(path)) == {"한글": "위험성평가"}


def test_mysql_batch_escaping_is_undone() -> None:
    original = {"a": {"queryType": "x", "title": '따옴표 "안" 줄\n바꿈'}}
    mysql_text = json.dumps(original, ensure_ascii=False).replace("\\", "\\\\")  # mysql -B 가 하는 이스케이프
    assert cfs.parse_json_text(mysql_text) == original


def test_broken_json_gives_a_hint() -> None:
    with pytest.raises(cfs.SnapshotError) as exc:
        cfs.parse_json_text("{not json")
    assert "--raw" in str(exc.value)


def test_main_writes_markdown_with_overall_section_and_snapshot_numbers(tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    saved = tmp_path / "input.json"
    code = cfs.main([
        str(SNAPSHOT), "--site-id", "1", "--from", "2026-10-01", "--to", "2026-10-31",
        "--md", str(out), "--input-out", str(saved),
    ])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "종합 내역" in text
    assert "작업 12건, 작업허가 14건, 개선조치 6건" in text
    assert "CP-202610-0002" in text  # 화면에는 내부 id가 아니라 허가 번호
    saved_body = json.loads(saved.read_text(encoding="utf-8"))
    assert saved_body["siteId"] == 1 and len(saved_body["items"]) == 5


def test_main_json_output_has_overall(capsys: pytest.CaptureFixture[str]) -> None:
    code = cfs.main([str(SNAPSHOT), "--site-id", "1", "--from", "2026-10-01", "--to", "2026-10-31", "--json"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE"
    assert result["overall"]["counts"]["works"] == 12


def test_main_reports_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cfs.main([str(SNAPSHOT)]) == 2
    assert "--site-id" in capsys.readouterr().err
    assert cfs.main([str(tmp_path / "없는파일.json"), "--site-id", "1", "--from", "2026-10-01", "--to", "2026-10-31"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cfs.main([str(bad), "--site-id", "1", "--from", "2026-10-01", "--to", "2026-10-31"]) == 2
    reversed_dates = cfs.main([str(SNAPSHOT), "--site-id", "1", "--from", "2026-10-31", "--to", "2026-10-01"])
    assert reversed_dates == 2
    assert "시작일" in capsys.readouterr().err


def test_total_and_truncated_from_spring_are_cross_checked() -> None:
    snap = load_snapshot()
    snap["blocked_records"]["total"] = 99  # Spring이 준 전체 건수가 stats.count와 다르면 경고
    snap["missing_measurements"]["truncated"] = True
    body = cfs.to_compose_input(snap, site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    from app.agent.audit_compose import compose_audit_report
    from app.schemas.audit_compose import AuditComposeInput

    result = compose_audit_report(AuditComposeInput.model_validate(body))
    codes = {v.code for v in result.verification}
    assert {"TOTAL_MISMATCH", "TRUNCATED_MISMATCH"} <= codes
    # 보낸 값이 틀려도 보고서의 숫자는 stats.count 기준이다
    blocked = next(i for g in result.groups for i in g.items if i.query_type == "blocked_records")
    assert blocked.facts.total == 3


def test_input_snapshot_is_not_modified() -> None:
    snap = load_snapshot()
    before = copy.deepcopy(snap)
    cfs.to_compose_input(snap, site_id=1, date_from="2026-10-01", date_to="2026-10-31")
    assert snap == before


def test_script_has_no_llm_or_database_imports() -> None:
    tree = ast.parse(Path(cfs.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = ("llm", "openai", "transformers", "sqlalchemy", "pymysql", "mysql", "requests", "httpx")
    assert not [m for m in imported if any(b in m.lower() for b in banned)], imported
