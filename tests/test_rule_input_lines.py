"""어댑터 줄 형식 출력 (11/12): 줄 → dict 변환과, 어댑터일 때 서비스가 줄·JSON 둘 다 읽는지. 모델 없이."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.agent.rule_input_parse import format_lines, lines_to_raw, parse_rule_input
from app.llm.llm_fake import FakeLLMClient, text_response
from app.services.rule_input_extract_service import extract_rule_input

ITEMS = ["O2", "CO2", "LEL", "H2S", "CO", "NH3", "CL2"]
TYPES = ["HOT", "CONFINED", "HEIGHT"]
TEXT = "탱크 안 진입 산소농도 20.3프로 황화수소 13ppm 가연성가스 2 감시자 한다은"
ANSWER = {
    "measurements": [
        {"item": "O2", "value": 20.3, "unit": "PCT", "evidence": "산소농도 20.3프로"},
        {"item": "H2S", "value": 13, "unit": "PPM", "evidence": "황화수소 13ppm"},
        {"item": "LEL", "value": 2, "unit": None, "evidence": "가연성가스 2"},
    ],
    "workTypes": [{"type": "CONFINED", "evidence": "탱크 안 진입"}],
    "watcherName": "한다은", "watcherEvidence": "감시자 한다은",
}
LINES = "O2|20.3|PCT|산소농도 20.3프로\nH2S|13|PPM|황화수소 13ppm\nLEL|2|-|가연성가스 2\nW|CONFINED|탱크 안 진입\nS|한다은|감시자 한다은"


def test_format_and_read_back_give_same_result():
    assert format_lines(ANSWER) == LINES
    a = parse_rule_input(lines_to_raw(LINES), TEXT, ITEMS, TYPES)
    b = parse_rule_input(ANSWER, TEXT, ITEMS, TYPES)
    assert a == b and a.dropped == []
    assert [(x.item, x.value, x.unit) for x in a.measurements] == [
        ("O2", Decimal("20.3"), "PCT"), ("H2S", Decimal("13"), "PPM"), ("LEL", Decimal("2"), None)]   # - → 단위 없음
    assert a.watcher_name == "한다은" and [w.type for w in a.work_types] == ["CONFINED"]


def test_nothing_line_is_parsed_empty_and_garbage_is_not_parsed():
    empty = {"measurements": [], "workTypes": [], "watcherName": None}
    assert format_lines(empty) == "없음"
    raw = lines_to_raw("없음")
    assert raw is not None and parse_rule_input(raw, TEXT, ITEMS, TYPES).measurements == []
    assert lines_to_raw("모르겠습니다") is None
    assert lines_to_raw("") is None


def test_evidence_may_contain_bar_and_extra_spaces():
    raw = lines_to_raw(" o2 | 19 | PCT | 산소 | 19%\n")
    assert raw["measurements"] == [{"item": "O2", "value": "19", "unit": "PCT", "evidence": "산소 | 19%"}]


@pytest.mark.parametrize("cut", [
    "O2|20.3|PCT|산소농도 20.3프로\nH2S|13",             # 칸이 모자란 마지막 줄 → 버림
    "O2|20.3|PCT|산소농도 20.3프로\nH2S|13|PPM|황화수소 1",  # 근거가 끊김 → 근거 숫자 1 이 값 13 과 달라 버림
])
def test_cut_output_keeps_only_complete_lines(cut):
    out = parse_rule_input(lines_to_raw(cut), TEXT, ITEMS, TYPES)
    assert [(x.item, x.value) for x in out.measurements] == [("O2", Decimal("20.3"))]


def _adapter(tmp_path, monkeypatch, reply: str):
    import app.core as core

    folder = tmp_path / "extract"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(core, "get_settings", lambda: SimpleNamespace(llm_adapters={"extract": str(folder)}))

    class AdapterClient(FakeLLMClient):
        adapter = "extract"

    return AdapterClient([text_response(reply)])


def test_adapter_reads_line_output(tmp_path, monkeypatch):
    out = extract_rule_input(_adapter(tmp_path, monkeypatch, LINES), TEXT, ITEMS, TYPES)
    assert out.parsed and len(out.measurements) == 3 and out.watcher_name == "한다은"


def test_adapter_still_reads_json_output_of_previous_adapter(tmp_path, monkeypatch):
    out = extract_rule_input(_adapter(tmp_path, monkeypatch, json.dumps(ANSWER, ensure_ascii=False)), TEXT, ITEMS, TYPES)
    assert out.parsed and len(out.measurements) == 3


def test_adapter_unreadable_output_is_not_parsed(tmp_path, monkeypatch):
    out = extract_rule_input(_adapter(tmp_path, monkeypatch, "모르겠습니다"), TEXT, ITEMS, TYPES)
    assert out.parsed is False


def test_line_output_is_not_read_without_adapter():
    out = extract_rule_input(FakeLLMClient([text_response(LINES)]), TEXT, ITEMS, TYPES)
    assert out.parsed is False   # 긴 프롬프트(어댑터 없음)는 JSON 만 받는다
