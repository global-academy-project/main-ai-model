"""scripts/chunk_audit_reports.py 시험. 입력은 합성 텍스트(.txt, 쪽 구분 '\\n\\f\\n')이고 PDF 는 쓰지 않는다."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chunk_audit_reports as ca  # noqa: E402
import load_chunks as lc  # noqa: E402

PAGE = "\n\f\n"


def write_pages(tmp_path: Path, name: str, pages: list[str]) -> Path:
    p = tmp_path / name
    p.write_text(PAGE.join(pages), encoding="utf-8")
    return p


def body(n: int, word: str = "안전관리") -> str:
    """문단 n개. 각 문단은 마침표로 끝나 줄이 문단으로 합쳐지지 않는다."""
    return "\n".join(f"{i}. {word} 점검을 소홀히 하여 위험이 발생하였다." for i in range(1, n + 1))


TITLE_DOC = [
    "감사 결과 보고서\nⅠ. 감사 개요\n감사 대상은 ○○공사이다.\n- 1 -",
    ("일련번호 : 1\n행 정 안 전 부\n주의요구 및 통보\n제 목 추락방호장치 설치 소홀\n소 관 기 관 ○○시\n"
     "내 용\n" + body(3) + "\n- 2 -"),
    ("【별표】 이 줄은 이전 지적사항의 끝이다.\n"
     "일련번호 : 2\n행 정 안 전 부\n시정요구\n제 목 안전난간 관리 부적정\n소 관 기 관 ○○시\n"
     "내 용\n" + body(3, "난간") + "\n- 3 -"),
]


@pytest.fixture
def title_pdf(tmp_path):
    return write_pages(tmp_path, "01_테스트_감사.txt", TITLE_DOC)


def run_cli(tmp_path, *extra, inputs=None, out="out.jsonl"):
    outp = tmp_path / out
    argv = [str(p) for p in (inputs or [])] + ["-o", str(outp), *map(str, extra)]
    code = ca.main(argv)
    rows = [json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines()] if outp.exists() else []
    return code, rows, outp


# ------------------------------------------------------------ 정리
def test_clean_drops_footer_toc_watermark():
    pages = ["제목\n- 3 -\n1. 개요 ········ 5\nINSIDabcdef_:MS_01MS_01\n본문 한 줄"]
    texts = [l for _, l in ca.clean_lines(pages) if l]
    assert texts == ["제목", "본문 한 줄"]


def test_clean_keeps_page_numbers():
    lines = ca.clean_lines(["가\n나", "다"])
    assert [p for p, _ in lines] == [1, 1, 2]


def test_undouble_restores_doubled_title():
    assert ca.undouble("현현장장 안안전전관관리리 이이행행실실태태") == "현장 안전관리 이행실태"


def test_undouble_leaves_normal_text():
    assert ca.undouble("지지") == "지지"  # 어절이 하나뿐이면 건드리지 않는다
    assert ca.undouble("안전 관리 점검 결과") == "안전 관리 점검 결과"
    assert ca.undouble("맨 아래 문장 22") == "맨 아래 문장 22"


# ------------------------------------------------------------ 분할
def test_title_strategy_finds_two_findings(title_pdf):
    lines = ca.clean_lines(ca.extract_pages(title_pdf))
    name, hits = ca.detect(lines)
    assert name == "title"
    assert [h[2] for h in hits] == ["추락방호장치 설치 소홀", "안전난간 관리 부적정"]


def test_header_lines_join_next_finding_not_previous(title_pdf):
    lines = ca.clean_lines(ca.extract_pages(title_pdf))
    _, hits = ca.detect(lines)
    blocks, _ = ca.split_findings(lines, hits)
    first = " ".join(l for _, l in blocks[0]["lines"])
    second = " ".join(l for _, l in blocks[1]["lines"])
    assert "시정요구" not in first and "일련번호 : 2" not in first
    assert second.startswith("일련번호 : 2") or "일련번호 : 2" in second


def test_header_lookback_stops_at_page_boundary(tmp_path):
    pages = [
        "제 목 첫째 지적\n" + body(2) + "\n짧은 마지막 줄",
        "제 목 둘째 지적\n" + body(2),
    ]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    _, hits = ca.detect(lines)
    blocks, _ = ca.split_findings(lines, hits)
    assert "짧은 마지막 줄" in " ".join(l for _, l in blocks[0]["lines"])


def test_title_on_next_line(tmp_path):
    pages = ["제 목\n규정 미비\n" + body(2) + "\n- 1 -", "제 목 두 번째\n" + body(2)]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    _, hits = ca.detect(lines)
    assert hits[0][2] == "규정 미비"


def test_table_header_line_is_not_a_finding(tmp_path):
    pages = ["제목 순번 관계부서 성명\n" + body(2), "제 목 가\n" + body(2), "제 목 나\n" + body(2)]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    _, hits = ca.detect(lines)
    assert [h[2] for h in hits] == ["가", "나"]


def test_circled_strategy(tmp_path):
    pages = ["붙임 문제점 및 개선방안\n① 품질관리 부적정 전 부서\n【 문제점 】\n○ 문제 내용이다.\n【 개선방안 】\n○ 개선하라 (통보)\n"
             "② 안전관리비 점검 소홀 ○○청\n【 문제점 】\n○ 또 다른 문제이다.\n【 개선방안 】\n○ 확인하라 (주의)"]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    name, hits = ca.detect(lines)
    assert name == "circled" and len(hits) == 2


def test_circled_without_problem_heading_is_ignored(tmp_path):
    pages = ["① 개요\n내용\n② 현황\n내용\n③ 계획\n내용"]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    assert ca.detect(lines) == (None, [])


def test_single_match_falls_back_to_sections(tmp_path):
    pages = ["제 목 하나뿐인 지적\n" + body(2)]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    assert ca.detect(lines) == (None, [])


def test_roman_heading_not_in_body_but_becomes_section(tmp_path):
    pages = ["Ⅰ. 설계단계\n제 목 가\n" + body(2), "Ⅱ. 시공단계\n제 목 나\n" + body(2)]
    lines = ca.clean_lines(ca.extract_pages(write_pages(tmp_path, "a.txt", pages)))
    _, hits = ca.detect(lines)
    blocks, _ = ca.split_findings(lines, hits)
    assert [b["section"] for b in blocks] == ["Ⅰ 설계단계", "Ⅱ 시공단계"]
    assert not any("시공단계" in l for _, l in blocks[0]["lines"])


# ------------------------------------------------------------ 문단·청킹
def test_paragraph_join_and_markers():
    lines = [(1, "문장이 줄 끝에서"), (1, "이어지는 중이다."), (1, "○ 새 문단"), (1, "계속된다.")]
    paras = ca.to_paragraphs(lines)
    assert [t for _, t in paras] == ["문장이 줄 끝에서 이어지는 중이다.", "○ 새 문단 계속된다."]


def test_digit_heavy_lines_stay_separate():
    lines = [(1, "표 앞 문장"), (1, "59건 10,044,582 146,759 50,903"), (1, "표 뒤 문장")]
    assert len(ca.to_paragraphs(lines)) == 3


def test_chunks_respect_max_and_keep_order():
    paras = [(1, f"문단{i} " + "가" * 90) for i in range(30)]
    chunks = ca.chunk_paragraphs(paras, 500, 100)
    assert all(len(c["text"]) <= int(500 * 1.25) for c in chunks)
    joined = "\n".join(c["text"] for c in chunks)
    assert [f"문단{i} " for i in range(30)] == [f"문단{i} " for i in range(30) if f"문단{i} " in joined]
    assert joined.index("문단0 ") < joined.index("문단29 ")


def test_long_paragraph_is_split_by_sentence_then_chars():
    text = "첫 문장이다. " * 200
    parts = ca._split_long(text, 300)
    assert all(len(p) <= 300 for p in parts) and len(parts) > 1
    hard = ca._split_long("가" * 1000, 300)
    assert [len(p) for p in hard] == [300, 300, 300, 100]


def test_tail_merge_only_within_limit():
    paras = [(1, "가" * 480), (2, "나" * 30)]
    chunks = ca.chunk_paragraphs(paras, 500, 100)
    assert len(chunks) == 1 and chunks[0]["page_end"] == 2
    paras = [(1, "가" * 500), (2, "나" * 300)]
    assert len(ca.chunk_paragraphs(paras, 500, 100)) == 2
    # 끝 조각이 기준보다 짧아도 합친 길이가 상한의 125%(625자)를 넘으면 붙이지 않는다
    assert len(ca.chunk_paragraphs([(1, "가" * 480), (2, "나" * 120)], 500, 300)) == 1
    assert len(ca.chunk_paragraphs([(1, "가" * 500), (2, "나" * 200)], 500, 300)) == 2


def test_page_range_tracking():
    paras = [(3, "가" * 400), (4, "나" * 400)]
    chunks = ca.chunk_paragraphs(paras, 500, 100)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(3, 3), (4, 4)]


def test_merge_small_merges_cover_into_next():
    chunks = [{"text": "표지", "page_start": 1, "page_end": 1}, {"text": "본문" * 200, "page_start": 2, "page_end": 2}]
    out = ca.merge_small(chunks, 1500, 250)
    assert len(out) == 1 and out[0]["page_start"] == 1 and out[0]["page_end"] == 2
    assert len(ca.merge_small(chunks, 300, 250)) == 2  # 합치면 상한을 넘는다


# ------------------------------------------------------------ 필드
def test_disposition_from_header_lines():
    lines = [(1, "일련번호 : 1"), (1, "주의요구 및 통보"), (1, "제 목 가")]
    assert ca.disposition_of(lines) == ["주의", "통보"]
    assert ca.disposition_of([(1, "통 보(모 범)")]) == ["통보", "모범"]


def test_disposition_falls_back_to_parenthesis():
    lines = [(1, "제 목 가")] + [(1, "본문이 길게 이어지는 문장이다 " * 6)] * 15 + [(2, "개선하라 (통보)")]
    assert ca.disposition_of(lines) == ["통보"]
    assert ca.disposition_of([(1, "제 목 가"), (1, "처분 없음")]) == []


def test_doc_defaults_from_filename():
    d = ca.doc_defaults_from_name(Path("06_시설공사+관리+실태+특정감사+결과(홈페이지).pdf"))
    assert d["doc_id"] == "06" and d["doc_title"] == "시설공사 관리 실태 특정감사 결과"
    d = ca.doc_defaults_from_name(Path("02_붙임.+2025년도+연간감사보고서.pdf"))
    assert d["year"] == 2025
    assert ca.doc_defaults_from_name(Path("보고서.pdf"))["doc_id"] == hashlib.sha256("보고서".encode()).hexdigest()[:6]


# ------------------------------------------------------------ 전체 처리
def test_end_to_end_findings(tmp_path, title_pdf):
    code, rows, _ = run_cli(tmp_path, inputs=[title_pdf])
    assert code == 0
    assert {r["external_id"] for r in rows} == {"AUD-01-001", "AUD-01-002"}
    first = [r for r in rows if r["external_id"] == "AUD-01-001"]
    assert [r["chunk_index"] for r in first] == list(range(len(first)))
    r = first[0]
    assert r["source_type"] == "ACCIDENT_REPORT" and r["file_path"] == "01_테스트_감사.txt"
    assert r["title"] == "추락방호장치 설치 소홀"
    assert r["embedding_text"].startswith("테스트 감사 > 추락방호장치 설치 소홀\n")
    assert r["payload"]["kind"] == "finding" and r["payload"]["disposition"] == ["주의", "통보"]
    assert all(r["payload"][k] == [] for k in ca.EMPTY_TAGS)
    assert r["locator"].startswith("p2")
    assert len(r["content_sha256"]) == 64
    # 개요(첫 지적사항 앞)는 기본으로 넣지 않는다
    assert "감사 대상은" not in "".join(x["text"] for x in rows)


def test_content_sha_matches_chunk_text(tmp_path, title_pdf):
    _, rows, _ = run_cli(tmp_path, inputs=[title_pdf])
    grp = [r for r in rows if r["external_id"] == "AUD-01-001"]
    assert grp[0]["content_sha256"] == hashlib.sha256("\n".join(r["text"] for r in grp).encode()).hexdigest()
    assert len({r["content_sha256"] for r in grp}) == 1


def test_overview_include(tmp_path, title_pdf):
    code, rows, _ = run_cli(tmp_path, "--overview", "include", inputs=[title_pdf])
    assert code == 0
    ovw = [r for r in rows if r["external_id"] == "AUD-01-OVW"]
    assert ovw and ovw[0]["payload"]["kind"] == "overview" and "감사 대상은" in ovw[0]["text"]


def test_use_headings_flag(tmp_path):
    p = write_pages(tmp_path, "03_a.txt", ["Ⅰ. 설계단계\n제 목 가\n" + body(2), "제 목 나\n" + body(2)])
    _, rows, _ = run_cli(tmp_path, inputs=[p])
    assert "section" not in rows[0]["payload"] and "Ⅰ" not in rows[0]["embedding_text"]
    _, rows, _ = run_cli(tmp_path, "--use-headings", inputs=[p])
    assert rows[0]["payload"]["section"] == "Ⅰ 설계단계" and "Ⅰ 설계단계" in rows[0]["embedding_text"]


def test_fallback_sections_single_source(tmp_path):
    p = write_pages(tmp_path, "09_개요.txt", ["제목 없는 보고\n□ 감사 목적\n" + body(3) + "\n□ 감사 결과\n" + body(3, "결과")])
    code, rows, _ = run_cli(tmp_path, inputs=[p])
    assert code == 0
    assert {r["external_id"] for r in rows} == {"AUD-09-DOC"}
    assert all(r["payload"]["kind"] == "section" for r in rows)
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))


def test_manifest_overrides_and_common_defaults(tmp_path, title_pdf):
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps({"01_테스트_감사.txt": {"doc_id": "BSN-A", "title": "부산 시범 감사", "publisher": "부산시",
                                                   "year": 2024, "url": "https://example.org/a", "license": "공공누리"}},
                             ensure_ascii=False), encoding="utf-8")
    code, rows, _ = run_cli(tmp_path, "--manifest", mf, "--publisher", "기본기관", inputs=[title_pdf])
    assert code == 0
    r = rows[0]
    assert r["external_id"].startswith("AUD-BSN-A-") and r["publisher"] == "부산시" and r["license"] == "공공누리"
    assert r["url"] == "https://example.org/a" and r["payload"]["year"] == 2024
    assert r["embedding_text"].startswith("부산 시범 감사 > ")
    code, rows, _ = run_cli(tmp_path, "--publisher", "기본기관", inputs=[title_pdf])
    assert rows[0]["publisher"] == "기본기관"


def test_minimal_payload(tmp_path, title_pdf):
    _, rows, _ = run_cli(tmp_path, "--minimal-payload", inputs=[title_pdf])
    assert set(rows[0]["payload"]) == set(ca.EMPTY_TAGS)


def test_output_is_deterministic(tmp_path, title_pdf):
    _, _, a = run_cli(tmp_path, inputs=[title_pdf], out="a.jsonl")
    _, _, b = run_cli(tmp_path, inputs=[title_pdf], out="b.jsonl")
    assert a.read_bytes() == b.read_bytes()
    assert b"\r\n" not in a.read_bytes()


def test_directory_input_and_sorted_order(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_pages(d, "02_b.txt", TITLE_DOC)
    write_pages(d, "01_a.txt", TITLE_DOC)
    code, rows, _ = run_cli(tmp_path, inputs=[d])
    assert code == 0
    ids = [r["external_id"] for r in rows]
    assert ids[0].startswith("AUD-01-") and ids[-1].startswith("AUD-02-")


def test_output_loads_with_load_chunks(tmp_path, title_pdf):
    _, _, outp = run_cli(tmp_path, inputs=[title_pdf])
    errors: list = []
    reqs, errs = lc.build_requests([outp], {})
    assert not errs
    assert [n for n, _ in reqs] and all(r["source"]["source_type"] == "ACCIDENT_REPORT" for _, r in reqs)
    import ingest_via_api as core
    for _, req in reqs:
        e, _, _ = core.validate_request(req)
        assert not e


# ------------------------------------------------------------ 입력·설정 오류(종료 코드 2)
def test_no_inputs_dir_empty(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    code, _, outp = run_cli(tmp_path, inputs=[d])
    assert code == 2 and not outp.exists()


def test_missing_file(tmp_path):
    code, _, outp = run_cli(tmp_path, inputs=[tmp_path / "nope.txt"])
    assert code == 2 and not outp.exists()


def test_empty_text_file(tmp_path):
    p = write_pages(tmp_path, "01_x.txt", ["- 1 -", "   "])
    code, _, outp = run_cli(tmp_path, inputs=[p])
    assert code == 2 and not outp.exists()


def test_duplicate_doc_id(tmp_path, title_pdf):
    other = write_pages(tmp_path, "01_다른.txt", TITLE_DOC)
    code, _, outp = run_cli(tmp_path, inputs=[title_pdf, other])
    assert code == 2 and not outp.exists()
    # 검증을 끈 상태에서도 doc_id 충돌 자체를 막는다
    code, _, outp = run_cli(tmp_path, "--no-validate", inputs=[title_pdf, other])
    assert code == 2 and not outp.exists()


def test_bad_manifest(tmp_path, title_pdf):
    mf = tmp_path / "m.json"
    mf.write_text("[1,2]", encoding="utf-8")
    assert run_cli(tmp_path, "--manifest", mf, inputs=[title_pdf])[0] == 2
    mf.write_text("{ not json", encoding="utf-8")
    assert run_cli(tmp_path, "--manifest", mf, inputs=[title_pdf])[0] == 2
    mf.write_text(json.dumps({"01": {"year": "2024"}}), encoding="utf-8")
    assert run_cli(tmp_path, "--manifest", mf, inputs=[title_pdf])[0] == 2
    assert run_cli(tmp_path, "--manifest", mf, "--no-validate", inputs=[title_pdf])[0] == 2


def test_validation_error_blocks_write(tmp_path, title_pdf):
    code, _, outp = run_cli(tmp_path, "--publisher", "가" * 101, inputs=[title_pdf])
    assert code == 2 and not outp.exists()
    code, _, outp = run_cli(tmp_path, "--license", "가" * 51, inputs=[title_pdf])
    assert code == 2 and not outp.exists()


@pytest.mark.parametrize("opts", [["--max-chars", "100"], ["--max-chars", "500", "--min-chars", "500"],
                                  ["--min-chars", "-1"]])
def test_bad_size_options(tmp_path, title_pdf, opts):
    code, _, outp = run_cli(tmp_path, *opts, inputs=[title_pdf])
    assert code == 2 and not outp.exists()


def test_title_over_300_chars_is_clipped_not_rejected(tmp_path):
    long_title = "긴제목" * 150
    pages = [f"제 목 {long_title}\n" + body(2), "제 목 짧은 제목\n" + body(2)]
    code, rows, _ = run_cli(tmp_path, inputs=[write_pages(tmp_path, "01_a.txt", pages)])
    assert code == 0 and len(rows[0]["title"]) == 300 and rows[0]["title"].endswith("…")


def test_validation_runs_and_can_be_skipped(tmp_path, title_pdf, monkeypatch):
    calls = []
    monkeypatch.setattr(ca, "validate_rows", lambda rows: (calls.append(len(rows)) or ["가짜 오류"], []))
    code, _, outp = run_cli(tmp_path, inputs=[title_pdf])
    assert code == 2 and calls and not outp.exists()
    calls.clear()
    code, _, outp = run_cli(tmp_path, "--no-validate", inputs=[title_pdf])
    assert code == 0 and not calls and outp.exists()


def test_stats_file(tmp_path, title_pdf):
    st = tmp_path / "s.json"
    code, rows, _ = run_cli(tmp_path, "--stats", st, inputs=[title_pdf])
    data = json.loads(st.read_text(encoding="utf-8"))
    assert code == 0 and data[0]["findings"] == 2 and data[0]["chunks"] == len(rows)


def test_missing_pdfplumber_message(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "01_a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setitem(sys.modules, "pdfplumber", None)  # import 실패를 흉내
    code = ca.main([str(pdf), "-o", str(tmp_path / "o.jsonl")])
    assert code == 2 and "pdfplumber" in capsys.readouterr().out


def test_corrupt_pdf_is_input_error(tmp_path):
    pytest.importorskip("pdfplumber")
    pdf = tmp_path / "01_a.pdf"
    pdf.write_bytes(b"not a pdf")
    assert ca.main([str(pdf), "-o", str(tmp_path / "o.jsonl")]) == 2


# ------------------------------------------------------------ 원본당 청크 수 상한(기본 8)
def long_finding_pages(n_par: int = 80):
    """청크가 여러 개 나오는 긴 지적사항 두 건(문단마다 마침표로 끝나 문단이 합쳐지지 않는다)."""
    return ["제 목 긴 지적 하나\n" + body(n_par) + "\n- 1 -", "제 목 짧은 지적\n" + body(2)]


@pytest.mark.parametrize("n,expect", [(1, [1]), (8, [8]), (9, [5, 4]), (16, [8, 8]), (17, [6, 6, 5]), (25, [7, 6, 6, 6])])
def test_partition_sizes_are_balanced_and_capped(n, expect):
    parts = ca.partition(list(range(n)), 8)
    assert [len(p) for p in parts] == expect
    assert [x for p in parts for x in p] == list(range(n))  # 순서·누락 없음


def test_part_ident():
    assert ca.part_ident("AUD-1-001", "제목", 1, 1) == ("AUD-1-001", "제목")
    assert ca.part_ident("AUD-1-001", "제목", 2, 3) == ("AUD-1-001-P2", "제목 (2/3)")
    pid, title = ca.part_ident("X", "가" * 400, 1, 2)
    assert len(title) == 300 and title.endswith(" (1/2)")


def test_long_finding_is_split_into_sources_of_at_most_8(tmp_path):
    p = write_pages(tmp_path, "01_긴.txt", long_finding_pages())
    code, rows, _ = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", inputs=[p])
    assert code == 0
    by = {}
    for r in rows:
        by.setdefault(r["external_id"], []).append(r)
    assert all(len(v) <= 8 for v in by.values())
    parts = sorted(k for k in by if k.startswith("AUD-01-001-P"))
    assert len(parts) >= 2 and "AUD-01-001" not in by
    assert "AUD-01-002" in by  # 짧은 지적사항은 그대로
    for k in parts:
        assert [r["chunk_index"] for r in by[k]] == list(range(len(by[k])))
        assert by[k][0]["title"].endswith(f"({parts.index(k) + 1}/{len(parts)})")
        assert by[k][0]["embedding_text"].startswith(f"긴 > {by[k][0]['title']}\n")
    assert len({by[k][0]["content_sha256"] for k in parts}) == len(parts)


def test_split_keeps_all_text_in_order(tmp_path):
    p = write_pages(tmp_path, "01_긴.txt", long_finding_pages())
    _, split, _ = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", inputs=[p], out="s.jsonl")
    _, whole, _ = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", "--max-chunks-per-source", "1000",
                          inputs=[p], out="w.jsonl")
    assert "\n".join(r["text"] for r in split if r["external_id"].startswith("AUD-01-001")) == \
        "\n".join(r["text"] for r in whole if r["external_id"] == "AUD-01-001")
    assert {r["external_id"] for r in whole} == {"AUD-01-001", "AUD-01-002"}  # 상한을 올리면 나뉘지 않는다


def test_fallback_doc_is_split_too(tmp_path):
    pages = ["제목 없는 보고\n□ 절 하나\n" + body(120)]
    p = write_pages(tmp_path, "09_개요.txt", pages)
    code, rows, _ = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", inputs=[p])
    assert code == 0
    ids = sorted({r["external_id"] for r in rows})
    assert ids[0].startswith("AUD-09-DOC-P") and len(ids) >= 2
    for i in ids:
        grp = [r for r in rows if r["external_id"] == i]
        assert len(grp) <= 8 and [r["chunk_index"] for r in grp] == list(range(len(grp)))
        assert grp[0]["title"].endswith(f"({ids.index(i) + 1}/{len(ids)})")
        assert grp[0]["embedding_text"].split("\n")[0].split(" > ")[0] == grp[0]["title"]  # 머리말은 (k/n) 붙은 제목으로 시작


def test_default_limit_is_8(tmp_path):
    assert ca.parse_args(["x"]).max_chunks_per_source == 8 == ca.DEFAULT_MAX_CHUNKS


def test_bad_max_chunks_option(tmp_path, title_pdf):
    for v in ("0", "-3"):
        code, _, outp = run_cli(tmp_path, "--max-chunks-per-source", v, inputs=[title_pdf])
        assert code == 2 and not outp.exists()


def test_stats_count_sources_and_splits(tmp_path):
    p = write_pages(tmp_path, "01_긴.txt", long_finding_pages())
    st = tmp_path / "s.json"
    _, rows, _ = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", "--stats", st, inputs=[p])
    data = json.loads(st.read_text(encoding="utf-8"))[0]
    assert data["sources"] == len({r["external_id"] for r in rows}) and data["split_sources"] == 1


def test_split_output_has_no_chunk_count_warning_in_load_chunks(tmp_path):
    p = write_pages(tmp_path, "01_긴.txt", long_finding_pages())
    _, rows, outp = run_cli(tmp_path, "--max-chars", "200", "--min-chars", "50", inputs=[p])
    reqs, errs = lc.build_requests([outp], {})
    assert not errs and all(len(r["chunks"]) <= 8 for _, r in reqs)
    import ingest_via_api as core
    for _, r in reqs:
        e, w, _ = core.validate_request(r)
        assert not e and not any("한 번에 8개" in m for m in w)
