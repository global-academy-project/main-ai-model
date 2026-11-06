"""법령 적재 준비 파이프라인(app/law_ingest) 시험.

합성 .hwpx를 만들어 단계별로 확인한다. 실제 파일 시험은 환경변수 LAW_HWPX(와 LAW_TOC)가 있을 때만 돈다.
  PowerShell:  $env:LAW_HWPX = "C:\\...\\산업안전보건기준에관한규칙.hwpx"; python -m pytest tests/test_law_ingest.py -q
"""

from __future__ import annotations

import copy
import io
import json
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from app.law_ingest import config, stages, update, validate
from app.law_ingest.chunk import make_pieces, split_article
from app.law_ingest.cli import main as cli_main
from app.law_ingest.clean import clean_law
from app.law_ingest.config import BuildSettings
from app.law_ingest.fields import build_records, to_ingest_item
from app.law_ingest.models import ParsedLaw, nonspace
from app.law_ingest.parse import ParseError, parse_source
from app.law_ingest.readers import ReaderError, read_source
from app.law_ingest.render import article_lines

NS_S = "http://www.hancom.co.kr/hwpml/2011/section"
NS_P = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _p(text: str) -> str:
    inner = "<hp:lineBreak/>".join(escape(part) for part in text.split("\n"))
    return f"<hp:p><hp:run><hp:t>{inner}</hp:t></hp:run></hp:p>"


def make_hwpx(path: Path, paragraphs: list[str], extra_sections: list[list[str]] | None = None) -> Path:
    def sec(paras: list[str]) -> str:
        return f'<hs:sec xmlns:hs="{NS_S}" xmlns:hp="{NS_P}">' + "".join(_p(x) for x in paras) + "</hs:sec>"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", sec(paragraphs))
        for i, paras in enumerate(extra_sections or [], start=1):
            z.writestr(f"Contents/section{i}.xml", sec(paras))
    path.write_bytes(buf.getvalue())
    return path


LONG_ITEMS = [f"  {i}. 사업주는 {i}번째 항목에 대하여 충분히 긴 설명을 덧붙여서 조각이 나뉘는지 확인한다." for i in range(1, 11)]

SAMPLE = [
    "",
    "테스트안전규칙( 약칭: 테스트규칙)",
    "테스트안전규칙 법제처-  /  - 국가법령정보센터[시행 2026. 3. 2.] [고용노동부령 제450호, 2025. 9. 1., 일부개정]",
    "",
    "고용노동부(산업안전보건정책과) 044-202-8809",
    "     제1편 총칙",
    "       제1장 통칙",
    "제1조(목적) 이 규칙은 테스트를 위한 규칙이다.<개정 2017. 3. 3.>",
    "제2조(정의) 이 규칙에서 사용하는 용어의 뜻은 다음과 같다.",
    "  1. “가”란  가나다를 말한다.",
    "  2. 삭제<2023. 11. 14.>",
    "  3. “나”란 다음 각 목의 것을 말한다.",
    "    가. 첫째 목",
    "    나. 둘째 목",
    "[본조신설 2017. 3. 3.]",
    "제3조 삭제<2024. 6. 28.>",
    "제3조의2(가지 조) ① 사업주는 안전하게 해야 한다.<신설 2019. 12. 26.>",
    "  ② 삭제<2012. 1. 1.>",
    "  ③ 사업주는 제2조에 따라 조치한다.",
    "제3조 및 제2조에 따른 이 줄은 조 시작이 아니다.",
    "제4조(긴 조문) 사업주는 다음 각 호의 사항을 지켜야 한다.",
    *LONG_ITEMS,
    "       제2장 작업",
    "제5조(다음) ① 첫 항입니다.",
    "  ② 둘째 항입니다.",
    "         제1절 하위",
    "             제1속 아래 제목",
    "제6조(마지막) 본문입니다.\n줄바꿈 뒤 글입니다.",
    "    부칙 <고용노동부령 제454호,  2025. 12. 1.>",
    "제1조(시행일) 이 규칙은 공포한 날부터 시행한다.",
    "",
]


@pytest.fixture()
def sample_path(tmp_path: Path) -> Path:
    return make_hwpx(tmp_path / "sample.hwpx", SAMPLE)


@pytest.fixture()
def source(sample_path: Path):
    return read_source(sample_path)


@pytest.fixture()
def parsed(source):
    return parse_source(source)


@pytest.fixture()
def cleaned(parsed):
    return clean_law(parsed)


def art(law: ParsedLaw, label: str):
    return next(a for a in law.articles if a.label == label)


# --- S1 읽기 -------------------------------------------------------------------------


class TestReader:
    def test_meta(self, source):
        m = source.meta
        assert m.law_name == "테스트안전규칙"
        assert m.law_alias == "테스트규칙"
        assert m.effective_date == "2026-03-02"
        assert m.promulgation_no == "고용노동부령 제450호"
        assert m.promulgation_date == "2025-09-01"
        assert m.amendment_type == "일부개정"
        assert m.source_format == "hwpx"
        assert len(m.source_sha256) == 64

    def test_line_break_becomes_newline(self, source):
        texts = [p.text for p in source.paragraphs]
        assert "제6조(마지막) 본문입니다.\n줄바꿈 뒤 글입니다." in texts

    def test_leading_spaces_are_kept(self, source):
        assert any(p.text.startswith("       제1장") for p in source.paragraphs)

    def test_sections_are_read_in_numeric_order(self, tmp_path):
        p = make_hwpx(tmp_path / "multi.hwpx", ["제1조(가) 하나"], extra_sections=[["제2조(나) 둘"]])
        texts = [x.text for x in read_source(p).paragraphs]
        assert texts == ["제1조(가) 하나", "제2조(나) 둘"]

    def test_not_a_zip(self, tmp_path):
        bad = tmp_path / "bad.hwpx"
        bad.write_bytes(b"not a zip")
        with pytest.raises(ReaderError):
            read_source(bad)

    def test_no_section(self, tmp_path):
        p = tmp_path / "nosec.hwpx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("mimetype", "x")
        with pytest.raises(ReaderError):
            read_source(p)

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "law.doc"
        f.write_bytes(b"x")
        with pytest.raises(ReaderError):
            read_source(f)


# --- S2 구조 파싱 --------------------------------------------------------------------


class TestParse:
    def test_articles_and_keys(self, parsed):
        assert [a.label for a in parsed.articles] == ["제1조", "제2조", "제3조", "제3조의2", "제4조", "제5조", "제6조"]
        assert art(parsed, "제3조의2").key == "3-2"

    def test_deleted_article(self, parsed):
        a = art(parsed, "제3조")
        assert a.status == "DELETED" and a.nodes == [] and a.title is None

    def test_inline_first_clause(self, parsed):
        a = art(parsed, "제3조의2")
        assert a.inline_first
        assert [n.marker for n in a.nodes] == ["①", "②", "③"]
        assert all(n.kind == "항" for n in a.nodes)

    def test_lead_ho_mok_nesting(self, parsed):
        a = art(parsed, "제2조")
        (lead,) = a.nodes
        assert lead.kind == "lead"
        assert [c.marker for c in lead.children] == ["1.", "2.", "3."]
        assert [m.marker for m in lead.children[2].children] == ["가.", "나."]

    def test_note_line_is_separated(self, parsed):
        a = art(parsed, "제2조")
        assert a.notes == ["[본조신설 2017. 3. 3.]"]
        assert not any("본조신설" in ln for n in a.nodes for ln in [n.text, *n.lines])

    def test_article_like_line_is_not_a_new_article(self, parsed):
        a = art(parsed, "제3조의2")
        assert len([x for x in parsed.articles if x.label == "제3조의2"]) == 1
        last = a.nodes[-1]
        assert any("조 시작이 아니다" in ln for ln in last.lines)
        assert any(w["code"] == "ARTICLE_LIKE_LINE_NOT_STARTED" for w in parsed.warnings)

    def test_heading_path_includes_all_levels(self, parsed):
        a = art(parsed, "제6조")
        levels = [parsed.headings[i].level for i in a.path_idx]
        assert levels == ["편", "장", "절", "속"][:0] or levels == ["편", "장", "절", "속"]

    def test_heading_resets_lower_levels(self, parsed):
        # 제2장 아래로 오면 제1장 아래 제목은 경로에서 빠진다
        a = art(parsed, "제5조")
        assert [parsed.headings[i].level for i in a.path_idx] == ["편", "장"]
        assert parsed.headings[a.path_idx[1]].number == "2"

    def test_addenda_boundary(self, parsed):
        assert parsed.addenda_src_start is not None
        assert parsed.addenda[0].strip().startswith("부칙")
        assert "공포한 날부터" in " ".join(parsed.addenda)
        assert not any(a.label == "제1조" and "공포한 날" in " ".join(a.raw_lines) for a in parsed.articles)
        # 부칙의 제1조가 본문 제1조를 덮어쓰지 않는다
        assert [a.label for a in parsed.articles].count("제1조") == 1

    def test_soft_line_break_becomes_continuation(self, parsed):
        a = art(parsed, "제6조")
        assert a.nodes[0].text == "본문입니다."
        assert a.nodes[0].lines == ["줄바꿈 뒤 글입니다."]

    def test_no_orphans_in_sample(self, parsed):
        assert parsed.orphans == []

    def test_no_structure_raises(self, tmp_path):
        p = make_hwpx(tmp_path / "x.hwpx", ["그냥 문장입니다.", "또 문장."])
        with pytest.raises(ParseError):
            parse_source(read_source(p))

    def test_unknown_heading_line_is_reported_not_lost(self, tmp_path):
        paras = ["제1장 총칙", "제1조(가) 본문", "제2조(나) 본문", "     제9단 이상한 제목", ]
        # 알 수 없는 제목 줄은 앞 조의 이어지는 줄이 되지만 글자는 보존된다(그리고 경고가 난다)
        p = make_hwpx(tmp_path / "u.hwpx", paras)
        doc = read_source(p)
        law = parse_source(doc)
        total = sum(nonspace(ln) for a in law.articles for ln in a.raw_lines)
        assert total == sum(nonspace(x) for x in paras[1:])
        assert [w["code"] for w in law.warnings] == ["UNKNOWN_HEADING_LEVEL"]


# --- S3 정리 -------------------------------------------------------------------------


class TestClean:
    def test_tags_are_moved_out_of_text(self, cleaned):
        a = art(cleaned, "제1조")
        assert a.nodes[0].text == "이 규칙은 테스트를 위한 규칙이다."
        assert a.amend_tags == ["<개정 2017. 3. 3.>"]

    def test_deleted_clause_is_marked_not_removed_from_numbering(self, cleaned):
        lead = art(cleaned, "제2조").nodes[0]
        assert [c.deleted for c in lead.children] == [False, True, False]
        lines = article_lines(art(cleaned, "제2조"))
        assert not any("삭제" in ln for ln in lines)
        assert any(ln.strip().startswith("3.") for ln in lines)  # 번호는 당기지 않는다

    def test_whitespace_collapses(self, cleaned):
        lines = article_lines(art(cleaned, "제2조"))
        assert any("“가”란 가나다를 말한다." in ln for ln in lines)

    def test_deleted_article_ledger(self, cleaned):
        a = art(cleaned, "제3조")
        assert a.ledger["deleted_article"] == a.ledger["raw"] > 0
        assert a.amend_tags == ["<2024. 6. 28.>"]

    def test_ledger_balances_for_every_article(self, parsed, cleaned):
        for a in cleaned.articles:
            lg = a.ledger
            kept = sum(nonspace(ln) for ln in article_lines(a)) if a.status == "ACTIVE" else 0
            assert lg["raw"] == sum(nonspace(ln) for ln in parsed.articles[a.order].raw_lines)
            assert lg["raw"] == kept + lg["tags"] + lg["notes"] + lg["deleted_nodes"] + lg["deleted_article"], a.label

    def test_non_date_angle_brackets_are_kept(self, tmp_path):
        p = make_hwpx(tmp_path / "t.hwpx", ["제1조(가) 기준은 <표 2>와 같다."])
        law = clean_law(parse_source(read_source(p)))
        assert "<표 2>" in article_lines(law.articles[0])[0]

    def test_heading_label_and_deleted(self, tmp_path):
        p = make_hwpx(
            tmp_path / "h.hwpx",
            ["제1장 총칙 <개정 2012. 5. 31.>", "제7절 삭제 <2024. 6. 28.>", "제1조(가) 본문"],
        )
        law = clean_law(parse_source(read_source(p)))
        assert law.headings[0].label == "제1장 총칙"
        assert law.headings[1].deleted

    def test_clean_does_not_modify_input(self, parsed):
        before = json.dumps(parsed.to_dict(), ensure_ascii=False, sort_keys=True)
        clean_law(parsed)
        assert json.dumps(parsed.to_dict(), ensure_ascii=False, sort_keys=True) == before


# --- S4 청킹 -------------------------------------------------------------------------


class TestChunk:
    def test_short_articles_are_single_pieces(self, cleaned):
        pieces = make_pieces(cleaned, 1000)
        by = {}
        for p in pieces:
            by.setdefault(p.article_key, []).append(p)
        assert len(by["1"]) == 1 and len(by["2"]) == 1
        assert "3" not in by  # 삭제된 조는 조각이 없다

    def test_inline_first_line(self, cleaned):
        (p,) = [x for x in make_pieces(cleaned, 1000) if x.article_key == "1"]
        assert p.lines[0] == "제1조(목적) 이 규칙은 테스트를 위한 규칙이다."

    def test_long_article_splits_at_ho_and_repeats_lead(self, cleaned):
        a = art(cleaned, "제4조")
        pieces = split_article(a, 400)
        assert len(pieces) > 1
        for p in pieces[1:]:
            assert p.lines[0] == "제4조(긴 조문)"
            assert p.lines[1].startswith("사업주는 다음 각 호의 사항")
            assert p.repeated_lead_lines == 2
        assert pieces[0].repeated_lead_lines == 0

    def test_pieces_reassemble_to_article(self, cleaned):
        a = art(cleaned, "제4조")
        for cap in (250, 400, 700, 5000):
            pieces = split_article(a, cap)
            joined = [ln for p in pieces for ln in p.lines[p.repeated_lead_lines :]]
            assert joined == article_lines(a), cap

    def test_pieces_respect_cap_when_possible(self, cleaned):
        for p in split_article(art(cleaned, "제4조"), 400):
            assert (len("\n".join(p.lines)) <= 400) or p.oversize

    def test_oversize_single_item_is_flagged(self, cleaned):
        pieces = split_article(art(cleaned, "제4조"), 80)  # 반복 머리줄 + 호 1개가 80자를 넘는다
        assert any(p.oversize for p in pieces)
        assert not any(p.oversize for p in split_article(art(cleaned, "제4조"), 120))
        joined = [ln for p in pieces for ln in p.lines[p.repeated_lead_lines :]]
        assert joined == article_lines(art(cleaned, "제4조"))

    def test_split_at_hang_boundaries(self):
        from app.law_ingest.models import Article, Node

        nodes = [Node("항", chr(0x2460 + i), "내용" * 60) for i in range(4)]
        a = Article(0, "9", "제9조", "항들", "ACTIVE", [], True, nodes, [], [], [], 0, 0)
        pieces = split_article(a, 300)
        assert len(pieces) >= 2
        assert pieces[0].span.startswith("제1항")
        for p in pieces[1:]:
            assert p.lines[0] == "제9조(항들)" and p.repeated_lead_lines == 1

    def test_part_counts(self, cleaned):
        pieces = [p for p in make_pieces(cleaned, 400) if p.article_key == "4"]
        assert [p.part for p in pieces] == list(range(len(pieces)))
        assert {p.parts_total for p in pieces} == {len(pieces)}


# --- S5 부가 필드 --------------------------------------------------------------------


class TestFields:
    def records(self, cleaned, **kw):
        settings = BuildSettings(**kw)
        return build_records(cleaned, make_pieces(cleaned, settings.max_chars), settings)

    def test_basic_fields(self, cleaned):
        r = next(x for x in self.records(cleaned) if x["article_label"] == "제1조")
        assert r["source_type"] == "LAW" and r["collection"] == "kb_guides"
        assert r["site_id"] == 0 and r["flagged"] is False
        assert r["title"] == "테스트안전규칙" and r["locator"] == "제1조"
        assert r["year"] == 2025  # 공포 연도
        assert r["file_path"] == "raw/sample.hwpx" and r["url"] is None
        assert r["natural_key"] == {"law_key": "테스트안전규칙", "article_key": "1", "part": 0}
        assert r["kb_source_id"] is None and r["kb_chunk_id"] is None and r["point_id"] is None
        assert r["cas"] == [] and r["hazard"] == [] and r["work_type"] == []

    def test_year_basis(self, cleaned):
        r = self.records(cleaned, year_basis="effective")[0]
        assert r["year"] == 2026

    def test_embed_header_on_and_off(self, cleaned):
        on = next(x for x in self.records(cleaned) if x["article_label"] == "제6조")
        assert on["embed_text"].startswith("테스트안전규칙 > 제1편 총칙 > 제2장 작업 > 제1절 하위 > 제1속 아래 제목\n")
        assert on["text"] == on["embed_text"].split("\n", 1)[1]
        off = next(x for x in self.records(cleaned, add_header=False) if x["article_label"] == "제6조")
        assert off["embed_text"] == off["text"]

    def test_text_never_contains_header(self, cleaned):
        assert all("테스트안전규칙 >" not in r["text"] for r in self.records(cleaned))

    def test_locator_is_unique_per_article(self, cleaned):
        recs = self.records(cleaned, max_chars=300)
        pairs = [(r["natural_key"]["article_key"], r["locator"]) for r in recs]
        assert len(pairs) == len(set(pairs))

    def test_tags_default_off_and_switch_on(self, cleaned, monkeypatch):
        from app.law_ingest import tags

        monkeypatch.setattr(tags, "TAG_RULES", [(r"제2장 작업", ["HOT"], ["CHEMICAL"])])
        off = next(x for x in self.records(cleaned) if x["article_label"] == "제5조")
        on = next(x for x in self.records(cleaned, tags_enabled=True) if x["article_label"] == "제5조")
        assert off["work_type"] == [] and on["work_type"] == ["HOT"] and on["hazard"] == ["CHEMICAL"]

    def test_article_hash_changes_with_text(self, cleaned):
        a = next(x for x in self.records(cleaned) if x["article_label"] == "제1조")
        c2 = copy.deepcopy(cleaned)
        art(c2, "제1조").nodes[0].text += " 추가"
        b = next(x for x in self.records(c2) if x["article_label"] == "제1조")
        assert a["article_sha256"] != b["article_sha256"]

    def test_to_ingest_item_requires_ids(self, cleaned):
        with pytest.raises(ValueError):
            to_ingest_item(self.records(cleaned)[0])

    def test_to_ingest_item_matches_ingest_api(self, cleaned):
        ingest = pytest.importorskip("app.ingest_main")
        items = []
        for i, r in enumerate(self.records(cleaned), start=1):
            r = dict(r, kb_source_id=1, kb_chunk_id=i, point_id=f"00000000-0000-5000-8000-{i:012d}")
            items.append(to_ingest_item(r))
        body = ingest.IngestRequest(collection="kb_guides", chunks=items[:32])
        assert len(body.chunks) == min(32, len(items))
        with pytest.raises(Exception):
            ingest.IngestRequest(collection="kb_accidents", chunks=items[:1])


# --- S6 검증 -------------------------------------------------------------------------


class TestValidate:
    def build(self, source, parsed, cleaned, max_chars=1000):
        s = BuildSettings(max_chars=max_chars)
        recs = build_records(cleaned, make_pieces(cleaned, max_chars), s)
        return recs

    def run(self, source, parsed, cleaned, recs, **kw):
        return validate.validate_all(source, parsed, cleaned, recs, **kw)

    def names(self, report, ok=False):
        return {c["name"] for c in report["checks"] if c["ok"] == ok and c["severity"] == "error"}

    def test_passes_on_sample(self, source, parsed, cleaned):
        rep = self.run(source, parsed, cleaned, self.build(source, parsed, cleaned))
        assert rep["ok"], rep["summary"]
        assert rep["summary"]["by_status"] == {"ACTIVE": 6, "DELETED": 1}

    def test_passes_with_small_cap_too(self, source, parsed, cleaned):
        rep = self.run(source, parsed, cleaned, self.build(source, parsed, cleaned, 300))
        assert rep["ok"], rep["summary"]
        assert rep["summary"]["split_articles"] >= 1

    def test_missing_chunk_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs = [r for r in recs if r["article_label"] != "제5조"]
        assert "chunk_coverage" in self.names(self.run(source, parsed, cleaned, recs))

    def test_lost_piece_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned, 300)
        recs = [r for r in recs if not (r["article_label"] == "제4조" and r["natural_key"]["part"] == 1)]
        names = self.names(self.run(source, parsed, cleaned, recs))
        assert "split_reconstruction" in names and "kept_matches_chunks" in names

    def test_noise_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs[0] = dict(recs[0], text=recs[0]["text"] + "\n국가법령정보센터")
        assert "no_noise" in self.names(self.run(source, parsed, cleaned, recs))

    def test_heading_leak_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs[0] = dict(recs[0], text=recs[0]["text"] + "\n             제1속 아래 제목")
        assert "heading_leak" in self.names(self.run(source, parsed, cleaned, recs))

    def test_addenda_leak_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs[0] = dict(recs[0], text=recs[0]["text"] + "\n제1조(시행일) 이 규칙은 공포한 날부터 시행한다.")
        assert "addenda_excluded" in self.names(self.run(source, parsed, cleaned, recs))

    def test_empty_text_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs[0] = dict(recs[0], text="  ")
        assert "chunk_fields" in self.names(self.run(source, parsed, cleaned, recs))

    def test_duplicate_natural_key_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        recs.append(dict(recs[0]))
        assert "natural_keys_unique" in self.names(self.run(source, parsed, cleaned, recs))

    def test_ledger_tamper_is_caught(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        bad = copy.deepcopy(cleaned)
        art(bad, "제1조").ledger["tags"] += 3
        assert "article_ledger" in self.names(self.run(source, parsed, bad, recs))

    def test_golden_checks_known_law(self, source, parsed, cleaned, monkeypatch):
        monkeypatch.setitem(validate.GOLDEN, "테스트안전규칙", [("제2조", ["가나다를 말한다"]), ("제1조", ["없는 문장"])])
        rep = self.run(source, parsed, cleaned, self.build(source, parsed, cleaned))
        assert "golden" in self.names(rep)
        monkeypatch.setitem(validate.GOLDEN, "테스트안전규칙", [("제2조", ["가나다를 말한다"])])
        assert self.run(source, parsed, cleaned, self.build(source, parsed, cleaned))["ok"]

    def test_toc_check(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        toc = ["제1조(목적)", "제2조(정의)", "제3조", "제3조의2(가지 조)", "제4조(긴 조문)", "제5조(다음)", "제6조(마지막)"]
        assert self.run(source, parsed, cleaned, recs, toc_lines=toc)["ok"]
        assert "toc_match" in self.names(self.run(source, parsed, cleaned, recs, toc_lines=[*toc, "제7조(없음)"]))
        swapped = [toc[1], toc[0], *toc[2:]]
        assert "toc_match" in self.names(self.run(source, parsed, cleaned, recs, toc_lines=swapped))

    def test_moved_article_in_toc_is_explained(self, tmp_path):
        paras = ["제1장 총칙", "제1조(가) 본문", "[제2조에서 이동 <2012. 3. 5.>]", "제3조(다) 본문"]
        p = make_hwpx(tmp_path / "m.hwpx", paras)
        src = read_source(p)
        par = parse_source(src)
        cl = clean_law(par)
        recs = build_records(cl, make_pieces(cl, 1000), BuildSettings())
        rep = validate.validate_all(src, par, cl, recs, toc_lines=["제1조(가)", "제2조(나)", "제3조(다)"])
        assert rep["ok"], rep["summary"]

    def test_expected_article_count(self, source, parsed, cleaned):
        recs = self.build(source, parsed, cleaned)
        assert self.run(source, parsed, cleaned, recs, expect_articles=7)["ok"]
        assert "articles_expected" in self.names(self.run(source, parsed, cleaned, recs, expect_articles=8))


# --- 단계별 실행 · CLI -----------------------------------------------------------------


class TestStagesAndCli:
    def test_run_all_writes_every_stage_file(self, sample_path, tmp_path):
        out = tmp_path / "out"
        assert cli_main(["run", str(sample_path), "--out", str(out)]) == 0
        for name in (
            stages.F_MANIFEST,
            stages.F_SOURCE,
            stages.F_PARSED,
            stages.F_CLEANED,
            stages.F_PIECES,
            stages.F_CHUNKS,
            stages.F_VALIDATION,
        ):
            assert (out / name).exists(), name
        assert (out / "raw" / "sample.hwpx").read_bytes() == sample_path.read_bytes()
        rep = json.loads((out / stages.F_VALIDATION).read_text(encoding="utf-8"))
        assert rep["ok"]

    def test_stage_by_stage_equals_run_all(self, sample_path, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        assert cli_main(["run", str(sample_path), "--out", str(a)]) == 0
        for cmd in (["read", str(sample_path)], ["parse"], ["clean"], ["chunk"], ["fields"], ["validate"]):
            assert cli_main([cmd[0], *cmd[1:], "--out", str(b)]) == 0
        assert (a / stages.F_CHUNKS).read_text(encoding="utf-8") == (b / stages.F_CHUNKS).read_text(encoding="utf-8")

    def test_stage_can_be_rerun_from_its_input_file(self, sample_path, tmp_path):
        out = tmp_path / "out"
        cli_main(["run", str(sample_path), "--out", str(out)])
        before = (out / stages.F_CHUNKS).read_text(encoding="utf-8")
        assert cli_main(["chunk", "--out", str(out), "--max-chars", "300"]) == 0
        assert cli_main(["fields", "--out", str(out), "--max-chars", "300"]) == 0
        after = (out / stages.F_CHUNKS).read_text(encoding="utf-8")
        assert before != after and len(after.splitlines()) > len(before.splitlines())
        assert cli_main(["validate", "--out", str(out)]) == 0

    def test_no_header_option(self, sample_path, tmp_path):
        out = tmp_path / "out"
        assert cli_main(["run", str(sample_path), "--out", str(out), "--no-header"]) == 0
        recs = stages.load_chunks(out)
        assert all(r["embed_text"] == r["text"] for r in recs)

    def test_missing_previous_stage_is_reported(self, tmp_path, capsys):
        assert cli_main(["parse", "--out", str(tmp_path / "empty")]) == 2
        assert "앞 단계" in capsys.readouterr().err

    def test_bad_input_is_reported(self, tmp_path, capsys):
        bad = tmp_path / "bad.hwpx"
        bad.write_bytes(b"x")
        assert cli_main(["run", str(bad), "--out", str(tmp_path / "o")]) == 2

    def test_failed_validation_returns_1(self, sample_path, tmp_path):
        out = tmp_path / "out"
        cli_main(["run", str(sample_path), "--out", str(out)])
        lines = (out / stages.F_CHUNKS).read_text(encoding="utf-8").splitlines()
        (out / stages.F_CHUNKS).write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
        assert cli_main(["validate", "--out", str(out)]) == 1

    def test_stage_files_round_trip(self, sample_path, tmp_path):
        out = tmp_path / "out"
        cli_main(["run", str(sample_path), "--out", str(out)])
        law = stages.load_cleaned(out)
        again = ParsedLaw.from_dict(json.loads(json.dumps(law.to_dict(), ensure_ascii=False)))
        assert again.to_dict() == law.to_dict()

    def test_settings_validation(self):
        with pytest.raises(ValueError):
            BuildSettings(max_chars=10)
        with pytest.raises(ValueError):
            BuildSettings(year_basis="x")

    def test_config_constants_match_a_contract(self):
        from app.rag.filters import COLLECTIONS

        assert config.SOURCE_TYPE in COLLECTIONS[config.COLLECTION]


# --- 국가법령정보센터 API XML -----------------------------------------------------------------


def _x(tag: str, text: str = "", **attrs: str) -> str:
    return f"<{tag}>{escape(text)}</{tag}>" if text or not attrs else f"<{tag}/>"


def xml_unit(number: str, content: str, title: str = "", branch: str = "", kind: str = "조문", children: str = "") -> str:
    return (
        '<조문단위 조문키="0000000">'
        f"<조문번호>{number}</조문번호><조문가지번호>{branch}</조문가지번호><조문여부>{kind}</조문여부>"
        f"<조문제목>{escape(title)}</조문제목><조문내용><![CDATA[{content}]]></조문내용>{children}</조문단위>"
    )


def xml_item(tag: str, number: str, content: str, children: str = "") -> str:
    return f"<{tag}><{tag}번호>{escape(number)}</{tag}번호><{tag}내용><![CDATA[{content}]]></{tag}내용>{children}</{tag}>"


def law_xml(units: list[str], addenda: str = "", info: str | None = None, root: str = "법령") -> str:
    info = info if info is not None else (
        "<기본정보><법령ID>1</법령ID><공포일자>20250901</공포일자><공포번호>450</공포번호>"
        '<법종구분 법종구분코드="A0003">부령</법종구분><법령명_한글><![CDATA[테스트안전규칙]]></법령명_한글>'
        "<법령명약칭>테스트규칙</법령명약칭><소관부처>고용노동부</소관부처><시행일자>20260302</시행일자>"
        "<제개정구분>일부개정</제개정구분></기본정보>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<{root} 법령키="1">{info}<조문>{"".join(units)}</조문>{addenda}<별표><별표단위><별표내용>무시</별표내용></별표단위></별표></{root}>'
    )


XML_UNITS = [
    xml_unit("", "제1편 총칙", kind="전문"),
    xml_unit("", "     제1장 통칙", kind="전문"),
    xml_unit("1", "제1조(목적) 이 규칙은 테스트를 위한 규칙이다.<개정 2017. 3. 3.>", "목적"),
    xml_unit(
        "2",
        "제2조(정의) 이 규칙에서 사용하는 용어의 뜻은 다음과 같다.",
        "정의",
        children=(
            xml_item("호", "1.", "1. “가”란  가나다를 말한다.")
            + xml_item("호", "2.", "2. 삭제<2023. 11. 14.>")
            + xml_item("호", "3.", "3. “나”란 다음 각 목의 것을 말한다.", xml_item("목", "가.", "가. 첫째 목") + xml_item("목", "나.", "나. 둘째 목"))
        ),
    ),
    xml_unit("3", "제3조 삭제<2024. 6. 28.>"),
    xml_unit(
        "3",
        "제3조의2(가지 조)",
        "가지 조",
        branch="2",
        children=(
            xml_item("항", "①", "① 사업주는 안전하게 해야 한다.<신설 2019. 12. 26.>")
            + xml_item("항", "②", "② 삭제<2012. 1. 1.>")
            + xml_item("항", "③", "③ 사업주는 제2조에 따라 조치한다.")
        ),
    ),
    xml_unit(
        "4",
        "제4조(긴 조문) 사업주는 다음 각 호의 사항을 지켜야 한다.",
        "긴 조문",
        children="".join(xml_item("호", f"{i}.", ln.strip()) for i, ln in enumerate(LONG_ITEMS, start=1)),
    ),
    xml_unit("", "       제2장 작업", kind="전문"),
    xml_unit("", "         제1절 하위", kind="전문"),
    xml_unit("", "             제1속 아래 제목", kind="전문"),
    xml_unit("6", "제6조(마지막) 본문입니다.\n줄바꿈 뒤 글입니다.", "마지막"),
]
XML_ADDENDA = "<부칙><부칙단위><부칙내용><![CDATA[부칙 <고용노동부령 제454호,  2025. 12. 1.>\n제1조(시행일) 이 규칙은 공포한 날부터 시행한다.]]></부칙내용></부칙단위></부칙>"

# 위 XML과 같은 법령을 hwpx 모양으로 쓴 것(비교용)
HWPX_EQUIV = [
    "테스트안전규칙( 약칭: 테스트규칙)",
    "테스트안전규칙 법제처-  /  - 국가법령정보센터[시행 2026. 3. 2.] [고용노동부령 제450호, 2025. 9. 1., 일부개정]",
    "     제1편 총칙",
    "       제1장 통칙",
    "제1조(목적) 이 규칙은 테스트를 위한 규칙이다.<개정 2017. 3. 3.>",
    "제2조(정의) 이 규칙에서 사용하는 용어의 뜻은 다음과 같다.",
    "  1. “가”란  가나다를 말한다.",
    "  2. 삭제<2023. 11. 14.>",
    "  3. “나”란 다음 각 목의 것을 말한다.",
    "    가. 첫째 목",
    "    나. 둘째 목",
    "제3조 삭제<2024. 6. 28.>",
    "제3조의2(가지 조) ① 사업주는 안전하게 해야 한다.<신설 2019. 12. 26.>",
    "  ② 삭제<2012. 1. 1.>",
    "  ③ 사업주는 제2조에 따라 조치한다.",
    "제4조(긴 조문) 사업주는 다음 각 호의 사항을 지켜야 한다.",
    *LONG_ITEMS,
    "       제2장 작업",
    "         제1절 하위",
    "             제1속 아래 제목",
    "제6조(마지막) 본문입니다.\n줄바꿈 뒤 글입니다.",
    "    부칙 <고용노동부령 제454호,  2025. 12. 1.>",
    "제1조(시행일) 이 규칙은 공포한 날부터 시행한다.",
]


@pytest.fixture()
def xml_path(tmp_path: Path) -> Path:
    p = tmp_path / "law.xml"
    p.write_text(law_xml(XML_UNITS, XML_ADDENDA), encoding="utf-8")
    return p


class TestXmlReader:
    def test_meta(self, xml_path):
        m = read_source(xml_path).meta
        assert (m.law_name, m.law_alias, m.effective_date) == ("테스트안전규칙", "테스트규칙", "2026-03-02")
        assert (m.promulgation_no, m.promulgation_date, m.amendment_type) == ("고용노동부령 제450호", "2025-09-01", "일부개정")
        assert m.source_format == "xml" and len(m.source_sha256) == 64

    def test_meta_matches_hwpx_meta(self, xml_path, tmp_path):
        x = read_source(xml_path).meta
        h = read_source(make_hwpx(tmp_path / "e.hwpx", HWPX_EQUIV)).meta
        for f in ("law_name", "law_alias", "effective_date", "promulgation_no", "promulgation_date", "amendment_type"):
            assert getattr(x, f) == getattr(h, f), f

    def test_title_only_article_gets_first_clause_on_same_line(self, xml_path):
        texts = [p.text for p in read_source(xml_path).paragraphs]
        assert "제3조의2(가지 조) ① 사업주는 안전하게 해야 한다.<신설 2019. 12. 26.>" in texts
        assert "  ② 삭제<2012. 1. 1.>" in texts

    def test_same_chunks_as_hwpx(self, xml_path, tmp_path):
        outs = {}
        for name, src in (("x", xml_path), ("h", make_hwpx(tmp_path / "e.hwpx", HWPX_EQUIV))):
            out = tmp_path / name
            assert cli_main(["run", str(src), "--out", str(out), "--max-chars", "300"]) == 0
            outs[name] = stages.load_chunks(out)
        keys = ("natural_key", "text", "embed_text", "locator", "hierarchy", "year", "article_sha256")
        assert [{k: r[k] for k in keys} for r in outs["x"]] == [{k: r[k] for k in keys} for r in outs["h"]]

    def test_update_plan_between_hwpx_and_xml_builds_is_empty(self, xml_path, tmp_path):
        old, new = tmp_path / "h", tmp_path / "x"
        assert cli_main(["run", str(make_hwpx(tmp_path / "e.hwpx", HWPX_EQUIV)), "--out", str(old), "--max-chars", "300"]) == 0
        assert cli_main(["run", str(xml_path), "--out", str(new), "--max-chars", "300", "--prev", str(old)]) == 0
        s = json.loads((new / stages.F_CHANGES).read_text(encoding="utf-8"))["summary"]
        assert (s["add"], s["update"], s["remove"]) == (0, 0, 0)

    def test_addenda_and_byeolpyo_are_not_chunked(self, xml_path, tmp_path):
        out = tmp_path / "o"
        assert cli_main(["run", str(xml_path), "--out", str(out)]) == 0
        body = "\n".join(r["text"] for r in stages.load_chunks(out))
        assert "공포한 날부터" not in body and "무시" not in body

    def test_numbers_missing_in_content_are_added_from_number_tags(self, tmp_path):
        units = [
            xml_unit("", "제1장 총칙", kind="전문"),
            xml_unit(
                "1",
                "제1조(가)",
                "가",
                children=xml_item("항", "①", "사업주는 지켜야 한다.", xml_item("호", "1.", "첫째", xml_item("목", "가.", "목 내용")))
                + xml_item("항", "②", "둘째 항이다."),
            ),
            xml_unit("2", "본문만 있다.", "번호없음"),
        ]
        p = tmp_path / "n.xml"
        p.write_text(law_xml(units), encoding="utf-8")
        law = parse_source(read_source(p))
        a1, a2 = law.articles
        assert [n.marker for n in a1.nodes] == ["①", "②"]
        assert a1.nodes[0].children[0].marker == "1." and a1.nodes[0].children[0].children[0].marker == "가."
        assert (a2.label, a2.title) == ("제2조", "번호없음")

    def test_note_inside_content_goes_to_end_and_does_not_block_merge(self, tmp_path):
        units = [
            xml_unit("", "제1장 총칙", kind="전문"),
            xml_unit("1", "제1조(가)\n[본조신설 2017. 3. 3.]", "가", children=xml_item("항", "①", "① 첫 항이다.") + xml_item("항", "②", "② 둘째 항이다.")),
        ]
        p = tmp_path / "nt.xml"
        p.write_text(law_xml(units), encoding="utf-8")
        texts = [x.text for x in read_source(p).paragraphs]
        assert texts[1:] == ["제1조(가) ① 첫 항이다.", "  ② 둘째 항이다.", "[본조신설 2017. 3. 3.]"]
        (a,) = parse_source(read_source(p)).articles
        assert a.notes == ["[본조신설 2017. 3. 3.]"] and a.inline_first

    def test_overlapping_clause_text_is_kept_once(self, tmp_path):
        units = [
            xml_unit("", "제1장 총칙", kind="전문"),
            xml_unit("1", "제1조(가) ① 사업주는 반드시 지켜야 한다.\n② 둘째 항도 반드시 지켜야 한다.", "가", children=xml_item("항", "①", "① 사업주는 반드시 지켜야 한다.") + xml_item("항", "②", "② 둘째 항도 반드시 지켜야 한다.")),
        ]
        p = tmp_path / "d.xml"
        p.write_text(law_xml(units), encoding="utf-8")
        text = "\n".join(x.text for x in read_source(p).paragraphs)
        assert text.count("사업주는 반드시") == 1 and text.count("둘째 항도") == 1

    def test_article_directly_under_root_wrapper(self, tmp_path):
        p = tmp_path / "w.xml"
        inner = law_xml(XML_UNITS, XML_ADDENDA).split("?>", 1)[1]
        p.write_text("<법령서비스>" + inner + "</법령서비스>", encoding="utf-8")
        assert read_source(p).meta.law_name == "테스트안전규칙"

    @pytest.mark.parametrize(
        "body,msg",
        [
            ("<html><body>로그인이 필요합니다</body></html>", "법령 본문 XML이 아닙니다"),
            ("<LawSearch><law><법령명한글>x</법령명한글></law></LawSearch>", "법령 본문 XML이 아닙니다"),
            ("<Law>일치하는 법령이 없습니다.</Law>", "법령 본문 XML이 아닙니다"),
            ("미신청된 목록/본문에 대한 접근입니다.", "XML이 아닙니다"),
            ('<!DOCTYPE x [<!ENTITY a "b">]><법령/>', "DOCTYPE"),
            ("<법령><기본정보/></법령>", "조문단위"),
            ("<법령><조문>", "해석"),
        ],
    )
    def test_not_a_law_body_is_rejected_with_a_reason(self, tmp_path, body, msg):
        p = tmp_path / "bad.xml"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ReaderError, match=msg):
            read_source(p)

    def test_addenda_header_is_added_only_when_missing(self, tmp_path):
        def addenda_lines(content):
            add = f"<부칙><부칙단위><부칙공포일자>20251201</부칙공포일자><부칙공포번호>454</부칙공포번호><부칙내용><![CDATA[{content}]]></부칙내용></부칙단위></부칙>"
            p = tmp_path / "a.xml"
            p.write_text(law_xml(XML_UNITS, add), encoding="utf-8")
            doc = read_source(p)
            law = parse_source(doc)
            return [x.text for x in doc.paragraphs if x.text.startswith("부칙")], law.addenda

        has, addenda = addenda_lines("부칙 <고용노동부령 제454호, 2025. 12. 1.>\n제1조(시행일) 공포한 날부터 시행한다.")
        assert len(has) == 1 and len(addenda) == 2
        missing, addenda = addenda_lines("제1조(시행일) 공포한 날부터 시행한다.")
        assert missing == ["부칙 <제454호, 2025-12-01>"] and len(addenda) == 2  # 머리가 없으면 만들어 붙여 본문 제1조와 섞이지 않게 한다

    def test_bom_is_accepted(self, tmp_path):
        p = tmp_path / "b.xml"
        p.write_bytes(b"\xef\xbb\xbf" + law_xml(XML_UNITS, XML_ADDENDA).encode("utf-8"))
        assert read_source(p).meta.law_name == "테스트안전규칙"

    def test_full_cli_with_validation_and_toc(self, xml_path, tmp_path):
        toc = tmp_path / "toc.txt"
        toc.write_text("제1조(목적)\n제2조(정의)\n제3조\n제3조의2(가지 조)\n제4조(긴 조문)\n제6조(마지막)\n", encoding="utf-8")
        out = tmp_path / "o"
        assert cli_main(["run", str(xml_path), "--out", str(out), "--toc", str(toc), "--expect-articles", "6"]) == 0
        assert json.loads((out / stages.F_VALIDATION).read_text(encoding="utf-8"))["ok"]


# --- 제목 단계(속 등) ------------------------------------------------------------------


class TestHeadingLevels:
    PARAS = [
        "제1편 총칙",
        "   제1장 굴착",
        "     제1관 노천굴착",
        "       제1속 기울기",
        "제1조(가) 본문",
        "       제2속 지보공",
        "제2조(나) 본문",
        "     제2관 터널",
        "       제1속 조사",
        "제3조(다) 본문",
    ]

    def law(self, tmp_path, levels=None):
        return parse_source(read_source(make_hwpx(tmp_path / "l.hwpx", self.PARAS)), levels)

    def test_sok_is_its_own_level_and_restarts_under_each_gwan(self, tmp_path):
        law = clean_law(self.law(tmp_path))
        assert [h.level for h in law.headings] == ["편", "장", "관", "속", "속", "관", "속"]
        by = {a.label: [law.headings[i].label for i in a.path_idx] for a in law.articles}
        assert by["제1조"][-2:] == ["제1관 노천굴착", "제1속 기울기"]
        assert by["제2조"][-1] == "제2속 지보공"
        assert by["제3조"][-2:] == ["제2관 터널", "제1속 조사"]  # 새 관이 오면 이전 속은 빠진다

    def test_heading_text_never_enters_article_body(self, tmp_path):
        law = self.law(tmp_path)
        assert all("속" not in ln for a in law.articles for ln in a.raw_lines if not ln.startswith("제"))
        assert law.orphans == []

    def test_hierarchy_field_splits_each_level(self, tmp_path):
        law = clean_law(self.law(tmp_path))
        recs = build_records(law, make_pieces(law, 1000))
        h = {r["article_label"]: r["hierarchy"] for r in recs}
        assert h["제1조"] == {"편": "제1편 총칙", "장": "제1장 굴착", "관": "제1관 노천굴착", "속": "제1속 기울기"}
        assert h["제3조"]["속"] == "제1속 조사" and h["제3조"]["관"] == "제2관 터널"
        assert list(h["제1조"].values()) == recs[0]["path"]  # path와 같은 순서·내용

    def test_missing_level_has_no_key(self, tmp_path):
        p = make_hwpx(tmp_path / "x.hwpx", ["제1장 총칙", "제1조(가) 본문"])
        law = clean_law(parse_source(read_source(p)))
        (r,) = build_records(law, make_pieces(law, 1000))
        assert r["hierarchy"] == {"장": "제1장 총칙"}

    def test_new_level_is_added_by_config_only(self, tmp_path):
        paras = ["제1장 총칙", "     제1관 가", "       제1속 나", "         제1단 다", "제1조(가) 본문"]
        p = make_hwpx(tmp_path / "d.hwpx", paras)
        default = parse_source(read_source(p))
        assert "UNKNOWN_HEADING_LEVEL" in [w["code"] for w in default.warnings]
        law = parse_source(read_source(p), ("편", "장", "절", "관", "속", "단"))
        assert [h.level for h in law.headings] == ["장", "관", "속", "단"]
        assert law.warnings == [] and len(law.articles[0].path_idx) == 4

    def test_level_without_it_is_caught_by_validation(self, tmp_path):
        p = make_hwpx(tmp_path / "v.hwpx", self.PARAS)
        src = read_source(p)
        par = parse_source(src, ("편", "장", "관"))  # 속을 빼먹은 설정
        cl = clean_law(par)
        recs = build_records(cl, make_pieces(cl, 1000))
        rep = validate.validate_all(src, par, cl, recs)
        names = {c["name"] for c in rep["checks"] if not c["ok"] and c["severity"] == "error"}
        assert not rep["ok"] and "unknown_heading_level" in names

    def test_bad_level_setting(self, tmp_path):
        with pytest.raises(ParseError):
            self.law(tmp_path, ("편", "장장"))
        with pytest.raises(ParseError):
            self.law(tmp_path, ("편", "편"))

    def test_cli_heading_levels_option(self, tmp_path):
        p = make_hwpx(tmp_path / "c.hwpx", ["제1장 총칙", "     제1관 가", "         제1단 다", "제1조(가) 본문"])
        out = tmp_path / "o"
        assert cli_main(["run", str(p), "--out", str(out)]) == 1  # 단을 몰라 검증 실패
        assert cli_main(["run", str(p), "--out", str(out), "--heading-levels", "편,장,절,관,속,단"]) == 0


# --- 갱신 계획(S7) ----------------------------------------------------------------------


def amended(**changes) -> list[str]:
    """SAMPLE을 개정한 문서. 키워드: edit2, add7, delete5, shrink4, date."""
    out = list(SAMPLE)
    if changes.get("date"):
        out[2] = out[2].replace("[시행 2026. 3. 2.]", f"[시행 {changes['date']}]")
    if changes.get("edit2"):
        out[out.index("  1. “가”란  가나다를 말한다.")] = "  1. “가”란  가나다라를 말한다."
    if changes.get("delete5"):
        i = out.index("제5조(다음) ① 첫 항입니다.")
        out[i : i + 2] = ["제5조 삭제<2026. 3. 2.>"]
    if changes.get("shrink4"):
        i = out.index("제4조(긴 조문) 사업주는 다음 각 호의 사항을 지켜야 한다.")
        out[i + 1 : i + 1 + len(LONG_ITEMS)] = LONG_ITEMS[:2]
    if changes.get("add7"):
        i = out.index("    부칙 <고용노동부령 제454호,  2025. 12. 1.>")
        out[i:i] = ["제7조(신설) 새 조문입니다."]
    return out


class TestUpdate:
    def build(self, tmp_path, name, paras, max_chars=300):
        src = make_hwpx(tmp_path / f"{name}.hwpx", paras)
        out = tmp_path / name
        assert cli_main(["run", str(src), "--out", str(out), "--max-chars", str(max_chars)]) == 0
        return out

    def plan(self, tmp_path, **changes):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", amended(**changes))
        return stages.stage_diff(new, old, allow_mass_remove=True)

    def labels(self, plan, kind):
        return sorted({e["article_label"] for e in plan[kind]})

    def test_identical_build_has_no_changes(self, tmp_path):
        plan = self.plan(tmp_path)
        s = plan["summary"]
        assert (s["add"], s["update"], s["remove"]) == (0, 0, 0) and s["unchanged"] == s["new_chunks"]
        assert s["same_source_file"] is True
        assert (tmp_path / "new" / stages.F_UPSERT).read_text(encoding="utf-8") == ""

    def test_edited_article_is_update_with_diff(self, tmp_path):
        plan = self.plan(tmp_path, edit2=True)
        assert self.labels(plan, "update") == ["제2조"] and not plan["add"] and not plan["remove"]
        assert any("가나다라" in ln for ln in plan["update"][0]["diff"])
        assert plan["summary"]["same_source_file"] is False

    def test_new_article_is_add(self, tmp_path):
        plan = self.plan(tmp_path, add7=True)
        assert self.labels(plan, "add") == ["제7조"] and not plan["update"] and not plan["remove"]

    def test_article_deleted_by_amendment_is_remove(self, tmp_path):
        plan = self.plan(tmp_path, delete5=True)
        assert self.labels(plan, "remove") == ["제5조"]
        assert {e["reason"] for e in plan["remove"]} == {"ARTICLE_DELETED"}
        assert plan["summary"]["needs_review"] == []

    def test_fewer_parts_removes_the_dropped_parts(self, tmp_path):
        plan = self.plan(tmp_path, shrink4=True)
        assert {e["reason"] for e in plan["remove"]} == {"PART_DROPPED"}
        assert self.labels(plan, "remove") == ["제4조"]
        kept = {e["natural_key"]["part"] for e in plan["update"] + plan["add"]}
        dropped = {e["natural_key"]["part"] for e in plan["remove"]}
        assert not kept & dropped  # 올리는 조각과 지우는 조각이 겹치지 않는다

    def test_article_vanishing_is_flagged_for_review(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        paras = [x for x in SAMPLE if not x.startswith("제6조") and "줄바꿈 뒤" not in x]
        new = self.build(tmp_path, "new", paras)
        plan = stages.stage_diff(new, old, allow_mass_remove=True)
        assert {e["reason"] for e in plan["remove"]} == {"ARTICLE_GONE"}
        assert plan["summary"]["needs_review"]

    def test_upsert_file_has_exactly_add_and_update(self, tmp_path):
        plan = self.plan(tmp_path, edit2=True, add7=True)
        rows = stages.load_chunks(tmp_path / "new") and [
            json.loads(x) for x in (tmp_path / "new" / stages.F_UPSERT).read_text(encoding="utf-8").splitlines()
        ]
        assert sorted(r["article_label"] for r in rows) == ["제2조", "제7조"]
        assert len(rows) == plan["summary"]["add"] + plan["summary"]["update"]

    def test_mass_remove_guard(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", amended(delete5=True))
        with pytest.raises(ValueError, match="allow-mass-remove"):
            stages.stage_diff(new, old)
        assert stages.stage_diff(new, old, allow_mass_remove=True)["summary"]["remove"] >= 1

    def test_older_file_is_refused(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", amended(date="2026. 1. 1."))
        with pytest.raises(ValueError, match="시행일"):
            stages.stage_diff(new, old)
        stages.stage_diff(new, old, allow_older=True)

    def test_other_law_is_refused(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        other = [x.replace("테스트안전규칙", "다른규칙") for x in SAMPLE]
        new = self.build(tmp_path, "new", other)
        with pytest.raises(ValueError, match="법령 식별자"):
            stages.stage_diff(new, old)

    def test_failed_validation_blocks_diff(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", SAMPLE)
        rep = json.loads((new / stages.F_VALIDATION).read_text(encoding="utf-8"))
        rep["ok"] = False
        (new / stages.F_VALIDATION).write_text(json.dumps(rep), encoding="utf-8")
        with pytest.raises(ValueError, match="검증"):
            stages.stage_diff(new, old)

    def test_untrusted_previous_build_is_refused(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", SAMPLE)
        rep = json.loads((old / stages.F_VALIDATION).read_text(encoding="utf-8"))
        rep["ok"] = False
        (old / stages.F_VALIDATION).write_text(json.dumps(rep), encoding="utf-8")
        with pytest.raises(ValueError, match="이전"):
            stages.stage_diff(new, old)

    def test_file_name_change_alone_is_not_an_update(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        src = make_hwpx(tmp_path / "renamed.hwpx", SAMPLE)
        new = tmp_path / "new"
        assert cli_main(["run", str(src), "--out", str(new), "--max-chars", "300"]) == 0
        recs = stages.load_chunks(new)
        assert all(r["file_path"] == "raw/renamed.hwpx" for r in recs)
        assert stages.stage_diff(new, old)["summary"]["update"] == 0

    def test_header_or_year_change_is_an_update(self, tmp_path):
        old = self.build(tmp_path, "old", SAMPLE)
        src = make_hwpx(tmp_path / "s.hwpx", SAMPLE)
        new = tmp_path / "new"
        assert cli_main(["run", str(src), "--out", str(new), "--max-chars", "300", "--no-header"]) == 0
        plan = stages.stage_diff(new, old)
        assert plan["summary"]["update"] == plan["summary"]["new_chunks"]
        assert plan["update"][0]["diff"][0].startswith("(본문은 같고")

    def test_unit_digest_ignores_non_payload_fields(self):
        a = {"embed_text": "x", "text": "x", "title": "t", "locator": "l", "year": 1, "file_path": "a", "article_sha256": "1"}
        b = dict(a, file_path="b", article_sha256="2")
        assert update.content_digest(a) == update.content_digest(b)
        assert update.content_digest(a) != update.content_digest(dict(a, text="y"))

    @pytest.mark.parametrize("field,value", [("locator", "제1조 제1항"), ("year", 2027), ("title", "새이름"), ("work_type", ["HOT"]), ("url", "http://x")])
    def test_each_payload_field_triggers_update(self, tmp_path, field, value):
        old = self.build(tmp_path, "old", SAMPLE)
        new = self.build(tmp_path, "new", SAMPLE)
        recs = stages.load_chunks(new)
        recs[0][field] = value
        plan = update.plan_update(stages.load_chunks(old), recs)
        assert plan["summary"]["update"] == 1 and plan["update"][0]["article_label"] == recs[0]["article_label"]

    def test_cli_run_with_prev_and_diff_command(self, tmp_path, capsys):
        old = self.build(tmp_path, "old", SAMPLE)
        src = make_hwpx(tmp_path / "n.hwpx", amended(edit2=True, add7=True))
        new = tmp_path / "new"
        assert cli_main(["run", str(src), "--out", str(new), "--max-chars", "300", "--prev", str(old)]) == 0
        assert "갱신 계획: 추가 1 / 변경 1 / 삭제 0" in capsys.readouterr().out
        assert cli_main(["diff", "--out", str(new), "--prev", str(old)]) == 0
        assert cli_main(["diff", "--out", str(new), "--prev", str(tmp_path / "none")]) == 2


# --- 실제 파일 (선택) ----------------------------------------------------------------------


@pytest.mark.skipif("LAW_HWPX" not in os.environ, reason="LAW_HWPX 환경변수가 없어 건너뜀")
class TestRealFile:
    def test_real_law(self, tmp_path):
        out = tmp_path / "real"
        argv = ["run", os.environ["LAW_HWPX"], "--out", str(out)]
        if "LAW_TOC" in os.environ:
            argv += ["--toc", os.environ["LAW_TOC"]]
        assert cli_main(argv) == 0
        rep = json.loads((out / stages.F_VALIDATION).read_text(encoding="utf-8"))
        assert rep["ok"]
        assert rep["summary"]["articles"] > 600
