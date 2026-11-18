"""계획 단계 측정 묶음(scripts/plan_eval) 시험. torch·transformers 없이 돈다 (모델은 가짜)."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import pytest

from app.agent import plan_brief as pb
from app.agent.text_rules import EQUIPMENT, MATERIALS
from scripts.plan_eval import compare, harness, run_quality, run_speed
from scripts.plan_eval.harness import GenResult, KINDS, load_cases


@pytest.fixture(scope="module")
def cases() -> list[harness.Case]:
    return load_cases()


def _case(cases: list[harness.Case], case_id: str) -> harness.Case:
    return next(c for c in cases if c.id == case_id)


def _gen(text: str, **kw) -> GenResult:
    return GenResult(text=text, prompt_tokens=kw.get("prompt_tokens", 100), gen_tokens=kw.get("gen_tokens", 40),
                     seconds=kw.get("seconds", 1.0), hit_max=kw.get("hit_max", False))


def _oracle(case: harness.Case) -> str:
    """정답 그대로 내는 가짜 모델의 출력."""
    return json.dumps({"summary": case.text[:40], "equipment": list(case.gold_equipment), "materials": list(case.gold_materials)},
                      ensure_ascii=False)


def _oracle_fn(by_user: dict[str, harness.Case]):
    def fn(system: str, user: str) -> GenResult:
        return _gen(_oracle(by_user[user]))
    return fn


# ── 평가 문장 ──


def test_eval_set_has_forty_cases_covering_every_kind(cases):
    assert len(cases) == 40
    assert len({c.id for c in cases}) == 40
    assert {c.kind for c in cases} == set(KINDS)


def test_eval_set_does_not_use_prompt_example_or_demo_sentence(cases):
    # 프롬프트 예시와 계획서 9장 데모 문장은 평가에 쓰지 않는다 (예시를 베끼는지가 아니라 처음 보는 문장을 재야 한다)
    banned = ("3호기 펌프 베어링 교체", "R-201 반응기 하부 배관 플랜지 볼트 교체", "T-105 탱크에 톨루엔")
    for c in cases:
        assert not any(b in c.text for b in banned), c.id


def test_load_cases_rejects_gold_missing_from_text(tmp_path):
    bad = {"id": "x1", "kind": "plain", "text": "바닥 청소만 함.", "gold": {"equipment": ["그라인더"], "materials": [], "excluded": []}}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="x1.*그라인더"):
        load_cases(path)


def test_load_cases_rejects_negated_gold_and_unnegated_excluded(tmp_path):
    rows = [
        {"id": "a", "kind": "negation", "text": "크레인은 안 씀.", "gold": {"equipment": ["크레인"], "materials": [], "excluded": []}},
        {"id": "b", "kind": "negation", "text": "토치를 씀.", "gold": {"equipment": [], "materials": [], "excluded": ["토치"]}},
    ]
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_cases(path)
    assert "a: 정답 '크레인'" in str(exc.value)
    assert "b: 제외 '토치'" in str(exc.value)


def test_load_cases_rejects_duplicate_id_and_unknown_kind(tmp_path):
    row = {"id": "a", "kind": "weird", "text": "바닥 청소.", "gold": {}}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_cases(path)
    assert "id 중복" in str(exc.value) and "weird" in str(exc.value)


# ── 서버와 같은 프롬프트 ──


def test_messages_are_the_ones_the_server_sends(cases):
    case = _case(cases, "d04")
    system, user = harness.build_messages(case)
    assert system == pb.build_brief_prompt()
    assert user.startswith(f"작업 내용: {case.text}")
    assert "장소: 드럼 보관 구역" in user
    assert "주변 물질: 톨루엔" in user  # 슬롯 추출이 찾은 물질이 프롬프트에 들어간다 (실서버와 같다)
    assert "인원: 2명" in user
    _, user_p02 = harness.build_messages(_case(cases, "p02"))
    assert "투입 장비: 용접기" in user_p02 and "주변 물질: 아세톤" in user_p02 and "종료 시간: 15:00" in user_p02
    assert "화기 종류: 용접" in user_p02


def test_slot_terms_skip_negated_equipment(cases):
    case = _case(cases, "n01")
    equipment, materials = harness.slot_terms(case.text)
    assert equipment == ["체인블록"] and "크레인" not in equipment
    assert "메탄올" in materials


# ── 한 건 채점 ──


def test_perfect_output_matches_gold(cases):
    case = _case(cases, "p02")
    rec = harness.score_case(case, _oracle(case), _gen(_oracle(case)))
    assert rec["parsed"] and rec["schema_ok"] and rec["summary_ok"]
    assert rec["llm_equipment"] == ["용접기"] and rec["llm_materials"] == ["아세톤"]
    assert rec["final_equipment"] == rec["gold_equipment"]
    assert rec["eq_dropped"] == rec["mat_dropped"] == 0


def test_fenced_json_and_think_block_are_still_parsed(cases):
    case = _case(cases, "p02")
    raw = "<think>\n\n</think>\n```json\n" + _oracle(case) + "\n```"
    assert harness.score_case(case, raw, _gen(raw))["parsed"]


def test_garbage_output_is_unparsed_and_falls_back_to_slots(cases):
    case = _case(cases, "p02")
    rec = harness.score_case(case, "알겠습니다. 작업을 정리하겠습니다.", _gen("x"))
    assert rec["parsed"] is False and rec["schema_ok"] is False
    assert rec["summary_reason"] == "no_json" and rec["summary_ok"] is False
    assert rec["llm_equipment"] == [] and rec["final_equipment"] == ["용접기"]  # 슬롯만 남는다


def test_hallucinated_and_negated_candidates_are_dropped_by_server_rules(cases):
    case = _case(cases, "n01")  # 크레인은 부정, 용접기는 원문에 없음
    raw = json.dumps({"summary": "3호기 펌프 분해 정비", "equipment": ["크레인", "체인블록", "용접기"], "materials": ["메탄올", "염산"]}, ensure_ascii=False)
    rec = harness.score_case(case, raw, _gen(raw))
    assert rec["llm_equipment"] == ["체인블록"] and rec["eq_dropped"] == 2
    assert rec["llm_materials"] == ["메탄올"] and rec["mat_dropped"] == 1
    assert rec["raw_eq"] == 3 and rec["raw_mat"] == 2


def test_part_and_equipment_words_are_not_materials(cases):
    case = _case(cases, "d01")
    raw = json.dumps({"summary": "탱크 플랜지 볼트 교체", "equipment": [], "materials": ["T-105", "플랜지", "볼트", "베어링"]}, ensure_ascii=False)
    rec = harness.score_case(case, raw, _gen(raw))
    assert rec["llm_materials"] == [] and rec["mat_dropped"] == 4


def test_summary_reasons_follow_clean_summary(cases):
    case = _case(cases, "p01")
    for summary, reason in (("", "empty"), ("점검", "short"), ("점검함", "short"), ("점검작업", None), ("보일러 9호 점검", "number"), ("보일러 연소실 점검 및 버너 팁 교체", None)):
        raw = json.dumps({"summary": summary, "equipment": [], "materials": []}, ensure_ascii=False)
        rec = harness.score_case(case, raw, _gen(raw))
        assert rec["summary_reason"] == reason, summary
        assert rec["summary_ok"] is (reason is None)


def test_summary_reason_is_none_exactly_when_clean_summary_accepts():
    for value in ("", "  ", None, 5, "짧", "보일러 점검 작업", "보일러 3호 점검"):
        text = "보일러 점검 작업"
        assert (harness.summary_reason(value, text) is None) == (pb.clean_summary(value, text) is not None), repr(value)


def test_schema_flags_extra_keys_and_wrong_types(cases):
    case = _case(cases, "p02")
    extra = json.dumps({"summary": "배관 용접 보수", "equipment": [], "materials": [], "work_types": ["화기"]}, ensure_ascii=False)
    rec = harness.score_case(case, extra, _gen(extra))
    assert rec["schema_ok"] is False and rec["extra_keys"] == ["work_types"]
    wrong = json.dumps({"summary": "배관 용접 보수", "equipment": "용접기", "materials": []}, ensure_ascii=False)
    assert harness.score_case(case, wrong, _gen(wrong))["schema_ok"] is False


def test_summary_truncation_is_flagged(cases):
    case = _case(cases, "p02")
    long_summary = "사무동 옥상 배관 용접 보수 작업을 진행하며 불티 주의와 아세톤 통 근접 여부를 함께 확인하고 작업 종료 후 정리정돈까지 마치는 작업"
    raw = json.dumps({"summary": long_summary, "equipment": [], "materials": []}, ensure_ascii=False)
    rec = harness.score_case(case, raw, _gen(raw))
    assert rec["summary_truncated"] is True and len(long_summary) > pb.BRIEF_SUMMARY_CHARS


def test_summary_support_flags_words_not_in_text():
    ratio, missing = harness.summary_support("보일러 연소실 교체함", "2호 보일러 연소실 점검. 버너 팁 교체함.")
    assert ratio == 1.0 and missing == []
    ratio, missing = harness.summary_support("보일러 고압가스 폭발 점검", "보일러 점검")
    assert missing == ["고압가스", "폭발"] and ratio == pytest.approx(0.5)
    assert harness.summary_support("", "아무 글")[0] == 1.0
    # 어미가 달라도(점검함 ↔ 점검했다) 앞부분이 원문에 있으면 근거 있음
    assert harness.summary_support("배관 점검함", "배관을 점검했다") == (1.0, [])


def test_match_counts_strict_vs_lenient():
    c = harness.match_counts(["에어그라인더", "용접기"], ["그라인더", "용접기"])
    assert c["strict_tp"] == 1  # 용접기만 정확히 같다
    assert c["lenient_pred_tp"] == 2 and c["lenient_gold_tp"] == 2  # 에어그라인더는 그라인더를 품는다
    assert harness.match_counts([], [])["n_pred"] == 0
    lopsided = harness.match_counts(["용접기", "토치", "사다리"], ["용접기"])  # 예측이 정답보다 많을 때 재현율은 정답 쪽 기준
    assert (lopsided["n_pred"], lopsided["n_gold"], lopsided["lenient_pred_tp"], lopsided["lenient_gold_tp"]) == (3, 1, 1, 1)


# ── 집계 ──


def _exact_slots(cases):
    """슬롯 추출을 '사전에 있는 정답만'으로 고정한다 (서버 text_rules의 부분 문자열 문제와 상관없이 하니스 계산만 본다)."""
    by_text = {c.text: c for c in cases}

    def fn(text: str):
        c = by_text[text]
        eq = [g for g in harness._canon(c.gold_equipment, pb._EQUIPMENT_KEYS) if g in EQUIPMENT]
        mat = [g for g in harness._canon(c.gold_materials, pb._MATERIAL_KEYS) if g in MATERIALS]
        return eq, mat

    return fn


def test_oracle_model_scores_perfect_and_beats_rule_only(cases, monkeypatch):
    monkeypatch.setattr(harness, "slot_terms", _exact_slots(cases))
    by_user = {harness.build_messages(c)[1]: c for c in cases}
    records, summary = harness.evaluate(cases, _oracle_fn(by_user), model="oracle/x", dtype="auto")
    assert summary["n"] == 40
    assert summary["format"]["json_parse_rate"] == 1.0 and summary["format"]["schema_rate"] == 1.0
    assert summary["items_all"]["llm"]["strict"]["f1"] == 1.0
    assert summary["items_all"]["final"]["strict"]["f1"] == 1.0
    # 사전 밖 물질·장비가 있어서 규칙만으로는 다 못 맞춘다 (모델이 더해야 할 몫)
    assert summary["items_all"]["rule"]["lenient"]["recall"] < 1.0
    assert summary["by_kind"]["out_of_dict"]["items_all"]["rule"]["lenient"]["recall"] < summary["by_kind"]["out_of_dict"]["items_all"]["final"]["lenient"]["recall"]
    for domain in ("equipment", "materials"):  # 영역별 표도 규칙만(rule)을 모델(llm)과 따로 센다
        assert summary["items"][domain]["rule"]["lenient"]["recall"] < summary["items"][domain]["llm"]["lenient"]["recall"]
    assert summary["by_kind"]["none"]["raw_candidate_cases"] == 0
    assert summary["by_kind"]["plain"]["kept_candidate_cases"] == 8  # 평범한 문장 8건 모두에서 검증을 통과한 후보가 남는다
    assert summary["candidates"]["equipment_drop_rate"] == 0
    assert summary["rule_extra_cases"] == []


def test_rule_extra_reports_slot_items_that_are_not_in_gold(cases, monkeypatch):
    # 슬롯 사전이 '메탄올'에서 '메탄'도 잡는 경우를 흉내 낸다 (부분 문자열 일치)
    case = _case(cases, "n01")
    monkeypatch.setattr(harness, "slot_terms", lambda text: (["체인블록"], ["메탄", "메탄올"]))
    raw = _oracle(case)
    rec = harness.score_case(case, raw, _gen(raw))
    assert rec["rule_extra"] == ["메탄"]
    assert rec["final_materials"] == ["메탄", "메탄올"]
    summary = harness.summarize([rec])
    assert summary["rule_extra_cases"] == [{"id": "n01", "items": ["메탄"]}]
    assert summary["items_all"]["llm"]["strict"]["f1"] == 1.0  # 모델이 낸 것만 보면 맞다
    assert summary["items_all"]["final"]["strict"]["precision"] < 1.0  # 최종 값에는 슬롯 쪽 군더더기가 섞인다
    assert summary["items_all"]["final"]["lenient"]["precision"] == 1.0  # 느슨한 비교는 이것을 못 잡는다 (엄격 비교를 같이 보는 이유)


def test_model_that_invents_items_shows_up_in_none_cases(cases):
    def liar(system: str, user: str) -> GenResult:
        return _gen(json.dumps({"summary": "정리정돈 작업", "equipment": ["그라인더"], "materials": ["톨루엔"]}, ensure_ascii=False))

    _, summary = harness.evaluate([c for c in cases if c.kind == "none"], liar)
    none = summary["by_kind"]["none"]
    assert none["raw_candidate_cases"] == 4  # 모델은 4건 모두에서 후보를 냈지만
    assert none["kept_candidate_cases"] == 0  # 원문에 없어서 검증이 전부 버렸다
    assert summary["candidates"]["equipment_drop_rate"] == 1.0


def test_unparsable_model_has_zero_parse_rate_and_summarizes_without_error(cases):
    _, summary = harness.evaluate(cases[:5], lambda s, u: _gen("죄송합니다."))
    assert summary["format"]["json_parse_rate"] == 0.0
    assert summary["summary"]["usable_rate"] == 0.0
    assert summary["summary"]["reasons"]["no_json"] == 5
    assert summary["summary"]["truncated_rate"] is None  # 파싱된 것이 없어 비율을 못 낸다


def test_token_stats_and_cut_rate(cases):
    gens = iter([_gen("{}", gen_tokens=10, hit_max=False), _gen("{}", gen_tokens=160, hit_max=True)])
    _, summary = harness.evaluate(cases[:2], lambda s, u: next(gens))
    assert summary["format"]["max_tokens_cut_rate"] == 0.5
    assert summary["tokens"]["gen_mean"] == 85 and summary["tokens"]["gen_max"] == 160


# ── 저장 ──


def test_write_outputs_makes_three_files_and_excel_friendly_review(cases, tmp_path):
    by_user = {harness.build_messages(c)[1]: c for c in cases}
    records, summary = harness.evaluate(cases, _oracle_fn(by_user), model="m/x", dtype="auto")
    paths = harness.write_outputs(tmp_path / "o", "tag", records, summary)
    assert {p.name for p in paths.values()} == {"tag.records.jsonl", "tag.summary.json", "tag.review.csv"}
    assert len(paths["records"].read_text(encoding="utf-8").splitlines()) == 40
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["n"] == 40
    raw = paths["review"].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: 엑셀에서 한글이 깨지지 않게
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    assert rows[0] == list(harness.REVIEW_COLUMNS) and len(rows) == 41


# ── 실행 스크립트 (가짜 모델) ──


class FakeRunner:
    load_seconds = 0.5

    def __init__(self, by_user):
        self.by_user = by_user
        self.calls: list[int | None] = []

    def generate(self, system: str, user: str, *, max_new_tokens: int | None = None) -> GenResult:
        self.calls.append(max_new_tokens)
        if max_new_tokens == 1:
            return _gen("{", prompt_tokens=300, gen_tokens=1, seconds=2.0)
        return _gen(_oracle(self.by_user[user]), prompt_tokens=300, gen_tokens=51, seconds=7.0)

    def __call__(self, system: str, user: str) -> GenResult:
        return self.generate(system, user)


def test_run_quality_writes_files_and_prints_headline(cases, tmp_path, capsys):
    by_user = {harness.build_messages(c)[1]: c for c in cases}
    seen = {}

    def factory(model, *, dtype, device):
        seen.update(model=model, dtype=dtype, device=device)
        runner = FakeRunner(by_user)
        runner.revision = "abc123def4567890"
        return runner

    code = run_quality.run(["--model", "Qwen/Qwen3-0.6B", "--dtype", "float16", "--out", str(tmp_path), "--limit", "10"], runner_factory=factory)
    assert code == 0 and seen == {"model": "Qwen/Qwen3-0.6B", "dtype": "float16", "device": "auto"}
    tag = "Qwen__Qwen3-0.6B__float16"
    assert (tmp_path / f"{tag}.summary.json").exists()
    saved = json.loads((tmp_path / f"{tag}.summary.json").read_text(encoding="utf-8"))
    assert saved["n"] == 10 and saved["revision"] == "abc123def4567890"
    assert "JSON 파싱 100%" in capsys.readouterr().out


def test_run_speed_math_and_output(cases, tmp_path, capsys):
    by_user = {harness.build_messages(c)[1]: c for c in cases}
    runners: list[FakeRunner] = []

    def factory(model, *, dtype, device):
        runners.append(FakeRunner(by_user))
        return runners[-1]

    code = run_speed.run(["--model", "Qwen/Qwen3-1.7B", "--dtypes", "auto,float32", "--n", "3", "--out", str(tmp_path)], runner_factory=factory)
    assert code == 0 and len(runners) == 2
    data = json.loads((tmp_path / "speed__Qwen__Qwen3-1.7B.json").read_text(encoding="utf-8"))
    assert len(data["cases"]) == 3 and set(data["dtypes"]) == {"auto", "float32"}
    s = data["dtypes"]["auto"]["summary"]
    # prefill 2초, 전체 7초, 50토큰 → 생성 5초 = 10 tok/s
    assert s["prefill_seconds"] == 2.0 and s["decode_seconds"] == 5.0 and s["tokens_per_second"] == pytest.approx(10.0)
    # 워밍업 1회 + (프롬프트 읽기 1회 + 전체 1회) x 3문장
    assert len(runners[0].calls) == 1 + 2 * 3


def _row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {label}"))


def test_compare_rows_point_at_the_right_numbers(cases):
    _, base = harness.evaluate(cases[:5], lambda s, u: _gen("죄송합니다."), model="m/a", dtype="auto")
    base["format"].update(json_parse_rate=0.9, schema_rate=0.5)
    base["items_all"]["final"]["strict"]["f1"] = 0.4
    base["items_all"]["final"]["lenient"]["f1"] = 0.8
    text = compare.render([base])
    assert _row(text, "JSON 파싱 성공").endswith("| 90% |")
    assert _row(text, "키·자료형 맞음").endswith("| 50% |")
    assert _row(text, "공구·물질 F1 (최종, 엄격").endswith("| 40% |")


def test_compare_renders_table_for_two_runs_even_with_missing_values(cases):
    by_user = {harness.build_messages(c)[1]: c for c in cases}
    _, good = harness.evaluate(cases, _oracle_fn(by_user), model="Qwen/Qwen3-0.6B", dtype="auto")
    _, bad = harness.evaluate(cases, lambda s, u: _gen("죄송합니다."), model="Qwen/Qwen3-1.7B", dtype="auto")
    text = compare.render([good, bad])
    assert "Qwen3-0.6B (auto)" in text and "Qwen3-1.7B (auto)" in text
    assert "JSON 파싱 성공(첫 시도)" in text and "100%" in text and "0%" in text
    assert _row(compare.render([dict(good, revision="abc123def4567890")]), "모델 revision").endswith("| abc123def4 |")
    assert "모델 없이 슬롯 사전만 쓴 기준" in text
    with pytest.raises(ValueError):
        compare.render([])


# ── 모델 호출부 (가짜 torch·transformers) ──


class _Seq:
    """토큰 목록 하나 (텐서 흉내). 인덱싱·자르기·shape만 쓴다."""

    def __init__(self, items):
        self.items = list(items)

    @property
    def shape(self):
        return (len(self.items),)

    def __getitem__(self, i):
        return _Seq(self.items[i]) if isinstance(i, slice) else self.items[i]


class _Batch(dict):
    def to(self, device):
        self.device_arg = device
        return self


class _FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, **kwargs})
        return _Batch(input_ids=_OneRow([1, 2, 3, 4]))

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return " " + "".join(chr(0xAC00 + int(i)) for i in ids.items if i != 99) + " "


class _OneRow(list):
    """input_ids 한 줄. shape[-1]이 토큰 수."""

    @property
    def shape(self):
        return (1, len(self))


class _FakeModel:
    device = "cpu"

    config = types.SimpleNamespace(_commit_hash="abc123def4567890")

    def __init__(self, tail):
        self.tail = tail
        self.generate_calls: list[dict] = []
        self.eval_called = False
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_calls.append({k: v for k, v in kwargs.items() if k != "input_ids"})
        n = min(kwargs["max_new_tokens"], len(self.tail))
        return [_Seq([1, 2, 3, 4, *self.tail[:n]])]


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def fake_stack(monkeypatch):
    holder = types.SimpleNamespace(tokenizer=_FakeTokenizer(), model=_FakeModel([5, 6, 7, 8, 99]), dtype_calls=[], reject_dtype_kw=False)

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32, torch.bfloat16, torch.float16 = "F32", "BF16", "F16"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kw):
            holder.tokenizer_model = model_id
            return holder.tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kw):
            holder.dtype_calls.append(kw)
            if holder.reject_dtype_kw and "dtype" in kw:
                raise TypeError("unexpected keyword argument 'dtype'")
            return holder.model

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer, transformers.AutoModelForCausalLM = AutoTokenizer, AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return holder


def test_runner_generates_like_the_server(fake_stack):
    from scripts.plan_eval import model_runner

    runner = model_runner.load_runner("Qwen/Qwen3-0.6B", dtype="float32", device="auto")
    assert fake_stack.model.eval_called and fake_stack.model.moved_to == "cpu"
    assert fake_stack.dtype_calls == [{"dtype": "F32"}]
    res = runner.generate("시스템", "사용자")
    call = fake_stack.tokenizer.template_calls[0]
    assert call["messages"] == [{"role": "system", "content": "시스템"}, {"role": "user", "content": "사용자"}]
    assert call["add_generation_prompt"] is True and call["enable_thinking"] is False  # 서버와 같은 템플릿 인자
    assert fake_stack.model.generate_calls[0] == {"max_new_tokens": pb.BRIEF_MAX_TOKENS, "do_sample": False}
    assert runner.revision == "abc123def4567890"
    assert res.prompt_tokens == 4 and res.gen_tokens == 5
    assert res.hit_max is False  # eos(99)로 끝났다
    assert res.text == "".join(chr(0xAC00 + i) for i in (5, 6, 7, 8))  # 앞뒤 공백 제거, eos는 건너뜀


def test_runner_marks_cut_when_limit_reached_without_eos(fake_stack):
    from scripts.plan_eval import model_runner

    fake_stack.model.tail = [5, 6, 7, 8, 9, 10]  # eos 없이 계속
    runner = model_runner.load_runner("m", dtype="auto")
    res = runner.generate("s", "u", max_new_tokens=3)
    assert res.gen_tokens == 3 and res.hit_max is True
    assert fake_stack.dtype_calls == [{"dtype": "auto"}]
    # 한도보다 일찍 끝나면(eos가 없어도) 잘린 것이 아니다 — 서버의 stop_reason 판정과 같다
    fake_stack.model.tail = [5, 6]
    assert runner.generate("s", "u", max_new_tokens=3).hit_max is False


def test_runner_eos_exactly_at_the_limit_is_not_a_cut(fake_stack):
    from scripts.plan_eval import model_runner

    fake_stack.model.tail = [5, 6, 99]  # 한도 3토큰에 딱 맞게 eos로 끝남
    runner = model_runner.load_runner("m", dtype="auto")
    res = runner.generate("s", "u", max_new_tokens=3)
    assert res.gen_tokens == 3 and res.hit_max is False


def test_runner_falls_back_to_torch_dtype_for_older_transformers(fake_stack):
    from scripts.plan_eval import model_runner

    fake_stack.reject_dtype_kw = True
    model_runner.load_runner("m", dtype="bfloat16")
    assert fake_stack.dtype_calls == [{"dtype": "BF16"}, {"torch_dtype": "BF16"}]


def test_runner_rejects_unknown_dtype(fake_stack):
    from scripts.plan_eval import model_runner

    with pytest.raises(ValueError):
        model_runner.load_runner("m", dtype="int4")


def test_harness_import_does_not_need_torch():
    code = "import sys; sys.modules['torch']=None; import scripts.plan_eval.harness, scripts.plan_eval.run_quality, scripts.plan_eval.run_speed, scripts.plan_eval.compare"
    import subprocess

    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
