"""평가 도구 확장(어댑터·짧은 프롬프트·여러 줄 문장·compare 경로 풀기) 시험. torch·transformers 없이 돈다 (모델은 가짜).

실제 모델에 어댑터를 붙이는 시험(load_runner)은 tests/test_plan_train_lora.py 에 있다 (아주 작은 모델이 거기 있다).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.agent import plan_brief as pb
from app.agent.evidence_collector.planner import build_planner_input
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, build_adapter_input
from app.agent.slots import merge_slots
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms
from scripts import make_plan_train as mpt
from scripts.plan_eval import compare, harness, model_runner, run_quality
from scripts.plan_eval.harness import GenResult, load_cases

MULTILINE = Path(harness.__file__).with_name("eval_set_multiline.jsonl")


@pytest.fixture(scope="module")
def cases() -> list[harness.Case]:
    return load_cases()


@pytest.fixture(scope="module")
def multi() -> list[harness.Case]:
    return load_cases(MULTILINE)


def _gen(text: str) -> GenResult:
    return GenResult(text=text, prompt_tokens=100, gen_tokens=40, seconds=1.0, hit_max=False)


def _oracle(case: harness.Case) -> str:
    return json.dumps({"summary": case.text.replace("\n", " ")[:40], "equipment": list(case.gold_equipment), "materials": list(case.gold_materials)},
                      ensure_ascii=False)


class FakeRunner:
    load_seconds = 0.1

    def __init__(self, by_user, **attrs):
        self.by_user = by_user
        self.__dict__.update(attrs)

    def __call__(self, system: str, user: str) -> GenResult:
        return _gen(_oracle(self.by_user[user]))


# ---------------------------------------------------------------- 프롬프트 고르기


def test_default_prompt_is_unchanged_long_one(cases):
    for case in cases:
        ctx = harness.work_context(case)
        assert harness.build_messages(case) == (pb.build_brief_prompt(), build_planner_input(ctx))
        assert harness.build_messages(case, "long") == harness.build_messages(case)


def test_short_prompt_is_what_the_server_sends_with_adapter(cases):
    for case in cases[:10]:
        system, user = harness.build_messages(case, "short")
        assert system == ADAPTER_SYSTEM_PROMPT
        assert user == build_adapter_input(harness.work_context(case))
        assert "장소" not in user and "인원" not in user  # 짧은 입력은 작업 내용·주변 물질·투입 장비만


def test_unknown_prompt_is_rejected(cases):
    with pytest.raises(ValueError, match="prompt"):
        harness.build_messages(cases[0], "medium")


def test_one_line_cases_keep_the_old_slot_terms(cases):
    """40문장 기준선이 바뀌면 이전 측정과 비교할 수 없다: 한 줄 문장의 슬롯은 예전 계산(find_terms)과 똑같다."""
    for case in cases:
        assert "\n" not in case.text
        assert harness.slot_terms(case.text) == (find_terms(case.text, EQUIPMENT)[0], find_terms(case.text, MATERIALS)[0])


# ---------------------------------------------------------------- 여러 줄 문장


def test_multiline_slot_terms_follow_the_server_merge(multi):
    for case in multi:
        slots = None
        for line in case.text.split("\n"):
            slots = merge_slots(slots, line, today=date(2026, 1, 15))
        assert harness.slot_terms(case.text) == (list(slots.equipment), list(slots.nearby_materials)), case.id


def test_multiline_differs_from_whole_text_when_a_later_line_withdraws_something(multi):
    """왜 줄마다 잇는가: 뒤 줄이 앞에서 말한 것을 거두면 문장 전체에 한 번에 find_terms 를 부르는 것은 서버와 달라진다."""
    by_id = {c.id: c for c in multi}
    assert harness.slot_terms(by_id["ml09"].text) == (["용접기"], [])
    assert find_terms(by_id["ml09"].text, EQUIPMENT)[0] == ["용접기", "토치"]  # 한 번에 부르면 거둔 토치가 남는다
    assert harness.slot_terms(by_id["ml02"].text) == ([], [])
    assert "그라인더" in find_terms(by_id["ml02"].text, EQUIPMENT)[0]


def test_multiline_set_is_well_formed(multi):
    assert len(multi) == 12
    assert len({c.id for c in multi}) == 12
    assert all(c.text.count("\n") >= 1 for c in multi)
    assert {c.kind for c in multi} >= {"plain", "negation", "conditional", "out_of_dict", "multi", "none", "distractor"}
    assert sum(1 for c in multi if not (c.gold_equipment or c.gold_materials)) >= 3  # 아무것도 안 뽑아야 하는 문장도 있다


def test_multiline_gold_is_reachable_through_server_validation(multi):
    """정답이 서버 검증(clean_*)에서 탈락하면 모델이 아무리 잘해도 못 맞춘다."""
    for case in multi:
        equipment, _ = pb.clean_equipment(list(case.gold_equipment), case.text)
        materials, _ = pb.clean_materials(list(case.gold_materials), case.text)
        assert set(equipment) == set(pb._canonical(list(case.gold_equipment), pb._EQUIPMENT_KEYS)), case.id  # noqa: SLF001
        assert set(materials) == set(pb._canonical(list(case.gold_materials), pb._MATERIAL_KEYS)), case.id  # noqa: SLF001


def test_multiline_set_does_not_overlap_the_40_eval_sentences(cases, multi):
    first = {" ".join(c.text.split()) for c in cases}
    assert not any(" ".join(c.text.split()) in first for c in multi)
    sentences = mpt.EvalGuard.load(harness.DEFAULT_EVAL_SET).sentences
    assert not any(mpt._sentences(c.text) and set(mpt._sentences(c.text)) & sentences for c in multi)  # noqa: SLF001


def test_generator_keeps_clear_of_both_eval_sets(multi, tmp_path):
    """학습 데이터가 여러 줄 평가 문장과 겹치지 않는다 (기본으로 둘 다 피한다)."""
    guard = mpt.EvalGuard.load()
    assert guard.texts and {c.text for c in multi} <= set(guard.texts)
    out = tmp_path / "d"
    assert mpt.main(["--out", str(out), "--train", "300", "--valid", "30"]) == 0
    eval_text = {" ".join(c.text.split()) for c in multi}
    for name in ("train.jsonl", "valid.jsonl"):
        for line in (out / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert " ".join("\n".join(row["utterances"]).split()) not in eval_text
            assert not set(mpt._sentences("\n".join(row["utterances"]))) & guard.sentences  # noqa: SLF001
    assert {"수산화나트륨", "과산화수소", "이소옥탄", "이동식 펌프"} <= guard.items  # 정답에 든 사전 밖 이름은 학습 어휘에서 빠진다


def test_vocab_check_also_looks_inside_eval_text_not_only_gold(monkeypatch):
    """ml08 의 '생석회'는 쓰지 않는다고 거둔 말이라 정답에도 제외에도 없다. 정답 목록만 보는 검사는 이것을 놓친다."""
    guard = mpt.EvalGuard.load()
    assert "생석회" not in guard.items and any("생석회" in t for t in guard.texts)
    mpt.check_vocab(guard)  # 지금 어휘는 통과
    monkeypatch.setattr(mpt, "OOD_MATERIALS_TRAIN", [*mpt.OOD_MATERIALS_TRAIN, "생석회"])
    with pytest.raises(AssertionError, match="평가 문장 본문"):
        mpt.check_vocab(guard)


def test_train_vocab_never_uses_multiline_eval_items(multi):
    guard = mpt.EvalGuard.load()
    mpt.check_vocab(guard)  # 사전 밖 이름이 두 평가 세트의 정답과 같거나 겹치면 AssertionError


# ---------------------------------------------------------------- run_quality 이름·기록


def test_tag_for_keeps_old_names_and_adds_prompt_and_adapter():
    assert run_quality.tag_for("Qwen/Qwen3-0.6B", "float16") == "Qwen__Qwen3-0.6B__float16"
    assert run_quality.tag_for("Qwen/Qwen3-0.6B", "float16", "short") == "Qwen__Qwen3-0.6B__float16__short"
    assert run_quality.tag_for("Qwen/Qwen3-0.6B", "float16", "short", "adapters/plan_brief_0.6b") == "Qwen__Qwen3-0.6B__float16__short__lora-plan_brief_0.6b"
    assert run_quality.tag_for("Qwen/Qwen3-0.6B", "float16", "short", "adapters\\plan_brief_0.6b\\") == "Qwen__Qwen3-0.6B__float16__short__lora-plan_brief_0.6b"
    assert run_quality.tag_for("Qwen/Qwen3-0.6B", "auto", "long", "a/x") != run_quality.tag_for("Qwen/Qwen3-0.6B", "auto")


def _run(cases, tmp_path, argv, **runner_attrs):
    by_user = {}
    for prompt in ("long", "short"):
        by_user.update({harness.build_messages(c, prompt)[1]: c for c in cases})
    seen = {}

    def factory(model, **kw):
        seen.update(kw, model=model)
        return FakeRunner(by_user, **runner_attrs)

    code = run_quality.run(["--model", "Qwen/Qwen3-0.6B", "--dtype", "float32", "--out", str(tmp_path), "--limit", "6", *argv], runner_factory=factory)
    return code, seen


def test_without_adapter_the_factory_gets_no_adapter_argument(cases, tmp_path):
    code, seen = _run(cases, tmp_path, [])
    assert code == 0 and "adapter" not in seen
    summary = json.loads((tmp_path / "Qwen__Qwen3-0.6B__float32.summary.json").read_text(encoding="utf-8"))
    assert summary["prompt"] == "long" and summary["adapter"] == "" and "adapter_info" not in summary


def test_adapter_defaults_to_short_prompt_and_records_everything(cases, tmp_path, capsys):
    meta = {"base_model": "Qwen/Qwen3-0.6B", "base_revision": "aaaaaaaaaa1111", "data_sha256": "ff" * 32}
    code, seen = _run(cases, tmp_path, ["--adapter", "adapters/plan_brief_0.6b"], revision="aaaaaaaaaa1111", adapter_meta=meta)
    assert code == 0 and seen["adapter"] == "adapters/plan_brief_0.6b"
    tag = "Qwen__Qwen3-0.6B__float32__short__lora-plan_brief_0.6b"
    summary = json.loads((tmp_path / f"{tag}.summary.json").read_text(encoding="utf-8"))
    assert summary["prompt"] == "short" and summary["adapter"] == "plan_brief_0.6b"
    assert summary["adapter_info"]["path"] == "adapters/plan_brief_0.6b" and summary["adapter_info"]["revision_match"] is True
    assert summary["adapter_info"]["data_sha256"] == "ff" * 32
    assert "[주의]" not in capsys.readouterr().out


def test_revision_mismatch_is_flagged_not_hidden(cases, tmp_path, capsys):
    meta = {"base_revision": "aaaaaaaaaa1111"}
    _run(cases, tmp_path, ["--adapter", "ad/x"], revision="bbbbbbbbbb2222", adapter_meta=meta)
    summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text(encoding="utf-8"))
    assert summary["adapter_info"]["revision_match"] is False
    assert "다릅니다" in capsys.readouterr().out


def test_unknown_revision_is_recorded_as_unknown(cases, tmp_path):
    _run(cases, tmp_path, ["--adapter", "ad/x"], revision="", adapter_meta={"base_revision": "aaaa"})
    summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text(encoding="utf-8"))
    assert summary["adapter_info"]["revision_match"] is None


def test_short_prompt_without_adapter_is_allowed_for_ablation(cases, tmp_path):
    code, seen = _run(cases, tmp_path, ["--prompt", "short"])
    assert code == 0 and "adapter" not in seen
    assert (tmp_path / "Qwen__Qwen3-0.6B__float32__short.summary.json").exists()


def test_adapter_with_long_prompt_warns(cases, tmp_path, capsys):
    _run(cases, tmp_path, ["--adapter", "ad/x", "--prompt", "long"])
    assert "짧은 프롬프트로 학습" in capsys.readouterr().out


def test_prompts_reach_the_model_as_chosen(cases, tmp_path):
    systems = []

    def factory(model, **kw):
        def fn(system, user):
            systems.append(system)
            return _gen("죄송합니다.")
        return fn

    run_quality.run(["--model", "m/a", "--out", str(tmp_path), "--limit", "3", "--prompt", "short"], runner_factory=factory)
    assert set(systems) == {ADAPTER_SYSTEM_PROMPT}
    systems.clear()
    run_quality.run(["--model", "m/a", "--out", str(tmp_path), "--limit", "3"], runner_factory=factory)
    assert set(systems) == {pb.build_brief_prompt()}


def test_multiline_eval_set_runs_through_run_quality(multi, tmp_path):
    by_user = {harness.build_messages(c, "short")[1]: c for c in multi}
    code = run_quality.run(["--model", "m/a", "--out", str(tmp_path), "--eval-set", str(MULTILINE), "--prompt", "short"],
                           runner_factory=lambda model, **kw: FakeRunner(by_user))
    assert code == 0
    summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text(encoding="utf-8"))
    assert summary["n"] == 12 and summary["format"]["json_parse_rate"] == 1.0


# ---------------------------------------------------------------- 어댑터 점검 (모델 없이)


def _adapter_dir(tmp_path, *, config_base="Qwen/Qwen3-0.6B", meta=None):
    folder = tmp_path / "ad"
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": config_base, "peft_type": "LORA"}), encoding="utf-8")
    if meta is not None:
        (folder / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def test_check_adapter_rejects_folder_without_config(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="adapter_config.json"):
        model_runner.check_adapter(str(tmp_path / "empty"), "Qwen/Qwen3-0.6B")


def test_check_adapter_rejects_other_base_model_by_meta_and_by_config(tmp_path):
    folder = _adapter_dir(tmp_path, meta={"base_model": "Qwen/Qwen3-0.6B"})
    with pytest.raises(ValueError, match="Qwen3-0.6B 로 학습"):
        model_runner.check_adapter(str(folder), "Qwen/Qwen3-1.7B")
    other = tmp_path / "o"
    other.mkdir()
    folder2 = _adapter_dir(other, config_base="Qwen/Qwen3-1.7B")  # train_meta 없음 → adapter_config 로 판단
    with pytest.raises(ValueError, match="Qwen3-1.7B 로 학습"):
        model_runner.check_adapter(str(folder2), "Qwen/Qwen3-0.6B")


def test_check_adapter_returns_what_gets_recorded(tmp_path):
    meta = {"base_model": "Qwen/Qwen3-0.6B", "base_revision": "abc", "dtype": "float16", "lora": {"r": 16}, "train": {"epochs": 3},
            "versions": {"peft": "x"}, "data": {"train_sha256": "dd" * 32}, "history": [1, 2, 3], "generate_check": {"n": 1}}
    got = model_runner.check_adapter(str(_adapter_dir(tmp_path, meta=meta)), "Qwen/Qwen3-0.6B")
    assert got["base_revision"] == "abc" and got["data_sha256"] == "dd" * 32 and got["lora"] == {"r": 16}
    assert "history" not in got  # 요약 파일에는 기록할 것만 싣는다


# ---------------------------------------------------------------- compare


def _summary(cases, **extra):
    _, s = harness.evaluate(cases[:5], lambda sys_, user: _gen("죄송합니다."), model="Qwen/Qwen3-0.6B", dtype="float16", **extra)
    return s


def test_expand_paths_expands_globs_itself(tmp_path):
    for name in ("a.summary.json", "b.summary.json", "c.records.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    got = compare.expand_paths([str(tmp_path / "*.summary.json")])
    assert [Path(p).name for p in got] == ["a.summary.json", "b.summary.json"]
    plain = str(tmp_path / "c.records.jsonl")
    assert compare.expand_paths([plain, str(tmp_path / "a*"), plain]) == [plain, str(tmp_path / "a.summary.json")]  # 중복은 한 번


def test_expand_paths_errors_when_glob_matches_nothing(tmp_path):
    with pytest.raises(ValueError, match="맞는 파일이 없습니다"):
        compare.expand_paths([str(tmp_path / "*.summary.json")])


def test_compare_main_accepts_a_quoted_glob_like_powershell_passes_it(cases, tmp_path, capsys):
    for i, dtype in enumerate(("float16", "float32")):
        s = _summary(cases)
        s["dtype"] = dtype
        (tmp_path / f"{i}.summary.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    assert compare.main([str(tmp_path / "*.summary.json"), "--out", str(tmp_path / "c.md")]) == 0
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "Qwen3-0.6B (float16)" in text and "Qwen3-0.6B (float32)" in text


def test_compare_labels_show_prompt_and_adapter(cases):
    plain = _summary(cases)
    tuned = _summary(cases, prompt="short", adapter="plan_brief_0.6b")
    tuned["adapter_info"] = {"revision_match": True}
    ablation = _summary(cases, prompt="short")
    text = compare.render([plain, ablation, tuned])
    assert "Qwen3-0.6B (float16) |" in text
    assert "Qwen3-0.6B (float16, 짧은 프롬프트) |" in text
    assert "Qwen3-0.6B (float16, 짧은 프롬프트, +plan_brief_0.6b) |" in text
    row = next(line for line in text.splitlines() if line.startswith("| 프롬프트 |"))
    assert row == "| 프롬프트 | 긴 것(기본) | 짧음(어댑터용) | 짧음(어댑터용) |"
    row = next(line for line in text.splitlines() if line.startswith("| 어댑터"))
    assert row.endswith("| - | - | plan_brief_0.6b (일치) |")


def test_compare_flags_adapter_on_a_different_base_revision(cases):
    tuned = _summary(cases, prompt="short", adapter="ad")
    tuned["adapter_info"] = {"revision_match": False}
    assert "**불일치**" in compare.render([tuned])
    tuned["adapter_info"] = {}
    assert "(모름)" in compare.render([tuned])


def test_compare_still_renders_old_summary_files_without_new_keys(cases):
    old = _summary(cases)
    for key in ("prompt", "adapter"):
        old.pop(key)
    text = compare.render([old])
    assert "Qwen3-0.6B (float16) |" in text and "긴 것(기본)" in text


def test_source_files_have_no_carriage_returns():
    for path in (MULTILINE, Path(__file__)):
        assert b"\r" not in path.read_bytes(), path


def test_case_ids_never_collide_across_eval_sets():
    """기본 40문장과 여러 줄 12문장은 같은 요약 파일에 섞여 비교되므로 id 가 겹치면 안 된다 (T28 10번: m01~m04 가 겹쳤었다)."""
    ids = [c.id for name in ("eval_set.jsonl", "eval_set_multiline.jsonl") for c in harness.load_cases(Path("scripts/plan_eval") / name)]
    assert len(ids) == 52 and len(set(ids)) == 52, sorted(i for i in ids if ids.count(i) > 1)

