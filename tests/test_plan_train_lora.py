"""scripts/plan_train/train_lora.py 시험. 아주 작은 Qwen3 모델과 토크나이저를 시험 안에서 만들어 CPU로 실제 학습·저장·다시 읽기를 돈다.

torch·transformers·peft·tokenizers 가 없으면 건너뛴다 (PC 에 peft 가 없을 수 있다).
작은 모델은 허브에서 받지 않고 만든다 (Qwen3 구조, 채팅 템플릿은 Qwen3 처럼 enable_thinking=False 면 빈 <think> 블록을 붙인다).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT  # noqa: E402
from scripts import make_plan_train as mpt  # noqa: E402
from scripts.plan_train import train_lora as tl  # noqa: E402

# Qwen3 템플릿과 같은 모양 (system/user/assistant 를 <|im_start|>role\n...<|im_end|>\n 로, 생성 시작은 assistant\n, 생각을 끄면 빈 think 블록)
CHAT_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("plan-data")
    assert mpt.main(["--out", str(out), "--train", "24", "--valid", "8"]) == 0
    return out


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, data_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    texts = [m["content"] for name in ("train.jsonl", "valid.jsonl") for r in tl.load_rows(data_dir / name) for m in r["messages"]]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=700, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=CHAT_TEMPLATE)
    path = tmp_path_factory.mktemp("tiny-qwen3")
    fast.save_pretrained(path)
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id, bos_token_id=None,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def tokenizer(tiny):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny))


@pytest.fixture(scope="module")
def rows(data_dir):
    return tl.load_rows(data_dir / "train.jsonl")


def train_args(tiny, data_dir, out, *extra):
    return tl.parse_args([
        "--model", str(tiny), "--data", str(data_dir), "--out", str(out), "--device", "cpu", "--epochs", "3", "--batch-size", "4",
        "--lr", "5e-3", "--r", "4", "--alpha", "8", "--max-len", "2048", "--gen-check", "3", "--log-every", "1000", *extra,
    ])


# ---------------------------------------------------------------- 입력 만들기 (서버와 같은 호출)


def test_prompt_ids_equal_server_call(tokenizer, rows):
    for row in rows[:5]:
        # 서버(llm_transformers.generate_chat)의 호출: system+user, add_generation_prompt=True, return_dict, enable_thinking=False
        server = tokenizer.apply_chat_template(
            row["messages"][:2], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False,
        )["input_ids"][0].tolist()
        assert tl.prompt_ids(tokenizer, row["messages"]) == server


def test_prompt_ends_with_empty_think_block_and_has_no_assistant_text(tokenizer, rows):
    text = tokenizer.decode(tl.prompt_ids(tokenizer, rows[0]["messages"]))
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert rows[0]["messages"][2]["content"] not in text


def test_example_masks_prompt_and_ends_with_eos(tokenizer, rows):
    for row in rows[:8]:
        ex = tl.build_example(tokenizer, row["messages"], 4096)
        assert ex.labels[: ex.prompt_len] == [tl.IGNORE] * ex.prompt_len
        assert ex.labels[ex.prompt_len :] == ex.input_ids[ex.prompt_len :]
        assert ex.input_ids[-1] == tokenizer.eos_token_id and ex.labels[-1] == tokenizer.eos_token_id
        assert ex.input_ids[: ex.prompt_len] == tl.prompt_ids(tokenizer, row["messages"])
        answer = tokenizer.decode(ex.input_ids[ex.prompt_len : -1])
        assert answer == row["messages"][2]["content"]
        assert json.loads(answer)["summary"]


def test_example_rejects_bad_inputs(tokenizer, rows):
    messages = rows[0]["messages"]
    with pytest.raises(ValueError, match="넘으면|--max-len"):
        tl.build_example(tokenizer, messages, 20)  # 자르면 정답이 잘리므로 멈춘다
    with pytest.raises(ValueError, match="system, user, assistant"):
        tl.build_example(tokenizer, messages[:2], 4096)
    with pytest.raises(ValueError, match="system, user, assistant"):
        tl.build_example(tokenizer, [messages[1], messages[0], messages[2]], 4096)


def test_collate_pads_right_with_mask_and_ignore():
    a = tl.Example([1, 2, 3], [-100, 2, 3], 1)
    b = tl.Example([4, 5], [-100, 5], 1)
    out = tl.collate([a, b], pad_id=9)
    assert out["input_ids"] == [[1, 2, 3], [4, 5, 9]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"] == [[-100, 2, 3], [-100, 5, -100]]


def test_batches_are_deterministic_and_cover_everything():
    import random

    items = [tl.Example([i], [i], 0) for i in range(10)]
    a = tl.batches(items, 4, random.Random(1), shuffle=True)
    b = tl.batches(items, 4, random.Random(1), shuffle=True)
    assert a == b and sorted(e.input_ids[0] for chunk in a for e in chunk) == list(range(10))
    assert [len(c) for c in a] == [4, 4, 2]
    assert [e.input_ids[0] for e in tl.batches(items, 4, random.Random(1), shuffle=False)[0]] == [0, 1, 2, 3]


def test_dtype_rules():
    assert tl.resolve_dtype("auto", "cpu")[1] == "float32"
    with pytest.raises(ValueError, match="CPU"):
        tl.resolve_dtype("float16", "cpu")


# ---------------------------------------------------------------- 학습·저장·다시 읽기


@pytest.fixture(scope="module")
def trained(tiny, data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("adapter-out") / "plan_brief"
    meta = tl.train(train_args(tiny, data_dir, out), log=lambda *_: None)
    return out, meta


def test_training_saves_adapter_only_with_meta(trained, tiny):
    out, meta = trained
    names = {p.name for p in out.iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "train_meta.json"} <= names
    assert not any(n in names for n in ("model.safetensors", "pytorch_model.bin", "config.json"))  # 기본 모델은 저장하지 않는다 (머지 안 함)
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 4 and config["lora_alpha"] == 8 and config["peft_type"] == "LORA"
    assert set(config["target_modules"]) == set(tl.DEFAULT_TARGET_MODULES)
    assert Path(config["base_model_name_or_path"]) == Path(str(tiny))


def test_meta_records_what_is_needed_to_reproduce_and_match(trained, data_dir):
    _, meta = trained
    saved = json.loads((trained[0] / "train_meta.json").read_text(encoding="utf-8"))
    assert saved["base_model"] == meta["base_model"] and saved["lora"]["r"] == 4
    assert saved["prompt"]["system"] == ADAPTER_SYSTEM_PROMPT
    assert saved["prompt"]["system_sha256"] == hashlib.sha256(ADAPTER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert saved["data"]["train_sha256"] == hashlib.sha256((data_dir / "train.jsonl").read_bytes()).hexdigest()
    assert saved["data"]["train_rows"] == 24 and saved["data"]["valid_rows"] == 8
    assert set(saved["versions"]) == {"torch", "transformers", "peft"} and saved["dtype"] == "float32"
    assert 0 < saved["trainable_params"] < saved["total_params"]
    assert saved["train"]["steps"] == 3 * 6  # 24건 / 배치 4 = 6회 × 3에폭


def test_loss_goes_down_and_gen_check_reports(trained):
    _, meta = trained
    history = meta["history"]
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert all(h["valid_loss"] > 0 for h in history)
    check = meta["generate_check"]
    assert check["n"] == 3 and set(check) >= {"json", "equipment_exact", "materials_exact", "both_exact", "summary_usable", "both_exact_by_kind"}
    assert all(0.0 <= check[k] <= 1.0 for k in ("json", "equipment_exact", "materials_exact", "both_exact", "summary_usable"))


def test_saved_adapter_loads_like_the_server_does_and_changes_output(trained, tiny, tokenizer, rows):
    """서버(llm_transformers._attach_adapter)와 같은 방식(PeftModel.from_pretrained, 이름 plan_brief)으로 읽어 쓸 수 있다."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out, _ = trained
    base = AutoModelForCausalLM.from_pretrained(str(tiny)).eval()
    ids = torch.tensor([tl.prompt_ids(tokenizer, rows[0]["messages"])])
    with torch.no_grad():
        before = base(input_ids=ids).logits
    model = PeftModel.from_pretrained(base, str(out), adapter_name="plan_brief").eval()
    model.set_adapter("plan_brief")
    with torch.no_grad():
        with_adapter = model(input_ids=ids).logits
    assert not torch.allclose(before, with_adapter, atol=1e-5)  # 어댑터가 실제로 학습돼 있다 (B 가 0이 아님)
    with model.disable_adapter(), torch.no_grad():
        disabled = model(input_ids=ids).logits
    assert torch.allclose(before, disabled, atol=1e-5)  # 끄면 기본 모델과 같다 (서버가 다른 엔진에서 어댑터를 끄는 방식)


def test_training_learns_the_training_rows(tiny, data_dir, tmp_path):
    """같은 8건을 오래 보면 손실이 크게 준다. 내용이 랜덤인 아주 작은 모델은 어텐션·MLP LoRA 만으로는 못 외우므로(실측: 60회 갱신에 6.6→6.1)
    출력층(lm_head)에도 LoRA 를 붙여 시험한다. 학습 경로(마스킹·패딩·갱신)가 틀렸다면 이것도 줄지 않는다."""
    args = train_args(tiny, data_dir, tmp_path / "x", "--epochs", "30", "--lr", "1e-2", "--r", "16", "--alpha", "32",
                      "--target-modules", *tl.DEFAULT_TARGET_MODULES, "lm_head", "--limit-train", "8", "--limit-valid", "4", "--gen-check", "0")
    meta = tl.train(args, log=lambda *_: None)
    assert meta["history"][-1]["train_loss"] < 0.5 * meta["history"][0]["train_loss"]
    assert meta["generate_check"] is None


def test_model_loss_equals_manual_loss_over_answer_tokens_only(tiny, tokenizer, rows):
    """labels 의 -100(프롬프트·패딩)이 정말 손실에서 빠지는지: 모델 손실 = 정답+eos 토큰만의 평균 교차 엔트로피."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(tiny)).eval()
    batch = [tl.build_example(tokenizer, r["messages"], 4096) for r in rows[:3]]
    tensors = tl._to_tensors(tl.collate(batch, tokenizer.pad_token_id), "cpu")  # 길이가 달라 패딩이 섞인다
    with torch.no_grad():
        got = float(model(**tensors).loss)
        logits = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]).logits
    labels = tensors["labels"]
    shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
    keep = shift_labels != tl.IGNORE
    want = float(torch.nn.functional.cross_entropy(shift_logits[keep], shift_labels[keep]))
    assert abs(got - want) < 1e-5
    assert int(keep.sum()) == sum(len(e.input_ids) - e.prompt_len for e in batch)  # 첫 정답 토큰은 shift 후에도 남는다 (프롬프트 마지막이 예측)


def test_eval_runner_attaches_the_trained_adapter_like_the_server(trained, tiny):
    """scripts.plan_eval.model_runner.load_runner(adapter=) : 학습한 폴더를 서버와 같은 방식으로 붙이고, 그대로 생성한다."""
    from scripts.plan_eval import model_runner

    out, meta = trained
    plain = model_runner.load_runner(str(tiny), dtype="float32", device="cpu", max_new_tokens=8)
    tuned = model_runner.load_runner(str(tiny), dtype="float32", device="cpu", max_new_tokens=8, adapter=str(out))
    assert plain.adapter == "" and plain.adapter_meta == {}
    assert tuned.adapter == str(out) and tuned.adapter_meta["base_model"] == str(tiny)
    assert tuned.adapter_meta["data_sha256"] == meta["data"]["train_sha256"]
    assert tuned.revision == plain.revision  # PeftModel 을 거쳐도 기본 모델의 config 에서 읽는다 (작은 모델은 허브 해시가 없어 둘 다 빈 문자열)
    assert list(tuned.model.peft_config) == [model_runner.ADAPTER_NAME]
    system, user = ADAPTER_SYSTEM_PROMPT, "작업 내용: 용접기로 배관 보수"
    a, b = plain.generate(system, user), tuned.generate(system, user)
    assert a.gen_tokens > 0 and b.gen_tokens > 0 and a.prompt_tokens == b.prompt_tokens  # 같은 입력
    assert a.text != b.text  # 어댑터가 출력을 바꾼다 (학습한 작은 모델은 JSON 꼴을 내기 시작한다)


def test_eval_runner_refuses_adapter_trained_on_another_base(trained, tiny, tmp_path):
    import shutil

    from scripts.plan_eval import model_runner

    copy = tmp_path / "wrong"
    shutil.copytree(trained[0], copy)
    meta = json.loads((copy / "train_meta.json").read_text(encoding="utf-8"))
    meta["base_model"] = "Qwen/Qwen3-1.7B"
    (copy / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="Qwen3-1.7B 로 학습"):
        model_runner.load_runner(str(tiny), dtype="float32", device="cpu", adapter=str(copy))


# ---------------------------------------------------------------- 거부해야 하는 것


def test_refuses_data_made_with_another_system_prompt(tiny, data_dir, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    row = tl.load_rows(data_dir / "train.jsonl")[0]
    row["messages"][0]["content"] = "다른 시스템 프롬프트"
    (bad / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (bad / "valid.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="ADAPTER_SYSTEM_PROMPT"):
        tl.train(train_args(tiny, bad, tmp_path / "o"), log=lambda *_: None)
    assert not (tmp_path / "o").exists()


def test_refuses_empty_training_data(tiny, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "train.jsonl").write_text("", encoding="utf-8")
    (empty / "valid.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="학습 데이터가 없습니다"):
        tl.train(train_args(tiny, empty, tmp_path / "o"), log=lambda *_: None)


def test_parse_args_rejects_nonsense():
    for argv in (["--epochs", "0"], ["--batch-size", "0"], ["--lr", "0"], ["--r", "0"]):
        with pytest.raises(SystemExit):
            tl.parse_args(["--model", "m", "--out", "o", *argv])


def test_source_hygiene():
    source = Path(tl.__file__).read_text(encoding="utf-8")
    assert "\r" not in source
    assert "merge_and_unload" not in source  # 어댑터는 머지하지 않는다
