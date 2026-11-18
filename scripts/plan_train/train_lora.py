"""계획 단계 LoRA 어댑터 학습 (B). transformers + peft 만 쓴다 (trl 등 다른 학습 라이브러리 없음).

    python -m scripts.plan_train.train_lora --model Qwen/Qwen3-0.6B --data data/plan-train --out adapters/plan_brief_0.6b
    python -m scripts.plan_train.train_lora --model Qwen/Qwen3-1.7B --data data/plan-train --out adapters/plan_brief_1.7b

데이터는 scripts/make_plan_train.py 가 만든 train.jsonl·valid.jsonl 이다 (messages = system, user, assistant).

**서버와 같게 맞춘 것** (같지 않으면 어댑터가 제 값을 못 낸다)
  - 프롬프트: 채팅 템플릿 apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False).
    서버의 llm_transformers.generate_chat 과 같은 호출이다. 템플릿은 토크나이저가 정하므로 여기서 따로 쓰지 않는다.
  - 학습 대상(손실): 정답 JSON + 끝 토큰(eos)만. 프롬프트 부분은 손실에서 뺀다 (labels = -100). 서버는 eos 가 나오면 생성을 멈춘다.
  - 프롬프트와 정답은 따로 토큰화해서 이어 붙인다 (서버도 프롬프트를 먼저 읽고 그 뒤를 생성한다).
**어댑터만 저장한다** (기본 모델은 머지하지 않는다, C 의 extract 와 같은 방식). 저장 폴더에 adapter_config.json·adapter_model.safetensors·
train_meta.json 이 생긴다. 어댑터는 학습한 기본 모델·판에만 붙으므로 train_meta.json 에 모델 이름과 허브 커밋 해시를 적어 둔다.

dtype: auto 는 GPU가 bfloat16 을 지원하면 bfloat16, 아니면 float16(T4 등)이다. float16 은 손실 스케일러를 쓴다. CPU 는 float32.
학습이 끝나면 valid 앞부분 몇 건을 서버와 같은 방식(그리디)으로 생성해 JSON 파싱·항목 일치·요약 사용 가능 비율을 출력한다 (간이 확인이며
최종 평가는 scripts/plan_eval 의 평가 40문장으로 한다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DTYPES = ("auto", "float32", "float16", "bfloat16")
IGNORE = -100


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]
    prompt_len: int


# ---------------------------------------------------------------- 데이터 (토크나이저만 필요, 모델은 필요 없다)


def load_rows(path: Path | str, limit: int = 0) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def prompt_ids(tokenizer: Any, messages: list[dict]) -> list[int]:
    """서버와 같은 호출로 만든 프롬프트 토큰 (system + user, 생성 시작 표시, 생각 끄기)."""
    out = tokenizer.apply_chat_template(
        messages[:2], add_generation_prompt=True, tokenize=True, return_dict=True, enable_thinking=False,
    )
    ids = out["input_ids"]
    return list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)


def build_example(tokenizer: Any, messages: list[dict], max_len: int) -> Example:
    """프롬프트 + 정답 JSON + eos. 손실은 정답과 eos 에만."""
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("messages 는 system, user, assistant 순서여야 합니다.")
    head = prompt_ids(tokenizer, messages)
    eos = tokenizer.eos_token
    if eos is None or tokenizer.eos_token_id is None:
        raise ValueError("토크나이저에 eos 토큰이 없습니다.")
    tail = tokenizer(messages[2]["content"] + eos, add_special_tokens=False)["input_ids"]
    if tail[-1] != tokenizer.eos_token_id:
        raise ValueError(f"정답 끝이 eos 토큰으로 토큰화되지 않았습니다: {tail[-3:]} != {tokenizer.eos_token_id}")
    ids = head + tail
    if len(ids) > max_len:
        raise ValueError(f"예시가 {len(ids)}토큰입니다 (--max-len {max_len}). 자르면 정답이 잘리므로 멈춥니다.")
    return Example(ids, [IGNORE] * len(head) + tail, len(head))


def build_examples(tokenizer: Any, rows: list[dict], max_len: int) -> list[Example]:
    return [build_example(tokenizer, r["messages"], max_len) for r in rows]


def collate(batch: list[Example], pad_id: int) -> dict[str, list[list[int]]]:
    """오른쪽을 pad 로 채운다 (학습은 오른쪽 패딩, 생성 확인은 한 건씩이라 패딩 없음)."""
    width = max(len(e.input_ids) for e in batch)
    return {
        "input_ids": [e.input_ids + [pad_id] * (width - len(e.input_ids)) for e in batch],
        "attention_mask": [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in batch],
        "labels": [e.labels + [IGNORE] * (width - len(e.labels)) for e in batch],
    }


def batches(examples: list[Example], size: int, rng: random.Random, *, shuffle: bool) -> list[list[Example]]:
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    return [[examples[i] for i in order[k : k + size]] for k in range(0, len(order), size)]


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 모델


def resolve_dtype(name: str, device: str) -> tuple[Any, str]:
    import torch

    if name == "auto":
        if device == "cpu":
            name = "float32"
        else:
            name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device == "cpu" and name != "float32":
        raise ValueError(f"CPU 에서는 float32 만 씁니다 (요청 {name}).")
    return getattr(torch, name), name


def load_base(model_id: str, dtype: Any, device: str) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:  # 옛 transformers 는 torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model.to(device)


def base_revision(model: Any) -> str:
    cfg = getattr(model, "config", None)
    return str(getattr(cfg, "_commit_hash", None) or "")


def lora_model(base: Any, *, r: int, alpha: int, dropout: float, target_modules: tuple[str, ...]) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, config)


# ---------------------------------------------------------------- 학습·확인


def _to_tensors(batch: dict[str, list[list[int]]], device: str) -> dict[str, Any]:
    import torch

    return {k: torch.tensor(v, device=device) for k, v in batch.items()}


def evaluate_loss(model: Any, examples: list[Example], pad_id: int, batch_size: int, device: str, autocast: Any) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for chunk in batches(examples, batch_size, random.Random(0), shuffle=False):
            tensors = _to_tensors(collate(chunk, pad_id), device)
            with autocast():
                loss = model(**tensors).loss
            tokens = sum(sum(1 for x in e.labels if x != IGNORE) for e in chunk)
            total += float(loss) * tokens  # 건수가 아니라 정답 토큰 수로 가중 (모델 손실은 토큰 평균)
            count += tokens
    return total / max(1, count)


def generate_check(model: Any, tokenizer: Any, rows: list[dict], device: str, *, max_new_tokens: int = 160) -> dict[str, Any]:
    """valid 앞부분을 서버와 같은 방식(그리디, 160토큰)으로 생성해 서버 검증 함수로 채점한다 (간이)."""
    import torch

    from app.agent import plan_brief as pb
    from app.agent.evidence_collector.textutil import extract_json_object

    model.eval()
    totals: Counter[str] = Counter()
    per_kind: dict[str, Counter[str]] = {}
    for row in rows:
        text = "\n".join(row["utterances"])
        ids = torch.tensor([prompt_ids(tokenizer, row["messages"])], device=device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        new_ids = out[0][ids.shape[-1] :]
        raw = extract_json_object(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
        gold = row["answer"]
        ok = {"n": 1, "json": raw is not None}
        if raw is not None:
            equipment, _ = pb.clean_equipment(raw.get("equipment"), text)
            materials, _ = pb.clean_materials(raw.get("materials"), text)
            want_e = pb._canonical(gold["equipment"], pb._EQUIPMENT_KEYS)  # noqa: SLF001
            want_m = pb._canonical(gold["materials"], pb._MATERIAL_KEYS)  # noqa: SLF001
            ok["equipment_exact"] = set(equipment) == set(want_e)
            ok["materials_exact"] = set(materials) == set(want_m)
            ok["both_exact"] = ok["equipment_exact"] and ok["materials_exact"]
            ok["summary_usable"] = pb.clean_summary(raw.get("summary"), text) is not None
        for key, value in ok.items():
            totals[key] += int(value)
            per_kind.setdefault(row["kind"], Counter())[key] += int(value)
    n = max(1, totals["n"])
    return {
        "n": totals["n"],
        **{k: round(totals[k] / n, 4) for k in ("json", "equipment_exact", "materials_exact", "both_exact", "summary_usable")},
        "both_exact_by_kind": {k: round(c["both_exact"] / max(1, c["n"]), 4) for k, c in sorted(per_kind.items())},
    }


def train(args: argparse.Namespace, *, log: Any = print) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype, dtype_name = resolve_dtype(args.dtype, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data = Path(args.data)
    train_rows = load_rows(data / "train.jsonl", args.limit_train)
    valid_rows = load_rows(data / "valid.jsonl", args.limit_valid)
    if not train_rows:
        raise ValueError(f"학습 데이터가 없습니다: {data / 'train.jsonl'}")
    if any(r["messages"][0]["content"] != ADAPTER_SYSTEM_PROMPT for r in [*train_rows, *valid_rows]):
        raise ValueError("데이터의 시스템 프롬프트가 지금 서버의 ADAPTER_SYSTEM_PROMPT 와 다릅니다. 생성기로 데이터를 다시 만드세요.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ex = build_examples(tokenizer, train_rows, args.max_len)
    valid_ex = build_examples(tokenizer, valid_rows, args.max_len)
    log(f"학습 {len(train_ex)}건, 검증 {len(valid_ex)}건. 토큰 길이 평균 {sum(len(e.input_ids) for e in train_ex) / len(train_ex):.0f}"
        f" (프롬프트 {sum(e.prompt_len for e in train_ex) / len(train_ex):.0f}), 최대 {max(len(e.input_ids) for e in train_ex)}")

    started = time.perf_counter()
    base = load_base(args.model, dtype, device)
    revision = base_revision(base)
    model = lora_model(base, r=args.r, alpha=args.alpha, dropout=args.dropout, target_modules=tuple(args.target_modules))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"모델 {args.model} ({dtype_name}, {device}, revision {revision[:10] or '모름'}) 읽음 {time.perf_counter() - started:.0f}초. "
        f"학습 파라미터 {trainable:,} / {total:,} ({trainable / total:.2%})")

    use_amp = device == "cuda" and dtype_name != "float32"
    scaler = torch.amp.GradScaler("cuda") if use_amp and dtype_name == "float16" else None

    def autocast() -> Any:
        return torch.autocast(device_type="cuda", dtype=dtype) if use_amp else torch.autocast(device_type="cpu", enabled=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(train_ex) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_steps * args.warmup)), total_steps)
    history: list[dict[str, Any]] = []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        chunks = batches(train_ex, args.batch_size, rng, shuffle=True)
        running, seen = 0.0, 0
        for index, chunk in enumerate(chunks, start=1):
            tensors = _to_tensors(collate(chunk, tokenizer.pad_token_id), device)
            with autocast():
                loss = model(**tensors).loss / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            running += float(loss.detach()) * args.grad_accum
            seen += 1
            if index % args.grad_accum == 0 or index == len(chunks):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    log(f"  epoch {epoch} step {step}/{total_steps} loss {running / seen:.4f} lr {scheduler.get_last_lr()[0]:.2e}")
        valid_loss = evaluate_loss(model, valid_ex, tokenizer.pad_token_id, args.batch_size, device, autocast) if valid_ex else None
        history.append({"epoch": epoch, "train_loss": round(running / max(1, seen), 5), "valid_loss": None if valid_loss is None else round(valid_loss, 5)})
        log(f"epoch {epoch}: 학습 손실 {history[-1]['train_loss']}, 검증 손실 {history[-1]['valid_loss']} ({time.perf_counter() - started:.0f}초)")

    check = generate_check(model, tokenizer, valid_rows[: args.gen_check], device) if args.gen_check and valid_rows else None
    if check:
        log("생성 확인(valid 앞 %d건): JSON %.0f%% · 공구 일치 %.0f%% · 물질 일치 %.0f%% · 둘 다 %.0f%% · 요약 사용 가능 %.0f%%" % (
            check["n"], check["json"] * 100, check["equipment_exact"] * 100, check["materials_exact"] * 100, check["both_exact"] * 100,
            check["summary_usable"] * 100))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))  # 어댑터만 저장 (adapter_config.json, adapter_model.safetensors)
    import peft
    import transformers

    meta = {
        "base_model": args.model, "base_revision": revision, "dtype": dtype_name, "device": device,
        "prompt": {"system": ADAPTER_SYSTEM_PROMPT, "system_sha256": hashlib.sha256(ADAPTER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()},
        "data": {"dir": str(data), "train_rows": len(train_ex), "valid_rows": len(valid_ex),
                 "train_sha256": file_sha256(data / "train.jsonl"), "valid_sha256": file_sha256(data / "valid.jsonl")},
        "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout, "target_modules": list(args.target_modules)},
        "train": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum, "max_len": args.max_len,
                  "warmup": args.warmup, "weight_decay": args.weight_decay, "seed": args.seed, "steps": step},
        "history": history, "generate_check": check,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "trainable_params": trainable, "total_params": total,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    log(f"저장: {out} (adapter_config.json, adapter_model.safetensors, train_meta.json)")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="기본 모델 (허브 이름 또는 로컬 폴더, 예: Qwen/Qwen3-0.6B)")
    p.add_argument("--data", default="data/plan-train", help="train.jsonl·valid.jsonl 이 있는 폴더 (scripts.make_plan_train 출력)")
    p.add_argument("--out", required=True, help="어댑터를 저장할 폴더 (예: adapters/plan_brief_0.6b)")
    p.add_argument("--dtype", default="auto", choices=DTYPES)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="이만큼 모아서 한 번 갱신 (실효 배치 = batch-size × grad-accum)")
    p.add_argument("--max-len", type=int, default=512, help="예시 최대 토큰 (넘으면 자르지 않고 멈춘다)")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    p.add_argument("--warmup", type=float, default=0.05, help="전체 갱신 횟수 중 워밍업 비율")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--gen-check", type=int, default=60, help="학습 뒤 valid 앞 N건을 생성해 확인 (0이면 건너뜀)")
    p.add_argument("--limit-train", type=int, default=0, help="앞에서 N건만 학습 (시험용)")
    p.add_argument("--limit-valid", type=int, default=0, help="앞에서 N건만 검증 (시험용)")
    args = p.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.r <= 0 or args.lr <= 0:
        p.error("--epochs, --batch-size, --grad-accum, --r, --lr 은 양수여야 합니다")
    return args


def main(argv: list[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
