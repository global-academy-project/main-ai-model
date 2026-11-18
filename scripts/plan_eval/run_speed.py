"""계획 단계 속도 측정 (PC에서, 서버와 같은 CPU로). 같은 모델을 dtype별로 읽어 5문장 안팎을 재 본다.

    python -m scripts.plan_eval.run_speed --model Qwen/Qwen3-0.6B --dtypes auto,float32,bfloat16 --n 5
    python -m scripts.plan_eval.run_speed --model Qwen/Qwen3-1.7B --dtypes auto,float32 --n 5

서버가 `torch_dtype="auto"`(체크포인트에 적힌 dtype, Qwen3는 보통 bfloat16)로 올리므로 auto가 서버와 같다.
문장마다 (1) 프롬프트만 읽는 시간(max_new_tokens=1)과 (2) 전체 생성 시간을 따로 재서
프롬프트 읽는 시간 / 토큰당 생성 시간 / 초당 토큰을 구한다. 첫 문장은 워밍업으로 버린다 (결과표에는 안 들어간다).
CPU만 쓰도록 --device cpu가 기본이다. 다른 프로그램을 끄고 돌린다 (CPU를 나눠 쓰면 값이 올라간다).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .harness import DEFAULT_EVAL_SET, build_messages, load_cases


def measure_case(runner: Any, system: str, user: str) -> dict[str, float]:
    """한 문장: 프롬프트 읽기(1토큰 생성)와 전체 생성 시간. runner는 generate(system, user, max_new_tokens=)를 가진다."""
    prefill = runner.generate(system, user, max_new_tokens=1)
    full = runner.generate(system, user)
    decode_tokens = max(full.gen_tokens - 1, 0)
    decode_seconds = max(full.seconds - prefill.seconds, 0.0)
    return {
        "prompt_tokens": full.prompt_tokens, "gen_tokens": full.gen_tokens,
        "prefill_seconds": prefill.seconds, "total_seconds": full.seconds,
        "decode_seconds": decode_seconds,
        "tokens_per_second": (decode_tokens / decode_seconds) if decode_seconds > 0 and decode_tokens else 0.0,
    }


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    mean = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    return {
        "n": n, "prompt_tokens": mean("prompt_tokens"), "gen_tokens": mean("gen_tokens"),
        "prefill_seconds": mean("prefill_seconds"), "total_seconds": mean("total_seconds"),
        "decode_seconds": mean("decode_seconds"), "tokens_per_second": mean("tokens_per_second"),
    }


def run(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[..., Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtypes", default="auto,float32", help="쉼표로 구분 (auto, float32, bfloat16, float16)")
    parser.add_argument("--n", type=int, default=5, help="재는 문장 수 (워밍업 한 문장 별도)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="torch 스레드 수 (0이면 그대로)")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out", default="out/plan_eval")
    args = parser.parse_args(argv)

    cases = load_cases(args.eval_set)
    # 길이가 다른 문장이 섞이게 간격을 두고 고른다 (앞 N건만 쓰면 한 유형에 쏠린다)
    step = max(len(cases) // max(args.n, 1), 1)
    picked = [cases[i] for i in range(0, len(cases), step)][: args.n + 1]
    warmup, measured = picked[0], picked[1:]

    if runner_factory is None:
        from .model_runner import load_runner

        runner_factory = load_runner
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    results: dict[str, Any] = {"model": args.model, "device": args.device, "cases": [c.id for c in measured], "dtypes": {}}
    for dtype in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
        print(f"[{dtype}] 모델을 읽습니다...", flush=True)
        runner = runner_factory(args.model, dtype=dtype, device=args.device)
        runner.generate(*build_messages(warmup))  # 워밍업
        rows = [measure_case(runner, *build_messages(c)) for c in measured]
        agg = aggregate(rows)
        agg["load_seconds"] = getattr(runner, "load_seconds", 0.0)
        results["dtypes"][dtype] = {"summary": agg, "rows": rows}
        print(
            f"[{dtype}] 프롬프트 {agg['prompt_tokens']:.0f}토큰 읽기 {agg['prefill_seconds']:.1f}초 · "
            f"생성 {agg['gen_tokens']:.0f}토큰 {agg['decode_seconds']:.1f}초 ({agg['tokens_per_second']:.1f} tok/s) · "
            f"합계 {agg['total_seconds']:.1f}초",
            flush=True,
        )
        del runner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"speed__{args.model.replace('/', '__')}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("저장:", path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
