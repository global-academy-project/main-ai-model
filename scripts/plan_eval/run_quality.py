"""계획 단계 품질 측정 (Colab 권장). 모델 하나를 평가 문장 전체로 돌려 JSON 준수율·후보 정확도·요약 근거를 낸다.

    python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --out out/plan_eval
    python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-1.7B --out out/plan_eval

결과(--out 아래, 모델·dtype별 세 파일):
    <태그>.records.jsonl   문장마다 모델 원문 출력과 채점 값
    <태그>.summary.json    집계
    <태그>.review.csv      요약을 사람이 O/X로 확인하는 표 (요약_사실일치 칸)
두 모델을 다 돌린 뒤 `python -m scripts.plan_eval.compare out/plan_eval/*.summary.json`.

어댑터를 측정할 때 (학습한 짧은 프롬프트로, 서버와 같은 방식으로 붙인다):
    python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --adapter adapters/plan_brief_0.6b --out out/plan_eval
    --prompt short 만 주면 어댑터 없이 짧은 프롬프트로 기본 모델을 잰다 (어댑터 효과와 프롬프트 효과를 가르는 비교용).
    --eval-set scripts/plan_eval/eval_set_multiline.jsonl 로 여러 줄(발화 이어 붙임) 문장도 잰다. 기본 40문장과는 따로 돌려 비교선을 지킨다.

속도는 이 스크립트로 판단하지 않는다 (GPU면 서버 CPU와 다르다). 속도는 run_speed.py를 PC에서 돌린다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .harness import DEFAULT_EVAL_SET, PROMPTS, GenFn, evaluate, load_cases, write_outputs


def adapter_name(adapter: str) -> str:
    """어댑터 폴더 이름 (윈도·리눅스 경로 모두)."""
    return adapter.replace(chr(92), "/").rstrip("/").split("/")[-1]


def tag_for(model: str, dtype: str, prompt: str = "long", adapter: str = "") -> str:
    tag = f"{model.replace('/', '__').replace(chr(92), '__')}__{dtype}"
    if prompt != "long":
        tag += f"__{prompt}"
    if adapter:
        tag += f"__lora-{adapter_name(adapter)}"
    return tag


def run(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[..., GenFn] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="허브 이름 또는 로컬 폴더 (예: Qwen/Qwen3-0.6B)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"],
                        help="서버는 auto. Colab T4는 bfloat16을 못 쓰므로 float16이나 float32")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--prompt", choices=PROMPTS, default=None,
                        help="long = 서버의 기본(어댑터 없음) 프롬프트, short = 어댑터 학습 때의 짧은 프롬프트. 기본: --adapter 가 있으면 short, 없으면 long")
    parser.add_argument("--adapter", default="", help="붙일 LoRA 어댑터 폴더 (adapter_config.json 이 있는 곳). 학습한 기본 모델과 --model 이 같아야 한다")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out", default="out/plan_eval")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (0이면 전부)")
    args = parser.parse_args(argv)

    cases = load_cases(args.eval_set)
    if args.limit:
        cases = cases[: args.limit]

    if runner_factory is None:
        from .model_runner import load_runner

        runner_factory = load_runner
    prompt = args.prompt or ("short" if args.adapter else "long")
    if args.adapter and prompt == "long":
        print("[주의] 어댑터는 짧은 프롬프트로 학습했습니다. 긴 프롬프트로 재면 서버에서 쓰는 조건과 다릅니다.", flush=True)
    extra = {"adapter": args.adapter} if args.adapter else {}
    print(f"모델을 읽습니다: {args.model} (dtype={args.dtype}, device={args.device}, 프롬프트={prompt}"
          f"{', 어댑터=' + args.adapter if args.adapter else ''})", flush=True)
    runner = runner_factory(args.model, dtype=args.dtype, device=args.device, **extra)
    print(f"읽었습니다 ({getattr(runner, 'load_seconds', 0.0):.1f}초, revision={getattr(runner, 'revision', '') or '모름'}). "
          f"{len(cases)}건을 돌립니다.", flush=True)

    def progress(i: int, total: int, case) -> None:
        print(f"  [{i}/{total}] {case.id} ({case.kind})", flush=True)

    records, summary = evaluate(
        cases, runner, model=args.model, dtype=args.dtype, revision=str(getattr(runner, "revision", "") or ""), progress=progress,
        prompt=prompt, adapter=adapter_name(args.adapter) if args.adapter else "",
    )
    if args.adapter:
        info = dict(getattr(runner, "adapter_meta", {}) or {})
        info["path"] = args.adapter
        trained, now = info.get("base_revision") or "", summary["revision"]
        info["revision_match"] = (trained == now) if trained and now else None  # 모르면 None
        if info["revision_match"] is False:
            print(f"[주의] 어댑터를 학습한 기본 모델 판({trained[:10]})과 지금 읽은 판({now[:10]})이 다릅니다.", flush=True)
        summary["adapter_info"] = info
    paths = write_outputs(args.out, tag_for(args.model, args.dtype, prompt, args.adapter), records, summary)
    for name, path in paths.items():
        print(f"{name}: {path}")
    fmt, items = summary["format"], summary["items_all"]
    print(
        f"JSON 파싱 {fmt['json_parse_rate']:.0%} · 형식 {fmt['schema_rate']:.0%} · 잘림 {fmt['max_tokens_cut_rate']:.0%} · "
        f"요약 사용 가능 {summary['summary']['usable_rate']:.0%} · "
        f"후보 F1(최종, 느슨) {_pct(items['final']['lenient']['f1'])} · 규칙만 {_pct(items['rule']['lenient']['f1'])}"
    )
    return 0


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


if __name__ == "__main__":
    sys.exit(run())
