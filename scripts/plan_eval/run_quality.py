"""계획 단계 품질 측정 (Colab 권장). 모델 하나를 평가 문장 전체로 돌려 JSON 준수율·후보 정확도·요약 근거를 낸다.

    python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-0.6B --out out/plan_eval
    python -m scripts.plan_eval.run_quality --model Qwen/Qwen3-1.7B --out out/plan_eval

결과(--out 아래, 모델·dtype별 세 파일):
    <태그>.records.jsonl   문장마다 모델 원문 출력과 채점 값
    <태그>.summary.json    집계
    <태그>.review.csv      요약을 사람이 O/X로 확인하는 표 (요약_사실일치 칸)
두 모델을 다 돌린 뒤 `python -m scripts.plan_eval.compare out/plan_eval/*.summary.json`.

속도는 이 스크립트로 판단하지 않는다 (GPU면 서버 CPU와 다르다). 속도는 run_speed.py를 PC에서 돌린다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .harness import DEFAULT_EVAL_SET, GenFn, evaluate, load_cases, write_outputs


def tag_for(model: str, dtype: str) -> str:
    return f"{model.replace('/', '__').replace(chr(92), '__')}__{dtype}"


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
    print(f"모델을 읽습니다: {args.model} (dtype={args.dtype}, device={args.device})", flush=True)
    runner = runner_factory(args.model, dtype=args.dtype, device=args.device)
    print(f"읽었습니다 ({getattr(runner, 'load_seconds', 0.0):.1f}초, revision={getattr(runner, 'revision', '') or '모름'}). "
          f"{len(cases)}건을 돌립니다.", flush=True)

    def progress(i: int, total: int, case) -> None:
        print(f"  [{i}/{total}] {case.id} ({case.kind})", flush=True)

    records, summary = evaluate(
        cases, runner, model=args.model, dtype=args.dtype, revision=str(getattr(runner, "revision", "") or ""), progress=progress,
    )
    paths = write_outputs(args.out, tag_for(args.model, args.dtype), records, summary)
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
