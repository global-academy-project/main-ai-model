"""run_quality가 만든 *.summary.json 여러 개를 한 표로 비교한다 (마크다운으로 출력, --out이면 파일로도 저장).

    python -m scripts.plan_eval.compare out/plan_eval/*.summary.json --out out/plan_eval/compare.md
파워셸은 *를 풀어 주지 않아서(T26) 이 스크립트가 직접 푼다. 따옴표로 감싸도 된다.
규칙만(모델 없이 슬롯 사전만 쓴 경우)의 값은 첫 파일에서 읽는다 (모든 파일에 같은 값이 들어 있다).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _label(summary: dict[str, Any]) -> str:
    extra = (", 짧은 프롬프트" if summary.get("prompt", "long") == "short" else "") + (f", +{summary['adapter']}" if summary.get("adapter") else "")
    return f"{summary['model'].split('/')[-1]} ({summary['dtype']}{extra})"


def expand_paths(patterns: Sequence[str]) -> list[str]:
    """*·? 가 든 인자는 직접 푼다 (윈도 파워셸은 풀지 않고 그대로 넘겨 Errno 22 가 났다). 맞는 파일이 없으면 ValueError."""
    paths: list[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            found = sorted(glob.glob(pattern))
            if not found:
                raise ValueError(f"맞는 파일이 없습니다: {pattern}")
            paths += found
        else:
            paths.append(pattern)
    return list(dict.fromkeys(paths))  # 같은 파일이 두 번 들어와도 한 번만


def render(summaries: Sequence[dict[str, Any]]) -> str:
    if not summaries:
        raise ValueError("비교할 summary가 없습니다.")
    labels = [_label(s) for s in summaries]
    lines: list[str] = []

    def table(title: str, rows: list[tuple[str, list[str]]]) -> None:
        lines.append(f"### {title}\n")
        lines.append("| 항목 | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in labels) + " |")
        for name, values in rows:
            lines.append(f"| {name} | " + " | ".join(values) + " |")
        lines.append("")

    table("형식", [
        ("평가 문장 수", [str(s["n"]) for s in summaries]),
        ("모델 revision (커밋)", [(s.get("revision") or "-")[:10] for s in summaries]),
        ("프롬프트", ["짧음(어댑터용)" if s.get("prompt", "long") == "short" else "긴 것(기본)" for s in summaries]),
        ("어댑터 (학습한 기본 모델 판과 일치)", [
            "-" if not s.get("adapter") else f"{s['adapter']} ({ {True: '일치', False: '**불일치**', None: '모름'}[(s.get('adapter_info') or {}).get('revision_match')] })"
            for s in summaries
        ]),
        ("JSON 파싱 성공(첫 시도)", [_pct(s["format"]["json_parse_rate"]) for s in summaries]),
        ("키·자료형 맞음", [_pct(s["format"]["schema_rate"]) for s in summaries]),
        ("생성 한도(max_new_tokens)에서 잘림", [_pct(s["format"]["max_tokens_cut_rate"]) for s in summaries]),
        ("생성 토큰 평균 / 최대", [f"{_num(s['tokens']['gen_mean'])} / {s['tokens']['gen_max']}" for s in summaries]),
        ("프롬프트 토큰 평균", [_num(s["tokens"]["prompt_mean"], 0) for s in summaries]),
    ])
    table("공구·물질 후보 (느슨한 비교: 한쪽이 다른 쪽에 포함돼도 맞음)", [
        ("공구 F1 (모델이 낸 것, 검증 통과분)", [_pct(s["items"]["equipment"]["llm"]["lenient"]["f1"]) for s in summaries]),
        ("물질 F1 (모델이 낸 것, 검증 통과분)", [_pct(s["items"]["materials"]["llm"]["lenient"]["f1"]) for s in summaries]),
        ("공구 F1 (최종 = 슬롯 + 모델)", [_pct(s["items"]["equipment"]["final"]["lenient"]["f1"]) for s in summaries]),
        ("물질 F1 (최종 = 슬롯 + 모델)", [_pct(s["items"]["materials"]["final"]["lenient"]["f1"]) for s in summaries]),
        ("공구·물질 F1 (최종, 엄격: 이름이 정확히 같아야 맞음)", [_pct(s["items_all"]["final"]["strict"]["f1"]) for s in summaries]),
        ("전체 재현율 (최종)", [_pct(s["items_all"]["final"]["lenient"]["recall"]) for s in summaries]),
        ("전체 정밀도 (최종)", [_pct(s["items_all"]["final"]["lenient"]["precision"]) for s in summaries]),
        ("공구 후보 버려진 비율", [_pct(s["candidates"]["equipment_drop_rate"]) for s in summaries]),
        ("물질 후보 버려진 비율", [_pct(s["candidates"]["materials_drop_rate"]) for s in summaries]),
    ])
    base = summaries[0]["items_all"]["rule"]["lenient"]
    lines.append(
        f"모델 없이 슬롯 사전만 쓴 기준: 전체 F1 {_pct(base['f1'])} (정밀도 {_pct(base['precision'])}, 재현율 {_pct(base['recall'])}). "
        "모델이 이보다 나아야 쓸 이유가 있다.\n"
    )
    extra = summaries[0].get("rule_extra_cases", [])
    if extra:
        shown = ", ".join(f"{e['id']}({'·'.join(e['items'])})" for e in extra)
        lines.append(f"슬롯 사전만으로 정답에 없는 이름이 잡힌 문장: {len(extra)}건 — {shown}. 모델과 상관없이 최종 값에 섞여 들어간다.\n")
    kinds = [k for k in summaries[0]["by_kind"]]
    table("유형별 전체 F1 (최종, 느슨) — 괄호는 규칙만", [
        (f"{k} (n={summaries[0]['by_kind'][k]['n']})", [
            f"{_pct(s['by_kind'][k]['items_all']['final']['lenient']['f1'])} ({_pct(s['by_kind'][k]['items_all']['rule']['lenient']['f1'])})"
            for s in summaries
        ]) for k in kinds if k in summaries[0]["by_kind"]
    ])
    if "none" in summaries[0]["by_kind"]:
        table("뽑을 것이 없는 문장(none)에서 지어낸 후보", [
            ("후보를 낸 문장 수 (검증 전)", [str(s["by_kind"]["none"]["raw_candidate_cases"]) for s in summaries]),
            ("검증 뒤에도 남은 문장 수", [str(s["by_kind"]["none"]["kept_candidate_cases"]) for s in summaries]),
        ])
    table("요약", [
        ("그대로 쓸 수 있음", [_pct(s["summary"]["usable_rate"]) for s in summaries]),
        ("쓸 수 없는 이유 (JSON없음/빈값/너무짧음/원문에 없는 숫자)",
         [" / ".join(str(s["summary"]["reasons"][k]) for k in ("no_json", "empty", "short", "number")) for s in summaries]),
        ("60자 넘어 잘림", [_pct(s["summary"]["truncated_rate"]) for s in summaries]),
        ("원문 근거 비율 평균 (거친 지표)", [_pct(s["summary"]["mean_support"]) for s in summaries]),
        ("근거 없는 단어가 든 요약 비율 (사람이 확인할 대상)", [_pct(s["summary"]["unsupported_case_rate"]) for s in summaries]),
    ])
    lines.append("요약의 사실 일치는 `*.review.csv`의 `요약_사실일치` 칸에 직접 O/X로 적어서 판단한다.\n")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in expand_paths(args.summaries)]
    text = render(summaries)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
