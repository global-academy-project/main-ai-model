"""아차사고 5축 태깅 정답 세트를 채점한다.

python -m scripts.eval_near_miss_tag eval/near_miss_tag_v1.json
python -m scripts.eval_near_miss_tag eval/near_miss_tag_v1.json --rules-only
python -m scripts.eval_near_miss_tag eval/near_miss_tag_v1.json --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import logging
import os
from collections import defaultdict
from pathlib import Path

from app.core import get_settings
from app.exceptions import LLMError
from app.llm.llm_base import LLMClient
from app.llm.llm_provider import LLMClientProvider
from app.near_miss.schemas import NearMissTagRequest
from app.near_miss.service import tag_near_miss
from app.near_miss.tagging import TAG_CODES, TAGGING_VERSION

AXES = ("ACCIDENT_TYPE", "HAZARD", "WORK_TYPE", "SUBSTANCE")


class RulesOnlyLLM(LLMClient):
    name = "rules-only"

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        raise LLMError("규칙만 평가 모드", engine=self.name)


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("정답 세트는 비어 있지 않은 JSON 배열이어야 합니다.")

    seen: set[int] = set()
    errors: list[str] = []
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, int):
            errors.append("id가 정수가 아닙니다")
        elif case_id in seen:
            errors.append(f"id 중복: {case_id}")
        seen.add(case_id)

        gold = case.get("gold")
        if not isinstance(gold, dict):
            errors.append(f"#{case_id}: gold가 객체가 아닙니다")
            continue
        for axis in AXES:
            values = gold.get(axis)
            if not isinstance(values, list):
                errors.append(f"#{case_id}: gold.{axis}가 배열이 아닙니다")
                continue
            if axis in TAG_CODES:
                invalid = [value for value in values if value not in TAG_CODES[axis]]
                if invalid:
                    errors.append(f"#{case_id}: 잘못된 {axis} 코드 {invalid}")

    if errors:
        raise ValueError("정답 세트 형식 오류:\n- " + "\n- ".join(errors))
    return cases


def metric(predicted: set[str], gold: set[str]) -> tuple[int, int, int, bool]:
    true_positive = len(predicted & gold)
    precision_base = len(predicted)
    recall_base = len(gold)
    return (
        true_positive,
        precision_base,
        recall_base,
        predicted == gold,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.gold)
    if args.rules_only:
        logging.getLogger("app.near_miss.service").setLevel(logging.ERROR)
        llm = RulesOnlyLLM()
    else:
        os.environ.setdefault("NETWORK_ENV", "OPEN")
        os.environ.setdefault("CHECK_RULES_TOOL", "MOCK")
        os.environ.setdefault("MSDS_TOOL", "MOCK")
        settings = get_settings()
        llm = LLMClientProvider.from_settings(settings).get("near_miss_warning")

    totals = {
        axis: {"tp": 0, "pred": 0, "gold": 0, "exact": 0}
        for axis in AXES
    }
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    elapsed: list[float] = []

    for case in cases:
        request = NearMissTagRequest(
            runId=1,
            siteId=1,
            nearMissId=case["id"],
            title=case["title"],
            description=case.get("description", ""),
            immediateAction=case.get("immediateAction"),
            zoneName=case.get("zoneName"),
            workType=case.get("workType"),
        )

        started = time.perf_counter()
        result = tag_near_miss(request, llm)
        elapsed.append(time.perf_counter() - started)

        predicted_by_axis = {
            axis: {
                tag.value
                for tag in result.tags
                if tag.axis == axis
            }
            for axis in AXES
        }
        exact_all = True
        mismatches: dict[str, tuple[set[str], set[str]]] = {}

        for axis in AXES:
            predicted = predicted_by_axis[axis]
            gold = set(case["gold"].get(axis, []))
            tp, pred, gold_count, exact = metric(predicted, gold)
            totals[axis]["tp"] += tp
            totals[axis]["pred"] += pred
            totals[axis]["gold"] += gold_count
            totals[axis]["exact"] += exact
            exact_all &= exact
            if predicted != gold:
                mismatches[axis] = (predicted, gold)

        tag = case.get("tag", "-")
        by_tag[tag][0] += exact_all
        by_tag[tag][1] += 1

        if args.show and mismatches:
            print(f"#{case['id']} {case['title']}")
            for axis, (predicted, gold) in mismatches.items():
                print(f"  {axis}: 정답 {sorted(gold)} / 예측 {sorted(predicted)}")

    n = len(cases)
    print(f"문항 {n}개 · 모델 {llm.name} · 태깅 버전 {TAGGING_VERSION}")
    print("축            정밀도  재현율  완전일치")
    for axis in AXES:
        values = totals[axis]
        precision = values["tp"] / values["pred"] if values["pred"] else 1.0
        recall = values["tp"] / values["gold"] if values["gold"] else 1.0
        exact = values["exact"] / n
        print(f"{axis:<14} {precision:.2f}    {recall:.2f}    {exact:.2f}")

    exact_all_total = sum(by_tag[tag][0] for tag in by_tag)
    print(f"완전 일치 문항 {exact_all_total}/{n} ({exact_all_total / n:.2f})")
    print(f"문항당 평균 {sum(elapsed) / n:.1f}초")

    if len(by_tag) > 1:
        print("tag별 완전 일치")
        for tag, (ok, total) in by_tag.items():
            print(f"  {tag}: {ok}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
