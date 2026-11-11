"""문장 → 규칙 입력 LLM 추출을 정답 문항으로 채점한다. 실제 모델(시나리오 "B")·실제 추출 경로와 같은 설정을 쓴다.

    python -m scripts.eval_rule_input_extract eval/rule_input_extract_v2.json [--show]
    python -m scripts.eval_rule_input_extract <main-server>/src/test/resources/extract/gold-v1.json

정답 파일 두 가지
  - eval/rule_input_extract_v2.json (LLM 평가 기본, 11/11): "LLM 은 적힌 대로 뽑고 판단은 Spring" 기준.
      measurement {항목: 값}            LLM 이 뽑아야 하는 값 (단위 없음·단위 틀림인 값도 적힌 숫자 그대로)
      duplicates  {항목: [값, 값]}      같은 항목이 두 번 → 두 값 모두 뽑아야 맞음 (Spring 이 "하나만 적어 주세요")
      spring      {항목: 사유}          Spring 이 어떻게 처리할지 (채점에는 안 씀, 검토용)
      tag                               유형. 유형별 점수를 따로 보여 준다
  - gold-v1.json (Spring 사전 추출 회귀 테스트용 20문항): notes 에 있는 항목은 LLM 이 비워야 맞음으로 본다 (옛 기준)
채점 (문항별)
  - 측정값: (항목, 값) 목록이 정답과 같으면 맞음 (순서 무관, 같은 항목 두 번도 각각 비교). 정답에 없는 값은 "지어냄·틀림"으로 센다.
  - 작업 종류: 집합이 같으면 맞음. 감시자: form.watcher_name 과 같으면 맞음 (없으면 null 이어야 맞음).
출력 상한은 실제 추출 경로와 같다 (EXTRACT_MAX_TOKENS). 판정은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from app.api.v1.agent.rule_input_extract import EXTRACT_MAX_TOKENS
from app.core import get_settings
from app.llm.llm_provider import LLMClientProvider
from app.schemas.rule_input_extract import DEFAULT_ITEMS, DEFAULT_WORK_TYPES
from app.services.rule_input_extract_service import SCENARIO_KEY, extract_rule_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("--show", action="store_true", help="틀린 문항의 뽑은 값과 정답을 보여 준다")
    args = parser.parse_args()

    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    settings = get_settings()
    llm = LLMClientProvider.from_settings(settings).get(SCENARIO_KEY)

    m_ok = t_ok = w_ok = invented = 0
    elapsed: list[float] = []
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # 유형 → [모두 맞음, 문항 수]
    max_tokens = min(settings.ai_max_output_tokens, EXTRACT_MAX_TOKENS)
    for case in cases:
        start = time.perf_counter()
        out = extract_rule_input(llm, case["text"], list(DEFAULT_ITEMS), list(DEFAULT_WORK_TYPES), max_tokens=max_tokens)
        elapsed.append(time.perf_counter() - start)

        want = _wanted(case)
        got = Counter((x.item, _num(x.value)) for x in out.measurements)
        invented += sum((got - want).values())
        m_hit = got == want
        t_hit = {w.type for w in out.work_types} == set(case.get("workTypes") or [])
        w_hit = out.watcher_name == (case.get("form") or {}).get("watcher_name")
        m_ok += m_hit
        t_ok += t_hit
        w_ok += w_hit
        tag = by_tag[case.get("tag", "-")]
        tag[0] += m_hit and t_hit and w_hit
        tag[1] += 1
        if args.show and not (m_hit and t_hit and w_hit):
            print(f"#{case['id']} {case['text']}")
            print(f"   측정값 정답 {sorted(want.elements())} / 뽑음 {sorted(got.elements())}")
            print(f"   작업   정답 {case.get('workTypes')} / 뽑음 {[w.type for w in out.work_types]}")
            if not w_hit:
                print(f"   감시자 정답 {(case.get('form') or {}).get('watcher_name')} / 뽑음 {out.watcher_name}")
            if out.dropped:
                print(f"   버림   {out.dropped}")

    n = len(cases)
    print(f"문항 {n}개 · 모델 {llm.name} · 출력 상한 {max_tokens}토큰")
    print(f"측정값 맞음 {m_ok}/{n} ({m_ok / n:.0%}) · 정답과 다른 값(지어냄·틀림) {invented}개")
    print(f"작업 종류 맞음 {t_ok}/{n} ({t_ok / n:.0%}) · 감시자 맞음 {w_ok}/{n} ({w_ok / n:.0%})")
    print(f"문항당 평균 {sum(elapsed) / n:.1f}초 · 최대 {max(elapsed):.1f}초 (Spring 대기 30초, 생성 제한 25초)")
    if len(by_tag) > 1:
        print("유형별 (측정값·작업·감시자 모두 맞음)")
        for name, (ok, total) in by_tag.items():
            print(f"   {name}: {ok}/{total}")
    return 0


def _num(v) -> Decimal:
    return Decimal(str(v)).normalize()


def _wanted(case: dict) -> Counter:
    """정답 (항목, 값) 목록. v2: measurement + duplicates. v1: notes 항목은 비워야 맞음 (옛 기준)"""
    unsure = set(case.get("notes") or [])
    want = Counter((k, _num(v)) for k, v in (case.get("measurement") or {}).items() if k not in unsure)
    for k, values in (case.get("duplicates") or {}).items():
        want.update((k, _num(v)) for v in values)
    return want


if __name__ == "__main__":
    sys.exit(main())
