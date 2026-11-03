"""문장 → 규칙 입력 LLM 추출을 Spring 정답 문항(gold-v1.json)으로 채점한다. 실제 모델(시나리오 "B")을 쓴다.

    python -m scripts.eval_rule_input_extract <gold-v1.json 경로> [--show]

gold-v1.json: main-server 저장소 src/test/resources/extract/gold-v1.json
    [{"id", "text", "workTypes": [...], "measurement": {"O2": 17.5, ...}, "form": {"watcher_name"}, "notes": [...]}, ...]
채점 (문항별)
  - 측정값: 정답 항목·값이 모두 맞고 여분이 없으면 맞음. 정답과 다른 값을 뽑으면 "지어냄·틀림"으로 따로 센다.
    notes 에 있는 항목(단위 없음 등 사람에게 물어야 하는 값)은 LLM 이 비워야 맞음으로 본다.
  - 작업 종류: 집합이 같으면 맞음. 감시자: form.watcher_name 과 같으면 맞음 (없으면 null 이어야 맞음).
모델 설정은 서버와 같다 (.env 의 SCENARIO_MODELS·엔진 설정). 판정은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

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
    for case in cases:
        start = time.perf_counter()
        out = extract_rule_input(llm, case["text"], list(DEFAULT_ITEMS), list(DEFAULT_WORK_TYPES),
                                 max_tokens=settings.ai_max_output_tokens)
        elapsed.append(time.perf_counter() - start)

        unsure = set(case.get("notes") or [])
        want = {k: Decimal(str(v)) for k, v in (case.get("measurement") or {}).items() if k not in unsure}
        got = {x.item: x.value for x in out.measurements}
        invented += sum(1 for k, v in got.items() if want.get(k) != v)
        m_hit = got == want
        t_hit = {w.type for w in out.work_types} == set(case.get("workTypes") or [])
        w_hit = out.watcher_name == (case.get("form") or {}).get("watcher_name")
        m_ok += m_hit
        t_ok += t_hit
        w_ok += w_hit
        if args.show and not (m_hit and t_hit and w_hit):
            print(f"#{case['id']} {case['text']}")
            print(f"   측정값 정답 {dict(want)} / 뽑음 {dict(got)}")
            print(f"   작업   정답 {case.get('workTypes')} / 뽑음 {[w.type for w in out.work_types]}")
            if not w_hit:
                print(f"   감시자 정답 {(case.get('form') or {}).get('watcher_name')} / 뽑음 {out.watcher_name}")
            if out.dropped:
                print(f"   버림   {out.dropped}")

    n = len(cases)
    print(f"문항 {n}개 · 모델 {llm.name}")
    print(f"측정값 맞음 {m_ok}/{n} ({m_ok / n:.0%}) · 정답과 다른 값(지어냄·틀림) {invented}개")
    print(f"작업 종류 맞음 {t_ok}/{n} ({t_ok / n:.0%}) · 감시자 맞음 {w_ok}/{n} ({w_ok / n:.0%})")
    print(f"문항당 평균 {sum(elapsed) / n:.1f}초 · 최대 {max(elapsed):.1f}초 (Spring 대기 15초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
