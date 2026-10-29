"""transformers 모델(예: Qwen/Qwen3-0.6B)로 시나리오 분류를 손으로 돌려 보는 스크립트.

저장소 루트에서 실행한다:
    python scripts/try_router.py Qwen/Qwen3-0.6B
    python scripts/try_router.py Qwen/Qwen3-0.6B --chat A,E,F      # 채팅 범위 제한
    python scripts/try_router.py ./models/qwen3-0.6b --closed      # 폐쇄망: 로컬 폴더, 다운로드 안 함
    python scripts/try_router.py Qwen/Qwen3-0.6B "직접 입력한 문장" "또 다른 문장"

첫 실행은 모델을 내려받고 읽느라 오래 걸린다. 문장마다 시나리오·확신도·소요 시간·`source`를 보여 준다.
`source=fallback`이 많으면 모델이 JSON을 못 낸 것이다 (프롬프트·모델 크기를 의심한다).
샘플 문장은 계획서 9장 데모 문장이 아니다 (평가 오염 방지). 정답 라벨은 사람이 붙인 임시값이다.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.scenario.scenario_dispatch import parse_chat_scenarios  # noqa: E402
from app.agent.scenario.scenario_router import Scenario, ScenarioRouter  # noqa: E402
from app.llm.llm_base import LLMClient  # noqa: E402

A, D, E, F, U = (
    Scenario.WORK_REGISTER,
    Scenario.DASHBOARD,
    Scenario.AUDIT_QUERY,
    Scenario.GUIDE,
    Scenario.UNKNOWN,
)

SAMPLES: list[tuple[str, Scenario]] = [
    ("내일 오전 10시에 저장탱크 내부 청소 작업이 있어요. 필요한 서류 만들어 주세요.", A),
    ("오늘 아직 안 쓴 서류가 뭐가 남았는지 알려 주세요.", D),
    ("작년 12월에 접수된 허가 중에 서명이 빠진 게 있었는지 찾아 주세요.", E),
    ("직원이 열 명 정도인 작은 공장인데, 옥상 방수 작업을 하려면 뭘 준비해야 하나요?", F),
    ("안녕하세요", U),
    ("점심 메뉴 추천해 줘", U),
]


def run(
    llm: LLMClient,
    *,
    allowed: Sequence[Scenario] | None = None,
    samples: Sequence[tuple[str, Scenario]] = SAMPLES,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    """샘플을 분류해 결과를 출력하고 집계를 돌려준다."""
    router = ScenarioRouter(llm, allowed=allowed)
    total = correct = fallback = 0
    for text, expected in samples:
        started = time.perf_counter()
        decision = router.route(text)
        elapsed = time.perf_counter() - started
        # 채팅 범위 밖 시나리오는 UNKNOWN + redirect가 정답이다
        hit = decision.scenario is expected or (decision.redirect is expected and expected is not U)
        total += 1
        correct += hit
        fallback += decision.source == "fallback"
        shown = decision.scenario.value if decision.redirect is None else f"UNKNOWN(→{decision.redirect.value} 전용 화면)"
        out(
            f"[{'O' if hit else 'X'}] 기대={expected.value:<7} 결과={shown:<18} "
            f"확신도={decision.confidence:.2f} source={decision.source:<8} {elapsed:5.1f}s | {text}"
        )
    out(f"\n정답 {correct}/{total}, 대체 경로(fallback) {fallback}/{total}")
    return {"total": total, "correct": correct, "fallback": fallback}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="허브 모델 ID 또는 로컬 폴더 경로 (예: Qwen/Qwen3-0.6B)")
    parser.add_argument("texts", nargs="*", help="직접 시험할 문장 (없으면 내장 샘플)")
    parser.add_argument("--chat", help="채팅에서 분류할 시나리오, 예: A,E,F (기본 전부)")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    parser.add_argument("--closed", action="store_true", help="폐쇄망처럼 허브 다운로드 없이 로컬 파일만 사용")
    parser.add_argument("--thinking", action="store_true", help="추론(<think>) 모드 켜기 (기본 꺼짐)")
    args = parser.parse_args(argv)

    from app.llm.llm_categorizator import build_categorizator_client

    settings = SimpleNamespace(
        network_env="CLOSED" if args.closed else "OPEN",
        categorizator_model_id=args.model,
        categorizator_device=args.device,
        categorizator_enable_thinking=args.thinking,
    )
    llm = build_categorizator_client(settings=settings)
    samples = [(t, U) for t in args.texts] if args.texts else SAMPLES
    allowed = parse_chat_scenarios(args.chat)
    print(f"모델={args.model} device={args.device} closed={args.closed} 채팅범위={args.chat or '전부'}\n")
    run(llm, allowed=allowed, samples=samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
