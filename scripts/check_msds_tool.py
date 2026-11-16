"""lookup_msds가 실제 공단 도구로 등록됐는지 확인한다 (PC 로컬·docker 컨테이너). 서버·LLM·Spring 없이 도구 레지스트리만 만든다.

서버와 같은 방식으로 설정(환경변수·.env)을 읽고 같은 get_tool_registry()를 쓰므로, 여기서 '실제'로 나오면 서버도 같다.
화면 확인 전에 "코드가 반영됐는지"를 이 출력으로 가린다 (시험 통과 ≠ 반영).

    python -B -m scripts.check_msds_tool                          # 기본 물질 4개
    python -B -m scripts.check_msds_tool 톨루엔 메탄올 시너        # 이름을 직접
    python -B -m scripts.check_msds_tool --require-real          # mock이면 종료 코드 1

docker: docker exec python-ai-container python -B -m scripts.check_msds_tool   (scripts 폴더가 이미지에 들어 있을 때)

키는 출력하지 않는다. MSDS_TOOL=KOSHA일 때 공단 API를 이름당 최대 4번 부른다.
"""

from __future__ import annotations

import argparse
import sys
import time

from pydantic import ValidationError

from app.agent import agent_deps
from app.core import get_settings
from app.exceptions import ToolError

DEFAULT_NAMES = ["톨루엔", "메탄올", "이소프로필알코올", "시너"]
FIRST_AID_CHARS = 80


def _first_line(text: object) -> str:
    line = next((" ".join(part.split()) for part in str(text or "").splitlines() if part.strip()), "")
    return line if len(line) <= FIRST_AID_CHARS else line[: FIRST_AID_CHARS - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="lookup_msds 등록 상태 확인")
    parser.add_argument("names", nargs="*", help="조회할 물질 이름 (비우면 기본 4개)")
    parser.add_argument("--require-real", action="store_true", help="mock이면 종료 코드 1")
    args = parser.parse_args(argv)
    names = args.names or DEFAULT_NAMES

    try:
        get_settings.cache_clear()
        settings = get_settings()
        agent_deps.get_tool_registry.cache_clear()
        registry = agent_deps.get_tool_registry()
    except (ValidationError, ValueError) as exc:
        print(f"설정 오류: {exc}")
        return 2

    tool = next((t for t in registry.tools() if t.name == "lookup_msds"), None)
    if tool is None:
        print("lookup_msds가 등록되어 있지 않습니다.")
        return 2
    kind = "mock (연습 데이터)" if tool.mock else "실제 (공단 MSDS API)"
    print(f"MSDS_TOOL={settings.msds_tool} → lookup_msds: {kind}, 담당 {tool.owner}")
    if args.require_real and tool.mock:
        print("실제 도구가 아닙니다. MSDS_TOOL=KOSHA와 KOSHA_MSDS_SERVICE_KEY를 확인하세요 (docker면 이미지 재빌드·--force-recreate).")
        return 1

    started = time.perf_counter()
    try:
        results = tool.handler({"materials": names})["results"]
    except ToolError as exc:
        print(f"조회 실패: {exc.code} - {exc.message}")
        return 1
    print(f"조회 {len(names)}건 {time.perf_counter() - started:.2f}초")
    for r in results:
        if not r.get("found"):
            note = f" ({r['lookup_error']})" if r.get("lookup_error") else ""
            print(f"  {r['query']}: 못 찾음{note}")
            continue
        print(f"  {r['query']}: 찾음 이름 '{r.get('name')}' CAS {r.get('cas_no')} | ghs {r.get('ghs_classes')} | 출처 {r.get('source')}")
        print(f"      응급조치 첫 줄: {_first_line(r.get('first_aid'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
