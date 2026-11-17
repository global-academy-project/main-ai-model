"""lookup_msds 실제 도구 준비 (MSDS_TOOL=KOSHA일 때만).

A의 공단 MSDS 클라이언트(app/publicdata/msds.py, KoshaMsdsClient)를 고치지 않고 바깥에서 감싸
(handler, parameters)로 돌려준다. tool_setup.build_tool_registry(msds_tool=...)가 이것으로 mock을 바꿔 등록한다.

- MOCK(기본)이면 None을 돌려주고 A의 모듈은 임포트하지 않는다 (A 모듈이 깨져 있어도 서버는 뜬다).
- KOSHA인데 키가 없으면 기동 때 Settings가 막는다. 여기서도 한 번 더 확인한다 (조용히 mock으로 남지 않게).
- 키는 Settings(환경변수 KOSHA_MSDS_SERVICE_KEY)에서만 받는다. 로그·응답에 쓰지 않는다.

래퍼가 하는 일 (11/16 PC 실측 40건에 맞춤, 설계: 시나리오A_MSDS_실제도구_등록_설계_v0.md)
 1. 표준 이름 → 공단에서 실제로 찾히는 조회어 (LOOKUP_TERMS). 사전 표기("메탄올")와 공단 이름("메틸 알코올")이 달라
    그대로는 못 찾는 물질이 있다. 표에 없는 이름은 그대로 조회한다.
 2. 물질마다 따로 조회한다. 한 물질이 실패해도 찾은 물질은 버리지 않는다 (실패한 물질은 found=false + lookup_error).
    조회를 시도한 물질이 모두 실패했을 때만 ToolError를 그대로 올린다 (에이전트가 MSDS_UNAVAILABLE로 기록).
 3. 전체 시간 상한(BUDGET_SECONDS). 넘으면 남은 물질은 found=false + lookup_error=MSDS_BUDGET_EXCEEDED.
 4. 결과의 query를 **요청받은 이름으로 되돌린다**. 에이전트는 query로 물질을 구분한다 (name은 공단 이름 그대로).
 5. ghs_classes 정리. 공단 값은 "라벨 : 구분|라벨 : 구분|…" 한 덩어리(+"자료없음")라, 그대로 두면 허가서 category와
    사고사례 검색 질의에 덩어리째 들어간다. mock과 같은 모양(라벨 목록)으로 풀고 원래 값은 ghs_detail에 남긴다.
    MsdsDto는 ghs_detail을 무시하지만 steps 기록에는 남는다.

실패하면 mock으로 대신하지 않는다 (가짜 값이 초안에 섞인다). 못 찾으면 MSDS_NOT_FOUND로 담당자가 확인한다.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from app.exceptions import ToolError

if TYPE_CHECKING:
    from app.core import Settings

logger = logging.getLogger(__name__)

CALL_TIMEOUT_SECONDS = 5.0  # 공단 API 호출 한 번의 제한 (실측 최대 0.6초). A의 기본값은 10초
BUDGET_SECONDS = 25.0  # 한 번의 lookup_msds 전체 시간 상한. Spring 제한 120초 중 계획 LLM(22~35초)과 검색이 쓰고 남는 쪽을 지킨다
MAX_MATERIALS = 20  # 클라이언트·MSDS_PARAMETERS와 같다
BUDGET_CODE = "MSDS_BUDGET_EXCEEDED"
TOOL_NAME = "lookup_msds"

# 표준 이름(app/agent/text_rules.MATERIALS의 키) → 공단에서 실제로 찾힌 조회어. 값이 이름이면 이름으로, CAS면 CAS로 조회한다.
# 11/16 PC 실측(probe_msds)으로 찾힌 것만 둔다. 한글 이름으로 찾히는 것은 한글 이름을 쓴다:
#   같은 CAS에 한글 항목과 영문 항목이 따로 있어(크실렌/Xylene, 톨루엔/Toluene) CAS로 조회하면 영문 항목이 걸릴 수 있다.
# CAS로 찾은 것(11/16 2차 실측, 이름으로는 못 찾음): 프로판→"프로페인", 메탄→"메테인", 질소→"질소(NITROGEN)", 경유→"디젤 연료".
#   경유는 공단에 "디젤 연료 2번"(68476-34-6) 등 다른 항목도 있다. 가장 일반적인 CAS(68334-30-5)를 쓰되 확인이 필요한 값이다.
# 표에 없는 것: 시너(혼합물이라 못 찾음이 맞다).
LOOKUP_TERMS: dict[str, str] = {
    "톨루엔": "톨루엔",
    "메탄올": "메틸알코올",
    "에탄올": "에탄올",
    "아세톤": "아세톤",
    "벤젠": "벤젠",
    "자일렌": "크실렌",
    "헥산": "헥산",
    "이소프로필알코올": "이소프로필알코올",
    "수소": "수소",
    "아세틸렌": "아세틸렌",
    "황화수소": "황화수소",
    "암모니아": "암모니아",
    "염소": "염소",
    "휘발유": "가솔린",
    "프로판": "74-98-6",
    "메탄": "74-82-8",
    "질소": "7727-37-9",
    "경유": "68334-30-5",
}

# "라벨 : 값" 구분. 라벨 안에 콜론이 올 수 있다
# (예: "급성 독성(흡입: 가스) : 구분2"). 공백이 앞에 붙은 콜론만 구분으로 본다.
_SEPARATOR = re.compile(r"\s:(?:\s|$)")
_NOT_APPLICABLE = ("해당없음", "자료없음", "분류되지않음", "분류안됨")


def ghs_labels(detail: object) -> list[str]:
    """공단 ghs_classes(한 덩어리 문자열 목록)를 mock과 같은 라벨 목록으로 푼다.

    '|'로 나눠 '라벨 : 값'이 온전한 항목의 라벨만 남긴다. 값이 비었거나(잘린 끝), 구분이 없거나("만성", "자료없음"),
    값이 '해당없음'류이면 버린다. 순서는 유지하고 같은 라벨은 한 번만 쓴다.
    """
    if not isinstance(detail, list):
        return []
    labels: list[str] = []
    for entry in detail:
        if not isinstance(entry, str):
            continue
        for part in entry.split("|"):
            part = part.strip()
            match = _SEPARATOR.search(part)
            if match is None:
                continue
            label, value = part[: match.start()].strip(), part[match.end():].strip()
            if not label or not value:
                continue
            if any(word in "".join(value.split()) for word in _NOT_APPLICABLE):
                continue
            if label not in labels:
                labels.append(label)
    return labels


def _found(raw: Mapping[str, Any], original: str) -> dict[str, Any]:
    result = dict(raw)
    result["query"] = original
    detail = raw.get("ghs_classes")
    result["ghs_detail"] = [str(item) for item in detail] if isinstance(detail, list) else []
    result["ghs_classes"] = ghs_labels(detail)
    return result


def make_msds_lookup(
    client: Any,
    *,
    lookup_terms: Mapping[str, str] | None = None,
    budget_seconds: float = BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """client.lookup({"materials": [조회어]})를 물질마다 불러 결과를 정리하는 handler를 만든다."""
    terms = LOOKUP_TERMS if lookup_terms is None else lookup_terms

    def lookup(arguments: Mapping[str, Any]) -> dict[str, Any]:
        materials = arguments.get("materials")
        if not isinstance(materials, list) or not all(isinstance(item, str) for item in materials):
            raise ToolError(
                "TOOL_INPUT_INVALID",
                "lookup_msds에는 물질 이름 목록(materials)이 필요합니다.",
                tool=TOOL_NAME,
            )
        if len(materials) > MAX_MATERIALS:
            raise ToolError(
                "TOOL_INPUT_INVALID",
                f"물질은 한 번에 {MAX_MATERIALS}개까지 조회할 수 있습니다.",
                tool=TOOL_NAME,
            )

        deadline = clock() + budget_seconds
        # 같은 이름은 한 번만 부른다: ("ok", 원본 결과) | ("error", 코드)
        memo: dict[str, tuple[str, Any]] = {}
        attempted = failed = 0
        last_error: ToolError | None = None
        results: list[dict[str, Any]] = []

        for original in materials:
            name = original.strip()
            if not name:
                results.append({"query": original, "found": False})
                continue
            key = name.casefold()
            if key not in memo:
                if clock() >= deadline:
                    memo[key] = ("error", BUDGET_CODE)
                else:
                    attempted += 1
                    try:
                        lookup_term = terms.get(name, name)
                        response = client.lookup({"materials": [lookup_term]})
                        memo[key] = ("ok", response["results"][0])
                    except ToolError as exc:
                        failed += 1
                        last_error = exc
                        logger.warning("MSDS 조회 실패: code=%s", exc.code)
                        memo[key] = ("error", exc.code)
            state, value = memo[key]
            if state == "error":
                results.append({"query": original, "found": False, "lookup_error": value})
            elif value.get("found"):
                results.append(_found(value, original))
            else:
                results.append({"query": original, "found": False})

        if attempted and failed == attempted and last_error is not None:
            raise last_error  # 시도한 조회가 모두 실패 → 못 찾음이 아니라 조회 실패로 알린다
        return {"results": results}

    return lookup


def build_msds_tool(
    settings: "Settings",
) -> tuple[Callable[[Mapping[str, Any]], Any], dict[str, Any]] | None:
    """KOSHA이면 (handler, parameters), MOCK이면 None."""
    if settings.msds_tool != "KOSHA":
        return None
    secret = settings.kosha_msds_service_key
    key = secret.get_secret_value().strip() if secret is not None else ""
    if not key:
        raise ValueError("MSDS_TOOL=KOSHA에는 KOSHA_MSDS_SERVICE_KEY가 필요합니다.")

    # 지연 임포트: KOSHA일 때만 A의 모듈을 불러온다
    from app.publicdata.msds import MSDS_PARAMETERS, KoshaMsdsClient

    client = KoshaMsdsClient(key, timeout_seconds=CALL_TIMEOUT_SECONDS)
    logger.info("MSDS 도구: KOSHA (조회어 표 %d개, 시간 상한 %.0f초)", len(LOOKUP_TERMS), BUDGET_SECONDS)
    return make_msds_lookup(client), MSDS_PARAMETERS
