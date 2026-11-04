"""문장 → 규칙 입력 추출 (C 담당): 프롬프트와 출력 JSON 스키마. 표준 라이브러리만 쓴다.

규칙 (B님께 드린 경로 요청서·계획서 7-4장과 같음)
  - 원문에 적힌 숫자만, 계산·환산·추정 금지. 단위가 원문에 없으면 그 항목은 뺀다.
  - evidence 는 원문을 한 글자도 바꾸지 않고 옮긴다 (Spring 이 원문 포함 여부로 걸러 낸다).
  - 확실하지 않으면 비운다. 판정(BLOCK·PASS)은 하지 않는다.
예시는 Spring 정답 문항(gold-v1.json)과 겹치지 않는 문장으로 둔다 (평가가 예시를 외운 것이 되지 않게).
"""

from __future__ import annotations

import json
from typing import Any

ITEM_NAMES: dict[str, str] = {
    "O2": "산소",
    "CO2": "이산화탄소(탄산가스)",
    "LEL": "가연성·인화성 가스(폭발하한)",
    "H2S": "황화수소",
    "CO": "일산화탄소",
    "NH3": "암모니아",
    "CL2": "염소",
}
WORK_TYPE_NAMES: dict[str, str] = {
    "HOT": "화기 작업 (용접·용단·그라인더·절단 등 불꽃·열이 나는 작업)",
    "CONFINED": "밀폐공간 작업 (탱크·맨홀·반응기·피트 내부 등에 들어가는 작업)",
    "HEIGHT": "고소 작업 (높이 2m 이상, 비계·사다리·지붕·고소작업대 위 작업)",
}

SYSTEM_PROMPT = """너는 산업현장 작업 문장에서 안전 판정에 쓸 값만 뽑는 추출기다. 판정은 하지 않는다.
반드시 JSON 객체 하나만 출력한다. 설명·마크다운·코드 펜스를 쓰지 않는다.

출력 형식:
{"measurements": [{"item": 항목코드, "value": 숫자, "unit": 단위코드, "evidence": 원문 일부}],
 "workTypes": [{"type": 작업코드, "evidence": 원문 일부}],
 "watcherName": 이름 또는 null, "watcherEvidence": 원문 일부 또는 null}

규칙:
1. value 는 원문에 적힌 숫자 그대로 쓴다. 계산·환산·반올림·추정을 하지 않는다.
2. unit 은 원문 단위로 고른다: "%"·"퍼센트"·"프로" → PCT, "ppm"·"피피엠" → PPM, "%LEL" → PCT_LEL. 원문에 단위가 없으면 그 측정값은 넣지 않는다.
3. evidence 는 원문에서 그 값(또는 작업)이 나온 부분을 한 글자도 바꾸지 않고 복사한다. evidence 안에 value 숫자가 그대로 있어야 한다.
4. item·type 은 아래 목록에 있는 코드만 쓴다.
5. watcherName 은 감시자·감시인·화재감시자로 지정된 사람 이름을 원문 그대로 쓴다. 없으면 null.
6. 확실하지 않으면 넣지 않는다. 빈 배열과 null 이 정상이다."""

# (입력 문장, 기대 출력)
FEW_SHOTS: list[tuple[str, dict[str, Any]]] = [
    (
        "B동 저장조 안에 들어가서 내부 도장합니다. 산소 재 보니 19.2% 나왔고 황화수소는 2 ppm",
        {
            "measurements": [
                {"item": "O2", "value": 19.2, "unit": "PCT", "evidence": "산소 재 보니 19.2%"},
                {"item": "H2S", "value": 2, "unit": "PPM", "evidence": "황화수소는 2 ppm"},
            ],
            "workTypes": [{"type": "CONFINED", "evidence": "저장조 안에 들어가서"}],
            "watcherName": None,
            "watcherEvidence": None,
        },
    ),
    (
        "옥상 배관 지지대 용접, 화재감시자는 박민수 씨가 맡아요. 가스는 아직 안 쟀어요",
        {
            "measurements": [],
            "workTypes": [{"type": "HOT", "evidence": "배관 지지대 용접"}],
            "watcherName": "박민수",
            "watcherEvidence": "화재감시자는 박민수",
        },
    ),
    (
        "펌프실 피트 점검인데 일산화탄소가 좀 높게 나온 것 같아요",
        {"measurements": [], "workTypes": [{"type": "CONFINED", "evidence": "펌프실 피트"}],
         "watcherName": None, "watcherEvidence": None},
    ),
]


def build_system_prompt(items: list[str], work_types: list[str]) -> str:
    item_lines = "\n".join(f"- {i}: {ITEM_NAMES.get(i, i)}" for i in items)
    type_lines = "\n".join(f"- {t}: {WORK_TYPE_NAMES.get(t, t)}" for t in work_types)
    shots = "\n\n".join(
        f"문장: {text}\n출력: {json.dumps(out, ensure_ascii=False)}" for text, out in FEW_SHOTS
    )
    return (f"{SYSTEM_PROMPT}\n\n측정 항목 코드:\n{item_lines}\n\n작업 코드:\n{type_lines}\n\n"
            f"예시 (마지막 예시처럼 숫자가 없으면 측정값을 넣지 않는다):\n\n{shots}")


def build_user_message(text: str) -> str:
    # Qwen3 의 생각 모드를 끈다 (/no_think). 켜져 있어도 extract_json_object 가 <think> 를 지운다
    return f"문장: {text}\n출력: /no_think"


def output_schema(items: list[str], work_types: list[str]) -> dict[str, Any]:
    """엔진이 json_schema 를 받으면 넘기는 출력 제약 (llama.cpp 문법 제약용)."""
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "enum": list(items)},
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["PCT", "PPM", "PCT_LEL"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["item", "value", "unit", "evidence"],
                },
            },
            "workTypes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(work_types)},
                        "evidence": {"type": "string"},
                    },
                    "required": ["type", "evidence"],
                },
            },
            "watcherName": nullable_str,
            "watcherEvidence": nullable_str,
        },
        "required": ["measurements", "workTypes", "watcherName", "watcherEvidence"],
    }
