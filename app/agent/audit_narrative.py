"""P2 감사보고서 문장 후보 생성기(D).

집계, 비율 계산, 상태 판정 및 최종 문장 채택은 수행하지 않는다.
반환한 후보 문장은 B 구성기의 숫자 대조를 통과해야만 사용할 수 있다.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.evidence_collector.textutil import extract_json_object
from app.exceptions import LLMError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.audit_compose import ComposeItem


SYSTEM_PROMPT = """너는 감사대응 보고서의 항목별 문장 후보 작성기다.
입력된 항목 facts와 기준만 사용해 보고서에 넣을 한국어 문장 후보를 작성한다.

규칙:
- 집계, 계산, 비율 산출, 반올림을 하지 않는다.
- 입력 facts에 없는 숫자·날짜·기록을 추가하지 않는다.
- state가 UNAVAILABLE이면 확인 불가 사실만 간결하게 쓴다.
- 위반, 부적정 등 법적 판단을 새로 만들지 않는다.
- 문장은 입력 facts를 설명하는 데 필요한 내용만 쓴다.
- JSON 객체 하나만 출력한다.

출력 형식:
{"sentences":["문장 후보"]}
"""


def generate_sentence_candidates(item: ComposeItem, llm: LLMClient) -> list[str]:
    """ComposeItem의 facts를 LLM에 전달해 문장 후보를 만든다.

    이 함수는 후보를 반환할 뿐 최종 보고서에 채택하지 않는다. 호출자는 반드시
    B 구성기의 숫자 대조기를 통과시킨 뒤 후보를 채택하고, 실패 시 템플릿을 유지해야 한다.
    """
    payload: dict[str, Any] = {
        "area": item.area,
        "kind": item.kind,
        "state": item.state,
        "stateLabel": item.state_label,
        "unavailableReason": item.unavailable_reason,
        "basis": [basis.model_dump(by_alias=True) for basis in item.basis],
        "facts": item.facts.model_dump(by_alias=True),
        "templateSentences": [sentence.text for sentence in item.sentences],
        "actionStatus": (
            item.action_status.model_dump(by_alias=True)
            if item.action_status is not None
            else None
        ),
    }
    response = llm.chat(
        [Message(role="user", content=json.dumps(payload, ensure_ascii=False))],
        system=SYSTEM_PROMPT,
    )
    output = extract_json_object(response.text or "")
    if not isinstance(output, dict) or not isinstance(output.get("sentences"), list):
        raise LLMError("감사보고서 문장 후보 응답 형식이 올바르지 않습니다.", engine="AUDIT_REPORT")

    sentences = output["sentences"]
    if not sentences or any(not isinstance(sentence, str) or not sentence.strip() for sentence in sentences):
        raise LLMError("감사보고서 문장 후보가 비어 있거나 올바르지 않습니다.", engine="AUDIT_REPORT")
    return [sentence.strip() for sentence in sentences]
