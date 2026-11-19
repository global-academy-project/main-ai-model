"""계획 단계 LoRA 어댑터 쓰기 (11/17, B).

계획 LLM(`plan_brief.BriefPlanner`)은 작업 내용에서 summary·equipment·materials 세 키만 뽑는다.
학습한 어댑터를 붙이면 긴 지시문과 예시가 필요 없어서 프롬프트가 짧아진다 (395 → 약 100토큰 추정).

이 모듈이 하는 일은 둘이다.
  1. 짧은 프롬프트의 **원본**을 둔다 (`ADAPTER_SYSTEM_PROMPT`, `build_adapter_input`).
     서버와 학습 데이터 생성기가 같은 것을 불러 쓴다. 학습 입력과 서버 입력이 한 글자라도 다르면 어댑터가 제 값을 못 낸다.
     그래서 이 문구·조립을 바꾸면 어댑터를 다시 학습해야 한다.
  2. 짧은 프롬프트를 **언제 쓸지** 정한다 (`adapter_ready`). 어댑터가 실제로 붙을 수 있을 때만이다.
     어댑터 없이 짧은 프롬프트가 나가면 품질이 소리 없이 떨어지기 때문이다.

어댑터를 붙이는 장치는 새로 만들지 않는다. 이미 있는 것을 그대로 쓴다.
  - 엔진: app/llm/engines/llm_work_register.py 가 어댑터 파일이 있을 때만 클라이언트에 `ADAPTER_NAME` 을 붙인다.
  - 붙이기·전환·끄기: app/llm/llm_transformers.py 의 `_attach_adapter`·`_use_adapter` (C). 어댑터가 없는 엔진(router·B·E 등)은
    같은 기본 모델을 쓰되 어댑터를 끄고 생성한다.
  - 설정: 환경변수 LLM_ADAPTERS 의 "work_register" (Settings.llm_adapters), 없으면 adapters/work_register (app/llm/adapter_paths.py). 그 폴더에 어댑터 파일이 있는지로 정한다.
  - "어댑터 폴더가 있으면 짧은 프롬프트" 판정은 app/services/rule_input_extract_service.py 의 `adapter_ready` (C)와 같은 방식이다.
    agent가 services를 부르면 순환 import가 생길 수 있어서 같은 규칙을 여기에 따로 두었다.

끄기: llm_work_register.py 의 `ADAPTER_ENABLED = False`, 또는 어댑터 파일을 치운다 (LLM_ADAPTERS 의 폴더 또는 adapters/work_register).
둘 다 변경 전과 같다 (긴 프롬프트, 어댑터 없음).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.llm.engines.llm_work_register import ADAPTER_NAME

logger = logging.getLogger(__name__)

# 어댑터를 쓸 때 시스템 프롬프트. 키 이름 세 개만 알려 준다. 규칙(부정·조건부·추측 금지)과 예시는 학습이 가르친다.
ADAPTER_SYSTEM_PROMPT = "작업 내용에서 summary, equipment, materials를 JSON 객체 하나로 뽑으세요."


def build_adapter_input(work: Any) -> str:
    """사용자 메시지. 작업 내용과 슬롯이 이미 찾은 주변 물질·투입 장비만 싣는다.

    장소·화기 종류·인원·종료 시간은 세 키 추출에 쓰이지 않아 뺀다 (기본 프롬프트의 build_planner_input 과 다른 점).
    work: WorkContext (text, nearby_materials, equipment).
    """
    lines = [f"작업 내용: {work.text.strip()}"]
    if work.nearby_materials:
        lines.append(f"주변 물질: {', '.join(work.nearby_materials)}")
    if work.equipment:
        lines.append(f"투입 장비: {', '.join(work.equipment)}")
    return "\n".join(lines)


def adapter_ready(llm: Any) -> bool:
    """이 클라이언트에 계획 어댑터를 실제로 쓸 수 있나. True일 때만 짧은 프롬프트를 쓴다.

    - 클라이언트에 어댑터 이름이 붙어 있고 (엔진이 어댑터 파일이 있을 때만 붙인다),
    - 그 이름의 폴더에 adapter_config.json 이 있고,
    - 이미 붙이기에 실패한 적이 없을 때.
    첫 요청에서 붙이기에 실패하면(예: peft 없음, 기본 모델 불일치) 그 요청은 짧은 프롬프트로 기본 모델에 나간다.
    실패가 기록된 뒤의 요청부터는 긴 프롬프트로 돌아간다.
    """
    name = getattr(llm, "adapter", None)
    if not name or name != ADAPTER_NAME:
        return False
    try:
        from app.core import get_settings
        from app.llm.adapter_paths import adapter_folders

        path = adapter_folders(get_settings()).get(name)  # LLM_ADAPTERS 의 work_register (없으면 adapters/work_register)의 어댑터 파일로 정한다
    except Exception:  # noqa: BLE001 - 설정을 못 읽으면 어댑터 없이 (긴 프롬프트)
        return False
    if not path or not (Path(path) / "adapter_config.json").is_file():
        return False
    if _attach_failed(llm, name):
        logger.warning("계획 어댑터 %s 를 붙이지 못한 적이 있어 긴 프롬프트로 돌아갑니다.", name)
        return False
    return True


def _attach_failed(llm: Any, name: str) -> bool:
    """엔진이 이 어댑터를 붙이다 실패해 건너뛰기로 했나 (llm_transformers 의 모델 캐시를 읽기만 한다).
    캐시 구조를 읽지 못하면 실패하지 않은 것으로 본다."""
    try:
        from app.llm import llm_transformers

        loaded = llm_transformers._cache.get(llm.config)
        return loaded is not None and name in loaded.skipped
    except Exception:  # noqa: BLE001
        return False
