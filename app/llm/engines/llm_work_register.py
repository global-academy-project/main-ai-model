"""work_register 엔진 — 계획서 9장 시나리오 A · 주인: B

짧은 요청 → 되묻기 → 아침 등록 항목(위험성평가·허가·점검표·TBM) 초안.
작업 등록 에이전트(WorkRegisterAgent)가 근거 검색 계획과 위험요인 판단 두 곳에서 부른다 (VERIFY 현장).
출력은 JSON이다. 수치 판정은 규칙 엔진이 하므로 모델은 후보만 낸다.

이 파일만 고치면 이 시나리오의 모델이 바뀐다. 다른 시나리오에는 영향이 없다.
  1. SPEC    모델·장치 등. None인 칸은 llm_default.py의 값을 따른다. 최대 생성 길이는 엔진마다 정하지 않고 config.py의 ai_max_output_tokens 하나를 쓴다.
  2. 클래스  WorkRegisterClient. 손질이 필요할 때만 아래 세 메서드 중 필요한 것의 주석을 풀어 쓴다. 메서드 이름은 그대로 둔다.
  3. 빌드    build_work_register_client. 설정을 합쳐 클라이언트를 만든다 (모델은 첫 chat() 때 읽는다).
이 파일에 다른 함수를 만들면 이름을 _로 시작한다 (테스트가 오타를 잡는다).
모델을 읽고 생성하는 코드는 LLMClient(llm_base.py, llm_transformers.py)가 맡는다. 여기서 transformers를 직접 부르지 않는다.
테스트: tests/test_scenario_engines.py의 fake_tf를 본떠, 모델을 부르지 않고 메서드를 확인한다.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import adapter_paths
from ..llm_base import LLMClient
from ._base import EngineSpec, build_config

logger = logging.getLogger(__name__)

ENGINE_NAME = "WORK_REGISTER"  # app/core/config.py의 Engine 이름
OWNER = "B"

# 계획 단계 LoRA 어댑터 (11/17, 11/18 갱신). 어댑터 파일이 있을 때만 클라이언트에 어댑터 이름을 붙인다 (build_work_register_client).
# 폴더는 환경변수 LLM_ADAPTERS 의 "work_register" (Settings.llm_adapters)이고, 없으면 기본 위치 adapters/work_register (app/llm/adapter_paths.py).
# 어댑터 파일(adapter_config.json + 가중치)이 있을 때만 붙인다.
# 어댑터 파일이 없으면 지금과 같다 (어댑터 없음, 긴 프롬프트, 경고 로그). 쓰는 순서와 짧은 프롬프트의 원본은 app/agent/plan_adapter.py.
# 어댑터는 학습한 기본 모델(같은 model_id·같은 버전)에만 붙는다. 이 엔진의 모델을 바꾸면 어댑터도 다시 학습한다.
ADAPTER_NAME = "work_register"
ADAPTER_ENABLED = True  # False면 어댑터 파일이 있어도 붙이지 않는다

SPEC = EngineSpec(
    model_id=None,          # 예: "Qwen/Qwen3-1.7B" 또는 로컬 경로 "D:/models/qwen3-1.7b". None이면 default 모델
    device=None,            # 예: "cuda". None이면 default
    revision=None,          # 예: 허브 커밋 해시. None이면 default
    enable_thinking=None,   # None이면 default(False)
    local_files_only=None,  # True면 내려받지 않고 캐시·로컬 파일만. 폐쇄망이면 어차피 켜진다
)


class WorkRegisterClient(LLMClient):
    name = ENGINE_NAME

    # ── 손질 지점 (필요할 때만 주석을 풀어 고친다. 기본 동작은 아무것도 하지 않는 것과 같다) ──
    #
    # def prepare_messages(self, chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    #     """모델에 넣기 전 메시지 손질. 새 목록을 돌려준다. 예: 마지막 user 메시지 끝에 "JSON 한 개만 출력" 덧붙이기"""
    #     return chat_messages
    #
    # def generate_options(self) -> dict:
    #     """generate()에 더 넘길 인자. 예: {"repetition_penalty": 1.1} (max_new_tokens·do_sample은 넣을 수 없다)"""
    #     return {}
    #
    # def finish_text(self, text: str) -> str:
    #     """모델 출력 손질. 예: 모델이 붙이는 머리말 떼기"""
    #     return text


def build_work_register_client(*, settings: Any, model_id: str | None = None) -> WorkRegisterClient:
    """엔진 등록표(engines/__init__.py)가 부른다. model_id는 시나리오 표의 값 (없으면 SPEC)."""
    client = WorkRegisterClient(build_config(SPEC, settings=settings, model_id=model_id, engine=ENGINE_NAME))
    if ADAPTER_ENABLED:
        folder = adapter_paths.adapter_folders(settings).get(ADAPTER_NAME)  # LLM_ADAPTERS 의 work_register (없으면 adapters/work_register)에 어댑터 파일이 있을 때만
        if folder:
            client.adapter = ADAPTER_NAME  # 이 인스턴스만. router·B·E·F 는 같은 기본 모델에서 어댑터를 끄고 생성한다
            logger.info("계획 어댑터 %s 를 씁니다 (짧은 프롬프트). 폴더=%s", ADAPTER_NAME, folder)
        else:
            logger.warning("계획 어댑터 파일이 없어 어댑터 없이 긴 프롬프트로 계속합니다. 찾은 곳: %s (adapter_config.json + adapter_model.safetensors). 환경변수 LLM_ADAPTERS 의 \"work_register\" 로 바꾼다",
                           adapter_paths.adapter_dir(ADAPTER_NAME, settings))
    return client
