"""near_miss_warning 엔진 — 계획서 9장 시나리오 C · 주인: A

아차사고 보고 → 5축 태깅 → 같은 구역 다음 고소작업에 경고.
아차사고 5축 태깅(심각도·발생형태·위험요인·작업유형·관련 물질)에 쓴다. 허용 값은 코드(태그 코드표)가 정한다.

이 파일만 고치면 이 시나리오의 모델이 바뀐다. 다른 시나리오에는 영향이 없다.
  1. SPEC    모델·장치 등. None인 칸은 llm_default.py의 값을 따른다. 최대 생성 길이는 엔진마다 정하지 않고 config.py의 ai_max_output_tokens 하나를 쓴다.
  2. 클래스  NearMissWarningClient. 손질이 필요할 때만 아래 세 메서드 중 필요한 것의 주석을 풀어 쓴다. 메서드 이름은 그대로 둔다.
  3. 빌드    build_near_miss_warning_client. 설정을 합쳐 클라이언트를 만든다 (모델은 첫 chat() 때 읽는다).
이 파일에 다른 함수를 만들면 이름을 _로 시작한다 (테스트가 오타를 잡는다).
모델을 읽고 생성하는 코드는 LLMClient(llm_base.py, llm_transformers.py)가 맡는다. 여기서 transformers를 직접 부르지 않는다.
테스트: tests/test_scenario_engines.py의 fake_tf를 본떠, 모델을 부르지 않고 메서드를 확인한다.
"""

from __future__ import annotations

from typing import Any

from ..llm_base import LLMClient
from ._base import EngineSpec, build_config

ENGINE_NAME = "NEAR_MISS_WARNING"  # app/core/config.py의 Engine 이름
OWNER = "A"

SPEC = EngineSpec(
    model_id=None,          # 예: "Qwen/Qwen3-1.7B" 또는 로컬 경로 "D:/models/qwen3-1.7b". None이면 default 모델
    device=None,            # 예: "cuda". None이면 default
    revision=None,          # 예: 허브 커밋 해시. None이면 default
    enable_thinking=None,   # None이면 default(False)
    local_files_only=None,  # True면 내려받지 않고 캐시·로컬 파일만. 폐쇄망이면 어차피 켜진다
)


class NearMissWarningClient(LLMClient):
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


def build_near_miss_warning_client(*, settings: Any, model_id: str | None = None) -> NearMissWarningClient:
    """엔진 등록표(engines/__init__.py)가 부른다. model_id는 시나리오 표의 값 (없으면 SPEC)."""
    return NearMissWarningClient(build_config(SPEC, settings=settings, model_id=model_id, engine=ENGINE_NAME))
