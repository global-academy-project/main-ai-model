"""confined_space_gate 엔진 — 계획서 9장 시나리오 B · 주인: C

밀폐공간 작업 → 측정값 기준 미달이면 승인 차단 (문장 → 값 추출).
문장에서 측정값·작업유형·장소를 뽑는 값 추출(계획서 7-4장)에 쓴다. 판정은 규칙 엔진(check_rules)이 한다.
숫자를 추정하거나 바꾸지 않고, 확실하지 않으면 비워 두게 한다.

이 파일만 고치면 이 시나리오의 모델이 바뀐다. 다른 시나리오에는 영향이 없다.
  1. SPEC    모델·장치 등. None인 칸은 llm_default.py의 값을 따른다. 최대 생성 길이는 엔진마다 정하지 않고 config.py의 ai_max_output_tokens 하나를 쓴다.
  2. 클래스  ConfinedSpaceGateClient. 손질이 필요할 때만 아래 세 메서드 중 필요한 것의 주석을 풀어 쓴다. 메서드 이름은 그대로 둔다.
  3. 빌드    build_confined_space_gate_client. 설정을 합쳐 클라이언트를 만든다 (모델은 첫 chat() 때 읽는다).
이 파일에 다른 함수를 만들면 이름을 _로 시작한다 (테스트가 오타를 잡는다).
모델을 읽고 생성하는 코드는 LLMClient(llm_base.py, llm_transformers.py)가 맡는다. 여기서 transformers를 직접 부르지 않는다.
테스트: tests/test_scenario_engines.py의 fake_tf를 본떠, 모델을 부르지 않고 메서드를 확인한다.
"""

from __future__ import annotations

from typing import Any

from ..llm_base import LLMClient
from ._base import EngineSpec, build_config

ENGINE_NAME = "CONFINED_SPACE_GATE"  # app/core/config.py의 Engine 이름
OWNER = "C"

# 추출 생성 시간 제한 (generate_options 의 max_time). Spring 대기(APP_EXTRACT_LLM_TIMEOUT_SECONDS 기본 50초)보다 짧게.
# 11/12: 25 → 45초. 밀폐 측정값 5개 + 감시인만 적어도 CPU에서 약 27초라 잘렸다. 보통 문장은 평균 14초로 그대로
_MAX_SECONDS = 45.0

SPEC = EngineSpec(
    model_id=None,          # 예: "Qwen/Qwen3-1.7B" 또는 로컬 경로 "D:/models/qwen3-1.7b". None이면 default 모델
    device=None,            # 예: "cuda". None이면 default
    revision=None,          # 예: 허브 커밋 해시. None이면 default
    enable_thinking=None,   # None이면 default(False)
    local_files_only=None,  # True면 내려받지 않고 캐시·로컬 파일만. 폐쇄망이면 어차피 켜진다
)


class ConfinedSpaceGateClient(LLMClient):
    name = ENGINE_NAME
    # ── LoRA 어댑터 (11/10 추가) ──────────────────────────────────────────────────────────────
    # 같은 기본 모델을 쓰는 엔진끼리 기본 모델은 메모리에 한 번만 올리고, 요청마다 이 엔진의 어댑터로
    # 바꿔 생성한다 (llm_transformers.py 의 _use_adapter). LLM_ADAPTERS 에 "extract" 가 없으면 기본 모델로 생성한다 (지금과 같음).
    # 기본 모델 (11/11): Qwen3-0.6B. config.py 시나리오 표의 router·A 와 같은 이름이라 같은 한 벌을 쓴다.
    #   - 어댑터를 붙이면 그 한 벌이 peft 로 감싸지지만, 어댑터가 없는 엔진(router·A 등) 요청은 어댑터를 끄고 생성하므로
    #     그쪽 결과는 기본 모델과 같다.
    #   - 생성 잠금(_generate_lock)을 같이 쓰므로 추출이 길어지면 다른 엔진이 기다린다 → 아래 generate_options 의 시간 제한.
    #   - 어댑터는 학습한 기본 모델에만 붙는다. 1.7B 로 학습한 어댑터는 0.6B 에 붙지 않는다 (붙이다 실패하면 기본 모델로 답함).
    # 쓰는 순서:
    #   1) 학습: config.py 시나리오 표에 적힌 것과 같은 기본 모델·같은 버전으로 LoRA(또는 QLoRA) 학습.
    #      프롬프트는 실행 때와 같게 (chat 템플릿 system+user, enable_thinking=False).
    #   2) 저장: peft 의 model.save_pretrained("<폴더>") 로 어댑터만 저장한다. 기본 모델에 머지하지 않는다.
    #      폴더 안에 adapter_config.json, adapter_model.safetensors 두 파일이 있으면 된다.
    #   3) 배치: 그 폴더를 ai-model 서버(로컬 디스크)에 둔다. 폐쇄망이면 미리 복사해 둔다 (허브에서 내려받지 않는다).
    #   4) 설정: .env 의 LLM_ADAPTERS 에 "이름": "폴더" 를 더한다 (JSON 한 줄, 이름은 아래 adapter 와 같게).
    #      예) LLM_ADAPTERS={"extract": "/models/adapters/extract", "record_query": "/models/adapters/record_query"}
    #   5) 서버를 다시 띄운다 (아래 adapter 는 이미 적혀 있음).
    #   6) 확인: 이 엔진의 첫 요청 로그에 "LoRA 어댑터 <이름> 를 붙였습니다" 가 나오면 적용된 것.
    #      "폴더가 없어 기본 모델로 생성합니다" 가 나오면 4) 의 경로·이름을 확인한다 (오류로 멈추지 않고 기본 모델로 답한다).
    # 주의: peft 패키지가 설치돼 있어야 한다 (requirements). 엔진마다 다른 어댑터를 쓰면 이름도 달라야 한다.
    adapter = "extract"

    # ── 손질 지점 (필요할 때만 주석을 풀어 고친다. 기본 동작은 아무것도 하지 않는 것과 같다) ──
    #
    # def prepare_messages(self, chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    #     """모델에 넣기 전 메시지 손질. 새 목록을 돌려준다. 예: 마지막 user 메시지 끝에 "JSON 한 개만 출력" 덧붙이기"""
    #     return chat_messages
    #
    def generate_options(self) -> dict:
        """generate()에 더 넘길 인자 (max_new_tokens·do_sample은 넣을 수 없다).

        max_time: 생성 시간 제한(초). 넘으면 transformers 가 그 자리에서 멈춘다 → 어댑터(줄 형식)는 끝까지 쓴 줄만 쓰고,
        긴 프롬프트(JSON)는 JSON 이 덜 끝나 LLMError(502)
        → Spring 은 사전(규칙) 추출 결과로 보여 준다. 생성 잠금을 빨리 풀어 같은 모델을 쓰는 다른 엔진이 오래 기다리지 않게 한다.
        Spring 대기(APP_EXTRACT_LLM_TIMEOUT_SECONDS, 기본 50초)보다 짧게 둔다 (프롬프트 읽는 시간은 이 안에 들어간다).
        """
        return {"max_time": _MAX_SECONDS}
    #
    # def finish_text(self, text: str) -> str:
    #     """모델 출력 손질. 예: 모델이 붙이는 머리말 떼기"""
    #     return text


def build_confined_space_gate_client(*, settings: Any, model_id: str | None = None) -> ConfinedSpaceGateClient:
    """엔진 등록표(engines/__init__.py)가 부른다. model_id는 시나리오 표의 값 (없으면 SPEC)."""
    return ConfinedSpaceGateClient(build_config(SPEC, settings=settings, model_id=model_id, engine=ENGINE_NAME))
