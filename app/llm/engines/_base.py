"""시나리오 엔진 공용 부품 (주인: B). 팀원은 이 파일이 아니라 자기 `llm_<시나리오>.py`만 고친다.

시나리오 엔진 모듈(llm_<키>.py)은 각자 엔진 이름(`ENGINE_NAME`, config.py의 `Engine`과 같은 대문자), 그 이름을 딴 클라이언트
클래스(`WorkRegisterClient` 등, `LLMClient` 상속), 빌드 함수(`build_work_register_client`)를 가진다. 기본 chat()은 LLMClient가
transformers로 생성하고, 모듈은 SPEC(설정)과 손질 메서드만 고친다. router만 분류 전용 `CategorizatorClient`를 상속한다.
등록표(`engines/__init__.py`)는 `build_<키>_client`를 이름으로 찾아 부른다.

설정을 정하는 순서 (앞에서 정해지면 뒤는 보지 않는다):
  1. model_id 인자          시나리오 표의 model_id
  2. 그 시나리오 모듈의 SPEC  llm_work_register.py 등. None인 칸은 다음으로
  3. default 모듈의 SPEC     llm_default.py (모든 시나리오의 기본값)
  4. 서버 설정               CATEGORIZATOR_DEVICE 등 (Settings에 필드가 있을 때만)
  5. 코드 기본값             cpu, 추론 끔
최대 생성 길이(max_new_tokens)는 엔진마다 정하지 않는다: 모든 엔진이 `Settings.ai_max_output_tokens`(AI_MAX_OUTPUT_TOKENS)를 쓴다.
NETWORK_ENV=CLOSED이면 local_files_only를 강제로 켠다 (허브에서 내려받지 않고 로컬 캐시·디렉터리만).
같은 설정(모델·장치·판·토큰·추론)이면 시나리오가 달라도 모델을 한 번만 메모리에 올린다 (llm_transformers 캐시).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.exceptions import LLMConfigError

from ..llm_transformers import TransformersConfig

# 시나리오 모듈이 이 이름으로 둘 수 있는 손질 함수. 그 밖의 함수는 _로 시작해야 한다 (오타가 조용히 무시되지 않게).
HOOK_NAMES: tuple[str, ...] = ("prepare_messages", "generate_options", "finish_text")


@dataclass(frozen=True)
class EngineSpec:
    """시나리오 엔진 설정. None인 칸은 default 엔진(llm_default.py)의 값을 따른다."""

    model_id: str | None = None          # 허브 저장소 이름(예: "Qwen/Qwen3-0.6B") 또는 로컬 디렉터리 경로
    device: str | None = None            # "cpu" | "cuda" | "cuda:0" | "mps" | "auto"(accelerate 필요)
    revision: str | None = None          # 허브 커밋 해시 (재현성을 위해 고정 권장)
    enable_thinking: bool | None = None  # Qwen3 등 <think> 모드. False 권장 (짧은 JSON이 잘리지 않게)
    local_files_only: bool | None = None  # True면 내려받지 않고 캐시·로컬 파일만


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_config(
    spec: EngineSpec,
    default: EngineSpec,
    *,
    settings: Any,
    model_id: str | None = None,
    engine: str,
) -> TransformersConfig:
    """설정 순서(이 파일 docstring)대로 TransformersConfig를 만든다."""
    resolved_model = _first(model_id, spec.model_id, default.model_id, getattr(settings, "categorizator_model_id", None))
    if not resolved_model or not str(resolved_model).strip():
        raise LLMConfigError(f"{engine} 엔진의 model_id가 없습니다 (llm_{engine.lower()}.py 또는 llm_default.py의 SPEC).", engine=engine)
    max_new_tokens = getattr(settings, "ai_max_output_tokens", None)  # 모든 엔진이 같은 값 (config.py의 ai_max_output_tokens)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise LLMConfigError(f"{engine} 엔진: AI_MAX_OUTPUT_TOKENS는 1 이상의 정수여야 합니다.", engine=engine)
    local_only = bool(_first(spec.local_files_only, default.local_files_only,
                             getattr(settings, "categorizator_local_files_only", None), False))
    if getattr(settings, "network_env", "OPEN") == "CLOSED":
        local_only = True
    return TransformersConfig(
        model_id=str(resolved_model).strip(),
        max_new_tokens=max_new_tokens,
        device=_first(spec.device, default.device, getattr(settings, "categorizator_device", None), "cpu"),
        local_files_only=local_only,
        revision=_first(spec.revision, default.revision, getattr(settings, "categorizator_revision", None)),
        enable_thinking=_first(spec.enable_thinking, default.enable_thinking,
                               getattr(settings, "categorizator_enable_thinking", None), False),
    )


def build_config(spec: EngineSpec, *, settings: Any, model_id: str | None = None, engine: str) -> TransformersConfig:
    """각 모듈의 build_<키>_client()가 부른다. default 모듈(llm_default.py)의 SPEC을 기본값으로 설정 순서대로 합친다.

    engine: 그 모듈의 ENGINE_NAME (예: "WORK_REGISTER"). 오류 메시지의 파일 이름은 소문자로 만든다.
    """
    from .llm_default import SPEC as DEFAULT_SPEC

    return resolve_config(spec, DEFAULT_SPEC, settings=settings, model_id=model_id, engine=engine)
