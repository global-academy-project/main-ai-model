"""LoRA 어댑터 폴더 찾기 (11/18, B). 환경변수는 쓰지 않는다: 폴더에 어댑터 파일이 있는지로만 정한다.

시나리오 A(work_register) 계획 단계 어댑터의 내장 위치:

  - 폴더 (코드 루트 기준): adapters/work_register/   (adapter_config.json + adapter_model.safetensors)
    PC 에서는 저장소 루트 아래, docker 에서는 코드 루트(예: /app) 아래의 같은 상대 위치다.
  - 어댑터 파일은 git 에 올리지 않는다 (팀 규칙). 직접 복사하거나 마운트한다.
  - 두 파일(adapter_config.json, 가중치 adapter_model.safetensors 또는 .bin)이 다 있으면 어댑터를 붙이고 계획 단계는 짧은 프롬프트를 쓴다.
    하나라도 없으면 어댑터 없이 긴 프롬프트다 (어댑터를 만들기 전과 같다, 서버는 정상 기동).
  - 이 이름(plan_brief)은 LLM_ADAPTERS 를 보지 않는다 (있어도 무시). 다른 이름(C 의 extract 등)은 LLM_ADAPTERS 그대로.

붙이는 장치는 새로 만들지 않았다: 이 모듈은 이름 -> 폴더 표만 만든다 (llm_transformers._adapter_paths, llm_work_register,
agent/plan_adapter.adapter_ready 가 같은 함수를 부른다). 어댑터는 학습한 기본 모델·판에만 붙는다 (계획 단계는 Qwen/Qwen3-0.6B).
끄기: BUILTIN_ADAPTERS 를 비우거나 llm_work_register.py 의 ADAPTER_ENABLED = False, 또는 폴더를 치운다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]  # <루트>/app/llm/adapter_paths.py -> <루트>

# 이름 -> 폴더. 이름은 엔진 클래스의 adapter 와 짝이다 (계획 단계: llm_work_register.ADAPTER_NAME).
BUILTIN_ADAPTERS: Mapping[str, Path] = {"plan_brief": _REPO_ROOT / "adapters" / "work_register"}
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def has_adapter_files(folder: Path) -> bool:
    """폴더에 어댑터 설정 파일과 가중치 파일이 모두 있나."""
    return (folder / "adapter_config.json").is_file() and any((folder / name).is_file() for name in WEIGHT_FILES)


def adapter_folders(settings: Any) -> dict[str, str]:
    """어댑터 이름 -> 폴더. 내장 이름은 폴더의 어댑터 파일로만 정하고 settings.llm_adapters(LLM_ADAPTERS)는 보지 않는다. 다른 이름은 LLM_ADAPTERS 그대로."""
    folders = {n: str(p) for n, p in (getattr(settings, "llm_adapters", None) or {}).items() if n not in BUILTIN_ADAPTERS}
    folders.update({n: str(p) for n, p in BUILTIN_ADAPTERS.items() if has_adapter_files(p)})
    return folders
