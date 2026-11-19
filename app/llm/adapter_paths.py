"""LoRA 어댑터 폴더 찾기 (11/19, B). 계획 단계 어댑터 plan_brief 는 환경변수 LLM_ADAPTERS 로 폴더를 주고, 없으면 기본 위치를 본다.

시나리오 A(work_register) 계획 단계 어댑터:

  - 폴더: 환경변수 LLM_ADAPTERS (Settings.llm_adapters, JSON 한 줄)의 "plan_brief" 값. 어댑터 폴더 자체를 가리킨다.
      LLM_ADAPTERS={"extract": "/models/adapters/extract", "plan_brief": "/models/adapters/work_register"}
    "plan_brief" 가 없으면 기본 위치 <코드 루트>/adapters/work_register 를 본다 (PC 에서 환경변수 없이 쓰는 경우. 어댑터 파일은 git 에 올리지 않는다).
    docker 에서는 컨테이너 안 경로를 LLM_ADAPTERS 에 주고 그 경로로 마운트한다.
  - 폴더 안: adapter_config.json + 가중치(adapter_model.safetensors 또는 .bin). 둘 다 있으면 어댑터를 붙이고 계획 단계는 짧은 프롬프트를 쓴다.
    하나라도 없으면 어댑터 없이 긴 프롬프트다 (어댑터를 만들기 전과 같다, 서버는 정상 기동, 경고 로그).
    LLM_ADAPTERS 에 plan_brief 를 줬는데 파일이 없으면 기본 위치로 돌아가지 않는다 (경로를 잘못 준 것이 로그에 보이게).
  - 다른 이름(C 의 extract 등)은 LLM_ADAPTERS 값을 그대로 쓴다.

붙이는 장치는 새로 만들지 않았다: 이 모듈은 이름 -> 폴더 표만 만든다 (llm_transformers._adapter_paths, llm_work_register,
agent/plan_adapter.adapter_ready 가 같은 함수를 부른다). 어댑터는 학습한 기본 모델·판에만 붙는다 (계획 단계는 Qwen/Qwen3-0.6B).
끄기: BUILTIN_ADAPTERS 를 비우거나 llm_work_register.py 의 ADAPTER_ENABLED = False, 또는 어댑터 파일을 치운다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]  # <루트>/app/llm/adapter_paths.py -> <루트>

# 이름 -> 기본 폴더 (LLM_ADAPTERS 에 이 이름이 없을 때). 이름은 엔진 클래스의 adapter 와 짝이다 (계획 단계: llm_work_register.ADAPTER_NAME).
BUILTIN_ADAPTERS: Mapping[str, Path] = {"plan_brief": _REPO_ROOT / "adapters" / "work_register"}
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def has_adapter_files(folder: Path) -> bool:
    """폴더에 어댑터 설정 파일과 가중치 파일이 모두 있나."""
    return (folder / "adapter_config.json").is_file() and any((folder / name).is_file() for name in WEIGHT_FILES)


def adapter_dir(name: str, settings: Any) -> Path | None:
    """기본 위치가 있는 이름의 어댑터 폴더: LLM_ADAPTERS 에 값이 있으면 그 경로, 없으면 기본 위치. 그런 이름이 아니면 None."""
    if name not in BUILTIN_ADAPTERS:
        return None
    configured = str((getattr(settings, "llm_adapters", None) or {}).get(name) or "").strip()
    return Path(configured) if configured else BUILTIN_ADAPTERS[name]


def adapter_folders(settings: Any) -> dict[str, str]:
    """어댑터 이름 -> 폴더. 기본 위치가 있는 이름(plan_brief)은 adapter_dir 의 어댑터 파일이 있을 때만 더하고, 다른 이름은 LLM_ADAPTERS 그대로."""
    folders = {n: str(p) for n, p in (getattr(settings, "llm_adapters", None) or {}).items() if n not in BUILTIN_ADAPTERS}
    for name in BUILTIN_ADAPTERS:
        path = adapter_dir(name, settings)
        if path is not None and has_adapter_files(path):
            folders[name] = str(path)
    return folders
