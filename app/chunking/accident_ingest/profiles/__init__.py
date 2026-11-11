"""데이터셋 프로필 레지스트리. 새 데이터셋을 더하려면 profiles/<이름>.py를 만들고 여기에 등록한다."""

from __future__ import annotations

from types import ModuleType

from . import sif

PROFILES: dict[str, ModuleType] = {"sif": sif}


def get_profile(name: str) -> ModuleType:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"알 수 없는 프로필: {name} (있는 것: {', '.join(PROFILES)})") from None
