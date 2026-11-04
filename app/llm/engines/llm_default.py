"""default 엔진 — 기본값 · 주인: 공용(B)

모든 시나리오 엔진의 기본값.
다른 시나리오 모듈의 SPEC에서 None인 칸은 여기 값을 쓴다. 이 값을 바꾸면 자기 값을 정하지 않은 모든 시나리오가 바뀐다.
시나리오 표에 줄이 없는 키가 쓰는 엔진이기도 하다.

이 파일을 고치면 SPEC을 비워 둔 모든 시나리오가 바뀐다. 바꾸기 전에 팀에 알린다.
엔진이 아니라 기본 설정만 가진다 (클라이언트 클래스·빌드 함수 없음). 여기는 기본값이라 모든 칸을 채워 둔다.
"""

from __future__ import annotations

from ._base import EngineSpec

OWNER = "공용(B)"

SPEC = EngineSpec(
    model_id="Qwen/Qwen3-0.6B",  # 허브 이름 또는 로컬 경로. 폐쇄망이면 미리 받아 둔 경로를 적는다
    device="cpu",                # "cpu" | "cuda" | "cuda:0" | "mps" | "auto"(accelerate 필요)
    revision=None,               # 허브 커밋 해시 (재현성을 위해 고정 권장)
    enable_thinking=False,       # Qwen3 <think> 모드. 끄면 짧은 JSON이 잘리지 않는다
    local_files_only=None,
)
