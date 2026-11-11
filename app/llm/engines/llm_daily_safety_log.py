"""daily_safety_log 엔진 — 계획서 9장 시나리오 D · 주인: D

마감 현황 → TBM 기록 → 일일 안전일지 요약.
집계(SQL)가 끝난 숫자를 정해진 형식으로 요약하는 데 쓴다. 요약 속 수치는 코드가 집계와 대조한다 (계획서 7-6장).

이 파일만 고치면 이 시나리오의 모델이 바뀐다. 다른 시나리오에는 영향이 없다.
  1. SPEC    모델·장치 등. None인 칸은 llm_default.py의 값을 따른다. 최대 생성 길이는 엔진마다 정하지 않고 config.py의 ai_max_output_tokens 하나를 쓴다.
  2. 클래스  DailySafetyLogClient. 손질이 필요할 때만 아래 세 메서드 중 필요한 것의 주석을 풀어 쓴다. 메서드 이름은 그대로 둔다.
  3. 빌드    build_daily_safety_log_client. 설정을 합쳐 클라이언트를 만든다 (모델은 첫 chat() 때 읽는다).
이 파일에 다른 함수를 만들면 이름을 _로 시작한다 (테스트가 오타를 잡는다).
모델을 읽고 생성하는 코드는 LLMClient(llm_base.py, llm_transformers.py)가 맡는다. 여기서 transformers를 직접 부르지 않는다.
테스트: tests/test_scenario_engines.py의 fake_tf를 본떠, 모델을 부르지 않고 메서드를 확인한다.
"""

from __future__ import annotations

from typing import Any

from ..llm_base import LLMClient
from ._base import EngineSpec, build_config

ENGINE_NAME = "DAILY_SAFETY_LOG"  # app/core/config.py의 Engine 이름
OWNER = "D"

SYSTEM_PROMPT = """너는 산업현장 안전관리 시나리오 D의 요약 모델이다.
입력으로 전달된 Spring 공식 집계 결과만 사용해 TBM, 마감현황, 일일 안전일지를 요약한다.
건수·비율·날짜·ID를 추측하거나 새로 계산하지 않는다.
입력에 없는 내용은 '확인되지 않음'이라고 표시한다.
반드시 JSON 객체 하나만 출력한다.
형식:
{"summary":"짧은 종합 요약", "tbm":"TBM 요약", "deadline":"마감현황 요약", "dailySafetyLog":"안전일지 요약", "warnings":[], "evidence":[]}
"""

SPEC = EngineSpec(
    model_id=None,          # 예: "Qwen/Qwen3-1.7B" 또는 로컬 경로 "D:/models/qwen3-1.7b". None이면 default 모델
    device=None,            # 예: "cuda". None이면 default
    revision=None,          # 예: 허브 커밋 해시. None이면 default
    enable_thinking=None,   # None이면 default(False)
    local_files_only=None,  # True면 내려받지 않고 캐시·로컬 파일만. 폐쇄망이면 어차피 켜진다
)


class DailySafetyLogClient(LLMClient):
    name = ENGINE_NAME

    def prepare_messages(self, chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """요약 입력 뒤에 JSON 출력 규칙을 붙인다."""
        messages = [dict(message) for message in chat_messages]
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = (
                f"{messages[-1].get('content', '')}\n\n"
                "공식 집계값만 사용하고 JSON 객체 하나만 출력하라. "
                "summary, tbm, deadline, dailySafetyLog, warnings, evidence 필드를 포함하라."
            )
        return messages

    def generate_options(self) -> dict[str, Any]:
        return {"repetition_penalty": 1.05}

    def finish_text(self, text: str) -> str:
        """코드 펜스와 앞뒤 설명을 제거해 JSON 파서가 처리할 수 있게 한다."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else cleaned
        return cleaned


def build_daily_safety_log_client(*, settings: Any, model_id: str | None = None) -> DailySafetyLogClient:
    """엔진 등록표(engines/__init__.py)가 부른다. model_id는 시나리오 표의 값 (없으면 SPEC)."""
    return DailySafetyLogClient(build_config(SPEC, settings=settings, model_id=model_id, engine=ENGINE_NAME))
