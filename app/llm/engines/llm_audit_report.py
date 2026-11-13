"""감사보고서 요약 엔진. SQL 집계와 수치 계산은 수행하지 않는다."""

from __future__ import annotations

from typing import Any

from ..llm_base import LLMClient
from ._base import EngineSpec, build_config

ENGINE_NAME = "AUDIT_REPORT"
OWNER = "D"

SYSTEM_PROMPT = """너는 감사보고서 요약 모델이다.
입력의 items[].stats는 서버가 SQL로 집계한 공식 결과다.
집계하거나 계산하거나 반올림하지 말고 stats에 있는 수치만 사용한다.
입력에 없는 수치를 만들지 않는다.
모든 항목을 빠짐없이 하나씩 요약한다.
반드시 JSON 객체 하나만 출력한다.
형식:
{"items":[{"queryType":"허용된 queryType","summary":"stats 수치를 포함한 한 문장"}]}
"""

SPEC = EngineSpec(model_id=None, device=None, revision=None, enable_thinking=None, local_files_only=None)


class AuditReportClient(LLMClient):
    name = ENGINE_NAME

    def prepare_messages(self, chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
        messages = [dict(message) for message in chat_messages]
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] += "\n\n입력의 모든 items를 같은 순서로 JSON에 포함하라."
        return messages

    def generate_options(self) -> dict[str, Any]:
        return {"repetition_penalty": 1.05}

    def finish_text(self, text: str) -> str:
        return text.strip()


def build_audit_report_client(*, settings: Any, model_id: str | None = None) -> AuditReportClient:
    return AuditReportClient(build_config(SPEC, settings=settings, model_id=model_id, engine=ENGINE_NAME))
