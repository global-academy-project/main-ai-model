"""LLM 클라이언트 공통 인터페이스.

모든 LLM 호출은 `LLMClient` 한 곳을 지난다. `chat()`의 기본 구현은 transformers 로컬 모델로 생성한다
(`llm_transformers.py`). 모델 설정(`config`)을 주면 그대로 동작하고, 다른 방식의 엔진(가짜 등)은 `chat()`을 덮어쓴다.

손질 지점 (기본은 아무것도 하지 않는다. 시나리오 엔진이 덮어쓴다):
    prepare_messages(chat_messages)  모델에 넣기 전 메시지 손질
    generate_options()               generate()에 더 넘길 인자 (max_new_tokens·do_sample은 넣을 수 없다)
    finish_text(text)                모델 출력 손질
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .llm_types import LLMResponse, Message, ToolSpec

if TYPE_CHECKING:
    from .llm_transformers import TransformersConfig


class LLMClient:
    """모든 LLM 호출은 이 인터페이스 한 곳을 지난다."""

    name: str = "base"
    config: TransformersConfig | None = None  # 기본 chat()이 쓰는 transformers 설정. 없으면 chat()을 덮어써야 한다

    def __init__(self, config: TransformersConfig | None = None) -> None:
        self.config = config

    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.config is None:
            from app.exceptions import LLMConfigError

            raise LLMConfigError(f"{self.name} 엔진에는 모델 설정(config)이 없습니다. config를 주거나 chat()을 구현하세요.", engine=self.name)
        from .llm_transformers import generate_chat  # transformers는 쓸 때만 불러온다

        return generate_chat(self, self.config, messages, system=system, tools=tools, max_tokens=max_tokens)

    # ── 손질 지점 ──

    def prepare_messages(self, chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """모델에 넣기 전 메시지(role·content 목록)를 손질한다. 새 목록을 돌려준다."""
        return chat_messages

    def generate_options(self) -> dict[str, Any]:
        """`model.generate()`에 더 넘길 인자 (예: {"repetition_penalty": 1.1}). max_new_tokens·do_sample은 넣지 않는다."""
        return {}

    def finish_text(self, text: str) -> str:
        """모델 출력(앞뒤 공백을 뗀 글자)을 손질한다."""
        return text
