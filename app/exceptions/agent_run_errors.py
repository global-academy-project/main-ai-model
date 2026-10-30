"""에이전트 실행 실패와 그때까지 모은 단계.

같은 실패가 HTTP 응답도 되고 Spring의 agent_runs 실패 기록도 된다. 실패해도 그때까지의 steps를
돌려줘야 Spring이 agent_steps로 저장할 수 있다 (계획서 7-2 "실패" 행). FastAPI를 모른다.
"""

from collections.abc import Sequence
from typing import Any
from .llm_errors import LLMError
from .tool_errors import ToolError

class AgentRunError(Exception):
    def __init__(self, cause: LLMError | ToolError, *, steps: Sequence[Any]) -> None:
        # 원문 메시지를 예외 문자열에 싣지 않는다 (로그·응답으로 새지 않게). 원인은 cause로만.
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.steps = tuple(steps)