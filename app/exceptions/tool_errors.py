"""도구 호출 예외.

도구가 "예상된 실패"(외부 서비스 불가, 입력 거부, 출력 규격 위반)를 낼 때 던진다.
FastAPI를 알지 못하며, HTTP 응답으로의 변환은 `tool_error_handler`가 담당한다.
(루프 러너는 같은 예외를 agent_runs 실패 기록으로 바꾼다.)

규약:
  - code는 AI 서버 내부 어휘다 (예: TOOL_UNAVAILABLE). 로그, agent_steps, 테스트에서 원인을 구분하는 데 쓴다.
    Spring에 나가는 응답의 code는 핸들러가 미리 정해진 코드로 번역한다 (exception_handlers.py).
  - status_code는 선택이다. 주지 않으면 핸들러가 code로 정한다. 특별한 상태가 필요할 때만 준다.
  - message는 도구 작성자가 직접 쓴 고정 문구만 쓴다. 외부 API·SDK 오류 원문, 키, 응답 본문은 넣지 않는다
    (원문은 도구 안에서 로그로만 남긴다). 이 메시지는 응답, agent_steps, EvidencePack에 그대로 남을 수 있다.
  - 프로그래밍 오류(ValueError, KeyError 등)는 ToolError로 바꾸지 않는다. 그대로 올라가야 발견된다.
"""


class ToolError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int | None = None,
        tool: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.tool = tool
        self.status_code = status_code
        super().__init__(message)
