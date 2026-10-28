"""도구 호출 최소 규약. 에이전트 쪽 코드가 도구를 부를 때 쓰는 모양만 정한다.
ToolError는 app/exceptions/tool_errors.py로 옮겼고, 이 모듈은 호환을 위해 다시 내보내기만 한다.

도구 레지스트리(1주차 7번)가 생기면 ToolInvoker는 레지스트리가 구현한다.
그때까지는 이 Protocol을 만족하는 가짜(테스트)나 mock 도구 묶음을 넣어 쓴다.

실패 규약:
  - 도구가 "예상된 실패"(외부 서비스 불가, 입력 거부, 출력 규격 위반)를 낼 때는 ToolError를 던진다.
  - code는 to_error_info처럼 어휘를 통일한다 (예: NOT_FOUND).
  - message는 도구 작성자가 직접 쓴 고정 문구만 쓴다. 외부 API·SDK 오류 원문, 키, 응답 본문은 넣지 않는다
    (원문은 도구 안에서 로그로만 남긴다). 이 메시지는 agent_steps와 EvidencePack에 그대로 남을 수 있다.
  - 프로그래밍 오류(ValueError, KeyError 등)는 ToolError로 바꾸지 않는다. 그대로 올라가야 발견된다.
"""

from collections.abc import Mapping
from typing import Any, Protocol

class ToolInvoker(Protocol):
    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """등록된 도구를 이름으로 호출한다. 없는 도구는 ToolError("NOT_FOUND")."""
        ...

