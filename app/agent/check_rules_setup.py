"""check_rules 실제 도구 준비 (CHECK_RULES_TOOL=SPRING일 때만).

C의 도구(app/agent/rule_tools)를 Spring 내부 API 클라이언트에 붙여 (handler, parameters)로 돌려준다.
tool_setup.build_tool_registry(check_rules_tool=...)가 이것으로 mock을 바꿔 등록한다.

- MOCK이면 None을 돌려주고 C의 모듈은 임포트하지 않는다 (C 모듈이 깨져 있어도 서버는 뜬다).
- 시간은 계획서 7-2장 값(연결 2초·읽기 3초). C가 rule_tools/__init__.py에 적은 값과 같다.
- URL·토큰이 없는 SPRING 설정은 Settings가 기동 때 이미 막는다. 여기서는 한 번 더 확인만 한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from app.core import Settings

CONNECT_TIMEOUT_SECONDS = 2.0
READ_TIMEOUT_SECONDS = 3.0


def build_check_rules_tool(settings: Settings) -> tuple[Callable[[Mapping[str, Any]], Any], dict[str, Any]] | None:
    """SPRING이면 (handler, parameters), MOCK이면 None."""
    if settings.check_rules_tool != "SPRING":
        return None
    if settings.internal_api_base_url is None or settings.internal_api_token is None:
        raise ValueError("CHECK_RULES_TOOL=SPRING에는 INTERNAL_API_BASE_URL과 INTERNAL_API_TOKEN이 필요합니다.")

    # 지연 임포트: SPRING일 때만 C의 모듈과 클라이언트를 불러온다
    from app.agent.rule_tools import CHECK_RULES_PARAMETERS, make_check_rules
    from app.integrations import InternalApiClient

    client = InternalApiClient(
        settings.internal_api_base_url,
        settings.internal_api_token.get_secret_value(),
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
    )
    return make_check_rules(client), CHECK_RULES_PARAMETERS
