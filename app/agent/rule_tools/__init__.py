"""C 담당 실제 도구. Spring 규칙 엔진을 부른다.

등록(tool_setup.py)과 설정(config.py)은 B 담당 파일이라 이 패키지는 건드리지 않는다. 등록 예:

    from app.agent.rule_tools import CHECK_RULES_PARAMETERS, make_check_rules
    from app.integrations import InternalApiClient

    client = InternalApiClient(settings.internal_api_base_url, settings.internal_api_token.get_secret_value(),
                               connect_timeout=2.0, read_timeout=3.0)
    registry.register("check_rules", make_check_rules(client), owner="C", replace=True,
                      description=..., parameters=CHECK_RULES_PARAMETERS)
"""

from .check_rules import CHECK_RULES_PARAMETERS, make_check_rules

__all__ = ["CHECK_RULES_PARAMETERS", "make_check_rules"]
