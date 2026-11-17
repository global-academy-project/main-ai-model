"""시험 공통 환경 (11/17, B).

설정(`Settings`)은 환경변수 > .env > 코드 기본값 순으로 읽는다. 시험이 개발 PC의 .env·터미널 환경변수에 따라
결과가 달라지지 않게, 시험마다 아래 값을 먼저 고정한다. 다른 값이 필요한 시험은 지금처럼 자기 안에서
monkeypatch.setenv/delenv 나 Settings(...) 인자로 바꾼다 (이 고정보다 나중에 돌아서 이긴다).

- MSDS_TOOL=MOCK: 코드 기본값은 KOSHA(키 필요)다. 시험은 공단 API를 부르지 않고 키도 없으므로 MOCK에서 시작한다.
- KOSHA_MSDS_SERVICE_KEY: 지운다. 실제 키가 시험에 섞이지 않게.
- NETWORK_ENV=OPEN, CHECK_RULES_TOOL=MOCK: 기본값이 없는 필수 설정이다.
- LLM_PRELOAD=false: `with TestClient(app)`가 기동 때 실제 모델을 읽지 않게.
- get_settings 캐시를 앞뒤로 비운다 (앞 시험의 설정이 남지 않게).
"""

from __future__ import annotations

import pytest

from app.core import get_settings

TEST_ENV = {
    "MSDS_TOOL": "MOCK",
    "NETWORK_ENV": "OPEN",
    "CHECK_RULES_TOOL": "MOCK",
    "LLM_PRELOAD": "false",
}
CLEARED_ENV = ("KOSHA_MSDS_SERVICE_KEY",)


@pytest.fixture(autouse=True)
def _test_settings_env(monkeypatch):
    for name, value in TEST_ENV.items():
        monkeypatch.setenv(name, value)
    for name in CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
