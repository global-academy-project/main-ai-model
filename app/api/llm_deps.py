"""각자 기능(라우터)에서 자기 시나리오 모델을 받는 의존성.

팀원은 엔진·모델 코드를 몰라도 된다. 라우터에 자기 키만 적으면, 기동 때 만든 제공자
(`app.state.llm_provider`)에서 시나리오 모델 표에 맞는 엔진을 받는다.

    from typing import Annotated
    from fastapi import APIRouter, Depends
    from app.api.llm_deps import scenario_llm
    from app.llm.llm_base import LLMClient

    RecordQueryLLM = Annotated[LLMClient, Depends(scenario_llm("audit_query"))]

    @router.post("/api/v1/agent/record-query")
    def record_query(body: RecordQueryRequest, llm: RecordQueryLLM): ...   # 동기 모델 호출이라 def

규칙
  - 키는 router 또는 시나리오 이름이다 (app/core/scenario_keys.py: work_register·confined_space_gate·near_miss_warning·
    daily_safety_log·audit_query·small_site_guide). 글자 코드(A~F)도 받는다. default는 표의 기본값이라 직접 받지 않는다.
    오타는 임포트 때 ValueError.
  - 엔진이 막혔거나 설정이 틀리면 provider.get()이 LLMError(EngineBlockedError·LLMConfigError)를 올리고,
    create_app()에 등록된 llm_error_handler가 공통 오류 응답으로 바꾼다. 여기서 잡지 않는다.
  - 요청자가 키·엔진을 고를 수 없다 (쿼리·본문 인자 없음, 지침 4장).
  - 파일에 `from __future__ import annotations`가 있으면 `RecordQueryLLM` 같은 Annotated 별칭은 **모듈 최상위**에 둔다.
    함수 안에 두면 FastAPI가 타입을 찾지 못해 쿼리 파라미터로 오인한다.
  - 테스트는 `app.state.llm_provider = LLMClientProvider(get_settings(), {"default": EngineBinding("FAKE")})`로 바꾼다.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request

from app.llm.llm_base import LLMClient
from app.core.scenario_keys import DEFAULT_KEY, canonical_key
from app.llm.llm_provider import LLMClientProvider


def _provider(request: Request) -> LLMClientProvider:
    provider = getattr(request.app.state, "llm_provider", None)
    if provider is None:
        # 설정 실수(프로그래밍 오류): create_app()이 기동 때 넣는다. 테스트 앱이면 직접 넣는다.
        raise RuntimeError("app.state.llm_provider가 없습니다. create_app()에서 LLMClientProvider를 넣어야 합니다.")
    return provider


def scenario_llm(key: str) -> Callable[[Request], LLMClient]:
    """`Depends(scenario_llm("audit_query"))` 형태로 쓴다. 같은 키는 같은 엔진 객체를 받는다 (provider가 공유)."""
    try:
        name = canonical_key(key)
    except ValueError as exc:
        raise ValueError(f"scenario_llm: {exc}") from None
    if name == DEFAULT_KEY:
        raise ValueError("scenario_llm: default는 직접 받지 않는다 (시나리오 이름을 쓴다)")

    def dependency(request: Request) -> LLMClient:
        return _provider(request).get(name)

    dependency.__name__ = f"scenario_llm_{name}"
    return dependency
