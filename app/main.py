from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
import logging
from app.agent.scenario.scenario_handlers import load_chat_scenarios
from app.api import health_router
from app.api.v1.agent.agent_run import router as agent_run_router
from app.api.v1.agent.near_miss_tag import router as near_miss_tag_router
from app.api.v1.agent.audit_compose import router as audit_compose_router
from app.api.v1.agent.record_query import router as record_query_router
from app.api.v1.agent.record_query_deps import get_record_query_agent
from app.api.v1.agent.record_query_rules_deps import get_rule_first_record_query_agent
from app.api.v1.agent.rule_input_extract import router as rule_input_extract_router
from app.api.v1.agent.stt import router as stt_router
from app.api.v1.agent.vision import router as vision_router
from app.core import get_settings
from app.exceptions import AgentRunError, BusinessError, LLMError
from app.handlers import (
    business_error_handler,
    llm_error_handler
)
from app.handlers.agent_run_handlers import (
    agent_run_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.llm.llm_preload import preload_models
from app.llm.llm_provider import LLMClientProvider
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 기동 때 LLM 모델을 한 번 읽어 둔다 (LLM_PRELOAD, 기본 켬). 시험이 바꿔 끼운 provider 도 따르게 기동 시점의 app.state 를 읽는다
        if settings.llm_preload:
            await run_in_threadpool(preload_models, app.state.llm_provider)
        yield

    app = FastAPI(lifespan=lifespan)

    app.add_exception_handler(
        BusinessError,
        business_error_handler,
    )

    app.add_exception_handler(
        LLMError,
        llm_error_handler,
    )
    app.add_exception_handler(AgentRunError, agent_run_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.state.llm_provider = LLMClientProvider.from_settings(settings)
    # 채팅 범위는 기동 때 한 번 검사한다 (잘못된 값이면 여기서 ValueError로 뜨지 않는다. 요청마다 500이 되지 않게)
    app.state.chat_scenarios = load_chat_scenarios(settings.chat_scenarios)

    # 라우터도 create_app() 안에서 붙인다 (테스트의 TestClient(create_app())에도 경로가 있게)
    app.include_router(health_router)
    app.include_router(agent_run_router)
    app.include_router(near_miss_tag_router)
    app.include_router(audit_compose_router)
    app.include_router(record_query_router)
    if settings.record_plan_rules:
        app.dependency_overrides[get_record_query_agent] = get_rule_first_record_query_agent
    app.include_router(rule_input_extract_router)
    app.include_router(vision_router)   # 사진 인식 (C, multipart → python-multipart 필요)
    app.include_router(stt_router)      # 음성 → 글 (C, multipart → python-multipart 필요)
    return app

app = create_app()
