from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .config import EmbeddingServerSettings
from .embedding_model import BgeM3Encoder, load_encoder

logger = logging.getLogger(__name__)
TextInput = Annotated[str, Field(min_length=1, max_length=32768)]
BatchInput = Annotated[list[TextInput], Field(min_length=1, max_length=32)]
bearer = HTTPBearer(auto_error=False)


class EmbeddingRequest(BaseModel):
    model: str
    input: TextInput | BatchInput
    encoding_format: Literal["float"] = "float"

    @field_validator("input")
    @classmethod
    def validate_text(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value
        if any(not text.strip() for text in texts):
            raise ValueError("빈 문장은 임베딩할 수 없습니다.")
        return value


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class TokenUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingItem]
    usage: TokenUsage


def create_app(
    settings: EmbeddingServerSettings | None = None,
    *,
    encoder_factory: Callable[[EmbeddingServerSettings], BgeM3Encoder] = load_encoder,
) -> FastAPI:
    settings = settings if settings is not None else EmbeddingServerSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.encoder = await run_in_threadpool(encoder_factory, settings)
        except (OSError, ValueError, RuntimeError, ImportError):
            logger.exception("임베딩 모델 초기화에 실패했습니다.")
            raise
        try:
            yield
        finally:
            app.state.encoder = None

    app = FastAPI(title="BGE-M3 Embedding API", lifespan=lifespan)
    app.state.encoder = None

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        key = settings.embedding_api_key
        if key is not None and key.get_secret_value():
            if credentials is None or not compare_digest(
                credentials.credentials.encode(), key.get_secret_value().encode(),
            ):
                raise HTTPException(
                    status_code=401, detail="임베딩 API 인증에 실패했습니다.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str | int]:
        if request.app.state.encoder is None:
            raise HTTPException(status_code=503, detail="임베딩 모델이 준비되지 않았습니다.")
        return {
            "status": "ok",
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dim,
        }

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        dependencies=[Depends(authenticate)],
    )
    def embeddings(body: EmbeddingRequest, request: Request) -> EmbeddingResponse:
        if body.model != settings.embedding_model:
            raise HTTPException(status_code=400, detail="요청한 임베딩 모델은 제공하지 않습니다.")
        encoder: BgeM3Encoder | None = request.app.state.encoder
        if encoder is None:
            raise HTTPException(status_code=503, detail="임베딩 모델이 준비되지 않았습니다.")
        texts = [body.input] if isinstance(body.input, str) else body.input
        try:
            vectors, token_count = encoder.encode(texts)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("임베딩 추론에 실패했습니다.")
            raise HTTPException(status_code=503, detail="임베딩 추론에 실패했습니다.") from exc
        return EmbeddingResponse(
            model=settings.embedding_model,
            data=[
                EmbeddingItem(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ],
            usage=TokenUsage(prompt_tokens=token_count, total_tokens=token_count),
        )

    return app
