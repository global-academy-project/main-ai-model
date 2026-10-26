from typing import Annotated
from fastapi import Depends
from app.clients.base import TextGenerationClient
from app.clients.fake_client import (
    FakeTextGenerationClient,
)
from app.clients.openai_client import (
    OpenAITextGenerationClient,
)
from app.core.config import Settings, get_settings
from app.services.summary_service import SummaryService

def get_text_generation_client(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> TextGenerationClient:
    if settings.engine == "FAKE":
        return FakeTextGenerationClient(
            response_text="Fake Provider 요약 결과",
        )

    if settings.engine == "OPENAI":
        if settings.openai_api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY가 필요합니다."
            )

        if settings.openai_model is None:
            raise RuntimeError(
                "OPENAI_MODEL이 필요합니다."
            )

        return OpenAITextGenerationClient(
            api_key=(
                settings.openai_api_key.get_secret_value()
            ),
            model=settings.openai_model,
            timeout_seconds=settings.ai_timeout_seconds,
            max_output_tokens=(
                settings.ai_max_output_tokens
            ),
        )

    raise RuntimeError(
        f"지원하지 않는 Provider: "
        f"{settings.engine}"
    )


def get_summary_service(
    client: Annotated[
        TextGenerationClient,
        Depends(get_text_generation_client),
    ],
) -> SummaryService:
    return SummaryService(
        client=client,
    )