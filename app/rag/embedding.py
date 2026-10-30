from __future__ import annotations

import math
from collections.abc import Mapping

from .config import RagSettings
from .errors import RagBackendError
from .http import post_json


def embed_text(settings: RagSettings, text: str) -> list[float]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    response = post_json(
        settings.embedding_url,
        {"model": settings.embedding_model, "input": text},
        headers,
        settings.timeout_seconds,
        "임베딩",
    )
    try:
        data = response["data"]
        if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
            raise TypeError
        vector = data[0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RagBackendError("임베딩 서버 응답 형식이 올바르지 않습니다.") from exc

    if (
        not isinstance(vector, list)
        or len(vector) != settings.embedding_dim
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        )
    ):
        raise RagBackendError("임베딩 벡터의 차원 또는 값이 올바르지 않습니다.")
    return [float(value) for value in vector]
