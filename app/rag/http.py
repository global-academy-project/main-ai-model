from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import RagBackendError


def post_json(
    url: str,
    body: Mapping[str, object],
    headers: Mapping[str, str],
    timeout: float,
    service_name: str,
) -> Mapping[str, object]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RagBackendError(f"{service_name} 요청이 실패했습니다 (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RagBackendError(f"{service_name}에 연결하거나 응답을 읽지 못했습니다.") from exc
    if not isinstance(result, Mapping):
        raise RagBackendError(f"{service_name} 응답 형식이 올바르지 않습니다.")
    return result
