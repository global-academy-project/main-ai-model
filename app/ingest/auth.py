"""내부 적재 API의 Bearer 토큰 인증을 제공한다."""

from __future__ import annotations

import os
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """환경변수 토큰과 요청 토큰을 상수 시간 비교로 인증한다."""
    token = os.getenv("KB_INGEST_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="적재 API 인증이 설정되지 않았습니다.")
    if credentials is None or not compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=401,
            detail="적재 API 인증에 실패했습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
