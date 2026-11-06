"""법령 적재 준비 파이프라인 설정.

값을 바꿀 때는 이 파일의 상수만 고친다. 명령줄 옵션(--max-chars 등)이 있으면 그쪽이 우선한다.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- A의 지식 DB 계약 (app/rag/filters.py COLLECTIONS, A님 문서 4장) -------------------
SOURCE_TYPE = "LAW"
COLLECTION = "kb_guides"
SITE_ID = 0  # 공공 자료는 Qdrant payload에서 0
FLAGGED = False

# --- 구조 파싱 ---------------------------------------------------------------------
# 제목 단계(위 → 아래 순서). "제N<단계>"로 시작하는 줄을 제목으로 읽고, 아래 단계가 나오면 같거나 낮은 단계는 경로에서 빠진다.
# 이 법령에는 관 아래에 '속'이 있다(제1속 굴착면의 기울기 등). 다른 법령에서 새 단계가 나오면 여기에만 더하면 된다.
# 명령줄: --heading-levels 편,장,절,관,속
HEADING_LEVELS = ("편", "장", "절", "관", "속")

# --- 청킹 --------------------------------------------------------------------------
# 조 본문이 이 글자 수를 넘을 때만 항 경계에서 나눈다. 넘지 않으면 조 1개 = 청크 1개.
MAX_CHUNK_CHARS = 1000

# 임베딩 입력 맨 앞에 "법령명 > 편 > 장 > 절" 머리말을 붙인다. False면 본문만 임베딩한다.
# (인용 원문 payload.text에는 머리말을 넣지 않는다.)
ADD_CONTEXT_HEADER = True

# payload year 기준: "promulgation"(공포 연도) 또는 "effective"(시행 연도)
YEAR_BASIS = "promulgation"

# 계층 경로로 hazard/work_type 태그를 붙인다. 매핑표(tags.py)가 초안이라 기본은 끔.
TAGS_ENABLED = False

# --- 검증 한도 ---------------------------------------------------------------------
MAX_TEXT_CHARS = 32768  # ingest_main.IngestChunk.text 최대
MAX_EMBED_CHARS = 8000  # 임베딩 8,192토큰의 글자 수 대용 경고선(정확한 토큰 수 아님)

# 청크 본문에 있으면 안 되는 머리글·바닥글 잡음
NOISE_STRINGS = ("국가법령정보센터", "법제처", "044-")


@dataclass
class BuildSettings:
    """한 번의 실행에서 쓰는 값. 기본값은 위 상수."""

    max_chars: int = MAX_CHUNK_CHARS
    add_header: bool = ADD_CONTEXT_HEADER
    year_basis: str = YEAR_BASIS
    tags_enabled: bool = TAGS_ENABLED
    law_key: str | None = None  # 없으면 법령명에서 공백을 뺀 값
    file_path: str | None = None  # 없으면 raw/<원본 파일명>
    url: str | None = None

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars는 100 이상이어야 합니다.")
        if self.year_basis not in {"promulgation", "effective"}:
            raise ValueError("year_basis는 promulgation 또는 effective여야 합니다.")
