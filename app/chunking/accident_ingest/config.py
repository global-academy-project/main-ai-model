"""사고 데이터 적재 준비 파이프라인 설정.

값을 바꿀 때는 이 파일의 상수만 고친다. 명령줄 옵션이 있으면 그쪽이 우선한다.
데이터셋마다 다른 값(티어, 태그 표, 라벨)은 profiles/ 안에 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- A의 지식 DB 계약 (app/rag/filters.py COLLECTIONS) -----------------------------------
SITE_ID = 0  # 공공 자료는 Qdrant payload에서 0
FLAGGED = False

# --- 청킹 (D6, D8) -----------------------------------------------------------------------
# 사례 1개 = 청크 1개. 합본이 이 글자 수를 넘을 때만 섹션 경계에서 나눈다(실측 최대 736자).
MAX_CHUNK_CHARS = 1500
# 소스당 청크 수 한도. 임베딩 서버 한도(기본 8)를 넘으면 502가 나므로 보내기 전에 막는다.
MAX_CHUNKS_PER_SOURCE = 8

# --- 적재 범위 (D9) ----------------------------------------------------------------------
# 6,026건 전부를 읽고 티어를 계산한 뒤 이 티어만 다음 단계로 보낸다.
INCLUDE_TIERS = ("A", "B")

# --- 임베딩 입력 -------------------------------------------------------------------------
ADD_CONTEXT_HEADER = True

# --- 태그 --------------------------------------------------------------------------------
# 전업종에서 accident_type이 비었을 때 개요 키워드로 되살린다. 오탐이 있어 기본은 끔.
OVERVIEW_RESCUE = False

# --- 검증 한도 ---------------------------------------------------------------------------
MAX_TEXT_CHARS = 32768
MAX_EMBED_CHARS = 8000
MAX_TITLE_CHARS = 300
MAX_LOCATOR_CHARS = 100
MAX_EXTERNAL_ID_CHARS = 100
YEAR_MIN = 1990
# SIF 가이드의 필드 최대 크기(재해개요 2,000 / 재해유발요인 1,000 / 감소대책 4,000).
# 이 값과 같거나 넘으면 API가 잘랐을 수 있다.
FIELD_GUIDE_MAX = {"disasterOverview": 2000, "disasterFactor": 1000, "dcrsCntrplnCn": 4000}

NOISE_STRINGS = ("국가법령정보센터", "법제처")

# 서술 필드 → (슬롯 이름, 라벨). 슬롯 번호는 SLOT에 고정하고 바꾸지 않는다(추가만).
SECTIONS = (
    ("overview", "disasterOverview", "재해개요"),
    ("cause", "disasterFactor", "재해유발요인"),
    ("measures", "dcrsCntrplnCn", "위험성 감소대책(예시)"),
)
# point id가 여기서 나온다: chunk_index = SLOT * 100 + part. 3, 4는 예약, 9 = other.
SLOT = {"case": 0, "overview": 0, "cause": 1, "measures": 2, "other": 9}
CASE_LOCATOR = "재해개요·유발요인·감소대책(예시)"

# 정보가 없는 값으로 보는 문자열(그 섹션을 만들지 않는다)
EMPTY_VALUES = {"", "-", "없음", "해당없음", "해당 없음", "n/a", "N/A"}


@dataclass
class BuildSettings:
    """한 번의 실행에서 쓰는 값. 기본값은 위 상수."""

    max_chars: int = MAX_CHUNK_CHARS
    include_tiers: tuple[str, ...] = INCLUDE_TIERS
    add_header: bool = ADD_CONTEXT_HEADER
    overview_rescue: bool = OVERVIEW_RESCUE
    year_max: int | None = None  # None이면 실행 시점의 올해

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars는 100 이상이어야 합니다.")
        bad = [t for t in self.include_tiers if t not in {"A", "B", "C", "X"}]
        if bad:
            raise ValueError(f"알 수 없는 티어: {bad}")
