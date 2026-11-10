"""SIF(고위험 요인 중대재해 사례, 공공데이터포털 15161362) 프로필.

데이터셋마다 다른 값(식별 정보, 티어, 태그 매핑 표)은 이 파일에만 둔다.
매핑 근거와 건수는 claude/사고데이터_SIF_태그매핑_초안.md(확정본)와 같다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NAME = "sif"
SOURCE_TYPE = "SIF"
COLLECTION = "kb_accidents"
DATASET_URL = "https://www.data.go.kr/data/15161362/openapi.do"
PUBLISHER = "한국산업안전보건공단"
LICENSE = "이용허락범위 제한 없음"
ID_PREFIX = "SIF-"

CATE_ALLOWED = ("건설업", "전업종")

# --- tag_codes 스냅샷 --------------------------------------------------------------------
TAG_CODES: dict[str, list[str]] = {
    k: v for k, v in json.loads((Path(__file__).with_name("tag_codes_v5.json")).read_text(encoding="utf-8")).items() if not k.startswith("_")
}

# --- 티어 (D9) ---------------------------------------------------------------------------
# A: 전업종 중 아래 대분류가 아닌 것 / X: 아래 대분류(플랜트 무관) / B: 건설업 중 설비공사·화기/질식/감전류 / C: 나머지 건설업
EXCLUDED_LCLSF = ("임업", "농업", "어업", "금융및보험업")
CONSTRUCTION_B_LCLSF_PREFIX = "5."  # 5. 전기·기계 설비공사
CONSTRUCTION_B_DISASTER_TYPES = frozenset({"화재", "폭발", "화상", "질식", "감전", "중독", "이상온도 접촉", "유해물 접촉", "파열"})


def squash(text: str | None) -> str:
    """모든 공백을 지운 값. 대분류 이름의 불규칙한 공백('어  업')을 맞추는 데 쓴다."""
    return "".join((text or "").split())


def tier_of(rec: dict) -> str:
    lcl = squash(rec.get("sifLclsfNm"))
    if rec.get("cateSeNm") == "건설업":
        if lcl.startswith(CONSTRUCTION_B_LCLSF_PREFIX) or rec.get("disasterType") in CONSTRUCTION_B_DISASTER_TYPES:
            return "B"
        return "C"
    return "X" if lcl in {squash(x) for x in EXCLUDED_LCLSF} else "A"


# --- accident_type -----------------------------------------------------------------------
# 건설업: disasterType 직접 매핑. 매핑에 없는 값(중독·화상·이상온도 접촉·기타·찔림·베임·익사)은 비운다.
CONSTRUCTION_ACCIDENT_TYPE = {
    "추락": ["FALL"],
    "전도": ["TRIP"],
    "낙하": ["STRUCK"],
    "부딪힘": ["STRUCK"],
    "깔림": ["CAUGHT"],
    "끼임": ["CAUGHT"],
    "붕괴": ["COLLAPSE"],
    "감전": ["ELECTRIC"],
    "질식": ["ASPHYXIATION"],
    "화재": ["FIRE"],
    "폭발": ["EXPLOSION"],
    "파열": ["EXPLOSION"],
    "유해물 접촉": ["LEAK"],
}

# 전업종: disasterFactor 템플릿 꼬리 키워드. 해당하면 모두 붙인다(순서는 코드표 순서 아님).
# '맞'만 쓰면 "균형이 맞지 않은"에 걸리므로 '맞음|맞거나'로 한정했다.
FACTOR_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FIRE", re.compile(r"화재")),
    ("EXPLOSION", re.compile(r"폭발|파열")),
    ("LEAK", re.compile(r"누출|화학물질 접촉|유해물")),
    ("ASPHYXIATION", re.compile(r"질식|중독")),
    ("FALL", re.compile(r"떨어|추락")),
    ("TRIP", re.compile(r"넘어")),
    ("CAUGHT", re.compile(r"끼임|말리|깔림")),
    ("STRUCK", re.compile(r"부딪|맞음|맞거나|낙하|비래|충돌")),
    ("COLLAPSE", re.compile(r"무너|붕괴")),
    ("ELECTRIC", re.compile(r"감전")),
)

# --- work_type ---------------------------------------------------------------------------
# 전업종: situation 정확 일치(공백 정규화 후). 표에 없는 situation은 비운다. GENERAL은 붙이지 않는다.
_SITUATION_WORK_TYPE = {
    "HOT": ("용접(용단)·절단·사상 또는 도장 작업", "그라인더 등 공구 사용 작업", "용해로 사용 작업"),
    "CONFINED": ("피트, 맨홀, 오·폐수 처리시설 등 밀폐된 장소에서 작업",),
    "HEIGHT": ("설비, 구조물 등 높은 장소에서의 작업", "지붕, 천장 등 위에서 작업", "사다리 사용 작업·통행", "고소작업대 사용 작업"),
    "ELECTRICAL": ("전기설비 취급 작업",),
    "LIFTING": ("크레인(천장주행, 갠트리, 지브 등) 사용 작업", "이동식크레인(카고 크레인, 집게차) 사용 작업", "중량물 취급·운반 작업"),
}
SITUATION_WORK_TYPE = {squash(s): code for code, names in _SITUATION_WORK_TYPE.items() for s in names}

# 건설업: 중분류·소분류·기인물 키워드(근거가 약함 → P1에서 눈으로 확인).
CONSTRUCTION_WORK_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("HOT", re.compile(r"용접|용단|화기")),
    ("CONFINED", re.compile(r"밀폐|맨홀|피트|탱크")),
    ("HEIGHT", re.compile(r"고소작업|비계|사다리|지붕|천장")),
    ("ELECTRICAL", re.compile(r"전기 설비|가설전기|충전부|활선|전주")),
    ("LIFTING", re.compile(r"양중|크레인|인양|줄걸이")),
)

# --- hazard (강한 근거 3개만) ------------------------------------------------------------
HAZARD_CHEMICAL_SITUATION = squash("위험물질 취급 또는 그 장소에서 작업")
HAZARD_CHEMICAL_ORGT = "화학물질"  # 건설업 기인물

# --- severity (D7 ①) ---------------------------------------------------------------------
FATAL_WORDS = ("사망", "익사")

# --- 연도 --------------------------------------------------------------------------------
YEAR_RE = re.compile(r"(\d{4})\s*년")


def severity_of(overview: str) -> str | None:
    return "FATAL" if any(w in overview for w in FATAL_WORDS) else None


def year_of(overview: str, year_min: int, year_max: int) -> int | None:
    m = YEAR_RE.search(overview)
    if not m:
        return None
    y = int(m.group(1))
    return y if year_min <= y <= year_max else None
