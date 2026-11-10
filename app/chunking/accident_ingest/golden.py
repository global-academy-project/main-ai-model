"""기준 사례: 규칙을 고쳤을 때 값이 조용히 바뀌는 것을 잡는 고정 예시.

ID_GOLDEN   정규화·해시 함수의 입력 → 기대 id. 이 값이 바뀌면 이미 올린 point id가 어긋난다(적재 후 변경 금지).
CASE_GOLDEN 손으로 읽어 확인한 사례(개요의 앞부분으로 찾는다). 데이터에 없으면 건너뛴다(info).
"""

from __future__ import annotations

# (입력 개요, 기대 external_id). 공백 종류가 달라도 같은 id가 나와야 한다.
ID_GOLDEN: list[tuple[str, str]] = [
    ("2019년 07월경 ○○현장에서 추락하여 사망", "SIF-3c4a3bb83c99ce9a"),
    ("2019년 07월경 ○○현장에서   추락하여 사망 ", "SIF-3c4a3bb83c99ce9a"),
]

# 개요 앞부분 → 기대 값. 11/10 표본을 눈으로 읽고 정했다(건설 3 + 전업종 4). 규칙을 고쳐 값이 바뀌면
# 의도한 변경인지 확인하고 이 표를 같이 고친다. 데이터에 없는 사례는 건너뛴다.
CASE_GOLDEN: list[dict] = [
    {
        "overview_prefix": "2021년 6월경 전신주의 저압케이블 연결작업을 완료하고",
        "expect": {"cate": "건설업", "severity": "FATAL", "accident_type": ["CAUGHT"], "work_type": ["HEIGHT", "ELECTRICAL"], "hazard": [], "year": 2021},
    },
    {
        "overview_prefix": "2020년 09월경 피재자가 각 층의 엘리베이터 설치현황을 확인하던 중",
        "expect": {"cate": "건설업", "severity": "FATAL", "accident_type": ["FALL"], "work_type": [], "hazard": [], "year": 2020},
    },
    {
        "overview_prefix": "2018년 05월경 ○○현장에서 피재자가 옹벽위에 설치되어 기존 건물로 연결되는 고압 케이블",
        "expect": {"cate": "건설업", "severity": "FATAL", "accident_type": ["FALL"], "work_type": ["ELECTRICAL"], "hazard": [], "year": 2018},
    },
    {
        "overview_prefix": "2022년 1월경 OO공장에서 반응기 내부 혼합물을 비닐호스",
        "expect": {"cate": "전업종", "severity": None, "accident_type": ["FIRE", "EXPLOSION"], "work_type": [], "hazard": ["CHEMICAL"], "year": 2022},
    },
    {
        "overview_prefix": "2022년 5월경 OO공장 사일로 전기실에서 재해자 2명이",
        "expect": {"cate": "전업종", "severity": None, "accident_type": ["ELECTRIC"], "work_type": ["ELECTRICAL"], "hazard": ["ELECTRICAL"], "year": 2022},
    },
    {
        "overview_prefix": "2023년 9월경 OO공장에서 재해자가 지게차에서 시동을 정지하지 않은 상태로 하차하여",
        "expect": {"cate": "전업종", "severity": "FATAL", "accident_type": ["CAUGHT", "STRUCK"], "work_type": [], "hazard": [], "year": 2023},
    },
    {
        "overview_prefix": "2021년 11월경 빌딩에서 재해자가 건물 울타리 안쪽 화단에",
        "expect": {"cate": "전업종", "severity": "FATAL", "accident_type": ["FALL"], "work_type": ["HEIGHT"], "hazard": [], "year": 2021},
    },
]
