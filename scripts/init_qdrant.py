"""Qdrant 컬렉션 생성 (벡터 DB 스키마).

담당: C 작성, 업무 소유 A (검색·태깅)

    python -m scripts.init_qdrant            # 없으면 만들기 (있으면 그대로 둠)
    python -m scripts.init_qdrant --recreate # 지우고 다시 만들기 (임베딩 모델을 바꿨을 때)
    python -m scripts.init_qdrant --check    # 테스트 점 하나 넣고 5축 필터 검색 후 삭제

환경 변수
    QDRANT_URL      기본 http://qdrant:6333 (compose 서비스명)
    QDRANT_API_KEY  쓰기 가능한 키 (이 스크립트와 적재 작업만 사용)
                    에이전트 실행은 QDRANT_READ_ONLY_API_KEY만 받는다 → 인젝션돼도 벡터를 바꿀 수 없음
    EMBEDDING_DIM   기본 1024 (bge-m3). 임베딩 모델을 바꾸면 이 값과 함께 --recreate

컬렉션 2개
    kb_accidents  재해조사보고서, SIF 아카이브, 사내 아차사고  -> search_accidents 도구
    kb_guides     KOSHA GUIDE, 위험성평가 안내서, 체크리스트, 법령 조항 -> search_guides 도구

벡터 (점 하나에 두 종류)
    dense   의미 검색용 밀집 벡터 (코사인)
    sparse  키워드 검색용 희소 벡터 (IDF 가중). 두 결과를 합쳐 하이브리드 검색

payload (점 하나 = 청크 하나). 값은 MySQL tag_codes의 코드와 같다.
    kb_source_id    int        agent_kb.kb_sources.id
    kb_chunk_id     int        agent_kb.kb_chunks.id
    source_type     keyword    ACCIDENT_REPORT | SIF | NEAR_MISS | KOSHA_GUIDE | RISK_GUIDE | CHECKLIST | LAW
    site_id         int        공공 데이터 0, 사내 데이터는 현장 id  -> 검색 시 site_id in [0, 내 현장]
    title           (색인 없음) 출력 인용용
    locator         (색인 없음) 조항·페이지
    url             (색인 없음)
    text            (색인 없음) 청크 원문 일부 (근거 인용용)
    severity        keyword    NEAR_MISS | MINOR | SERIOUS | FATAL           (kb_accidents)
    accident_type   keyword[]  FIRE | EXPLOSION | ASPHYXIATION | FALL ...    (kb_accidents)
    hazard          keyword[]  MECHANICAL | ELECTRICAL | CHEMICAL | ENVIRONMENTAL | HUMAN
    work_type       keyword[]  GENERAL | HOT | CONFINED | HEIGHT | ELECTRICAL | LIFTING
    cas             keyword[]  CAS 번호 (관련 물질)
    ghs             keyword[]  GHS 분류 코드
    year            int        발생·발행 연도
    guide_no        keyword    KOSHA GUIDE 번호 (kb_guides)
    flagged         bool       지시문처럼 보이는 문장 포함 의심 → 검토 전 검색 제외
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

K = models.PayloadSchemaType.KEYWORD
I = models.PayloadSchemaType.INTEGER

COMMON_INDEXES = {
    "kb_source_id": I,
    "kb_chunk_id": I,
    "source_type": K,
    "site_id": I,
    "hazard": K,
    "work_type": K,
    "cas": K,
    "year": I,
    "flagged": models.PayloadSchemaType.BOOL,   # 인젝션 의심 원본. 검색 시 항상 flagged=false 조건
}

COLLECTIONS: dict[str, dict[str, models.PayloadSchemaType]] = {
    "kb_accidents": {
        **COMMON_INDEXES,
        "severity": K,
        "accident_type": K,
        "ghs": K,
    },
    "kb_guides": {
        **COMMON_INDEXES,
        "guide_no": K,
    },
}


def point_id(source_type: str, external_id: str, chunk_index: int) -> str:
    """같은 청크는 항상 같은 id -> 다시 적재해도 중복 없이 덮어쓴다. kb_chunks.qdrant_point_id와 같은 값."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_type}:{external_id}:{chunk_index}"))


def create(client: QdrantClient, name: str, indexes: dict, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if exists:
        print(f"[skip] {name} 이미 있음")
    else:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        print(f"[create] {name} (dense {EMBEDDING_DIM} + sparse)")
    for field, schema in indexes.items():
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
    print(f"[index] {name}: {', '.join(indexes)}")


def check(client: QdrantClient) -> None:
    """테스트 점 하나로 5축 필터가 동작하는지 확인하고 지운다."""
    name = "kb_accidents"
    pid = point_id("ACCIDENT_REPORT", "__check__", 0)
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = 1.0
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=pid,
                vector={
                    "dense": vec,
                    "sparse": models.SparseVector(indices=[1, 7], values=[1.0, 0.5]),
                },
                payload={
                    "kb_source_id": 0, "kb_chunk_id": 0, "source_type": "ACCIDENT_REPORT", "site_id": 0,
                    "title": "점검용", "severity": "FATAL", "accident_type": ["FIRE", "EXPLOSION"],
                    "hazard": ["CHEMICAL"], "work_type": ["HOT"], "cas": ["108-88-3"], "year": 2024, "flagged": False,
                },
            )
        ],
        wait=True,
    )
    my_site = 1
    flt = models.Filter(
        must=[
            models.FieldCondition(key="work_type", match=models.MatchAny(any=["HOT"])),
            models.FieldCondition(key="accident_type", match=models.MatchAny(any=["FIRE", "EXPLOSION"])),
            models.FieldCondition(key="site_id", match=models.MatchAny(any=[0, my_site])),
            models.FieldCondition(key="flagged", match=models.MatchValue(value=False)),
        ]
    )
    hits = client.query_points(name, query=vec, using="dense", query_filter=flt, limit=3).points
    ok = any(str(h.id) == pid for h in hits)
    client.delete(collection_name=name, points_selector=models.PointIdsList(points=[pid]), wait=True)
    print("[check] 필터 검색", "통과" if ok else "실패")
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not QDRANT_API_KEY:
        sys.exit("QDRANT_API_KEY가 없습니다. 쓰기 키 없이 컬렉션을 만들 수 없습니다.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    for name, indexes in COLLECTIONS.items():
        create(client, name, indexes, args.recreate)
    if args.check:
        check(client)


if __name__ == "__main__":
    main()
