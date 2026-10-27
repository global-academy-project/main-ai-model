"""FastAPI 쪽 저장소 점검: 지식 DB(agent_kb) 계정 권한 + Qdrant 컬렉션·키 권한.

담당: C 작성. 스키마 PR 확인용이자, 이후 배포 환경 점검용.

    python -m scripts.verify_kb                 # MySQL + Qdrant 둘 다
    python -m scripts.verify_kb --skip-qdrant   # 지식 DB만
    python -m scripts.verify_kb --skip-mysql    # Qdrant만

확인하는 것
  지식 DB  : 테이블 6개, kb_reader 읽기만, kb_ingest 읽기·쓰기(DDL 불가), 둘 다 업무 DB 접근 불가, CHECK 동작
  Qdrant   : 키 없이 거부, 컬렉션 2개(dense 차원·sparse·payload 색인), 읽기 키는 검색만, 쓰기 키는 적재 가능
쓰기 확인은 트랜잭션 롤백 / 테스트 점 삭제로 흔적을 남기지 않는다.

환경 변수 (로컬 기본값은 APPLY.md 2번의 mysql-local 기준)
    KB_DB_HOST=localhost  KB_DB_PORT=3307  KB_DATABASE=agent_kb  BIZ_DATABASE=ocrdb
    KB_READER_PASSWORD  KB_INGEST_PASSWORD
    QDRANT_URL=http://localhost:6333  QDRANT_API_KEY  QDRANT_READ_ONLY_API_KEY  EMBEDDING_DIM=1024
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import requests

PASS: list[str] = []
FAIL: list[str] = []


def ok(cond: bool, desc: str, detail: str = "") -> None:
    if cond:
        PASS.append(desc)
        print(f"  [통과] {desc}")
    else:
        FAIL.append(desc)
        print(f"  [실패] {desc}" + (f"  ({detail})" if detail else ""))


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        sys.exit(f"{name} 환경 변수가 필요합니다.")
    return v


# ---------------------------------------------------------------- 지식 DB
EXPECTED_TABLES = {"kb_sources", "kb_chunks", "msds_cache", "eval_cases", "eval_runs", "eval_results"}


def mysql_error_code(fn) -> int | None:
    """fn 실행 결과 MySQL 오류 번호 (성공이면 None)."""
    import pymysql

    try:
        fn()
        return None
    except pymysql.MySQLError as e:  # OperationalError, IntegrityError, ProgrammingError ...
        return e.args[0] if e.args else -1


def verify_mysql() -> None:
    import pymysql

    host = os.getenv("KB_DB_HOST", "localhost")
    port = int(os.getenv("KB_DB_PORT", "3307"))
    kb = os.getenv("KB_DATABASE", "agent_kb")
    biz = os.getenv("BIZ_DATABASE", "ocrdb")

    def connect(user: str, pw_env: str):
        return pymysql.connect(host=host, port=port, user=user, password=env(pw_env), database=kb,
                               charset="utf8mb4", autocommit=False)

    print(f"\n=== 지식 DB {kb} ({host}:{port})")

    # kb_reader: 에이전트 실행 계정
    r = connect("kb_reader", "KB_READER_PASSWORD")
    with r.cursor() as c:
        c.execute("SHOW TABLES")
        tables = {row[0] for row in c.fetchall()}
        ok(EXPECTED_TABLES <= tables, f"테이블 6개 있음 ({len(tables & EXPECTED_TABLES)}/6)",
           f"없음: {sorted(EXPECTED_TABLES - tables)}")
        c.execute("SELECT COUNT(*) FROM kb_sources")
        ok(True, "kb_reader 조회 가능")

        code = mysql_error_code(lambda: c.execute(
            "INSERT INTO msds_cache (cas_no, raw_json) VALUES ('108-88-3', JSON_OBJECT())"))
        ok(code == 1142, "kb_reader 쓰기 거부 (1142)", f"결과 {code}")
        r.rollback()

        code = mysql_error_code(lambda: c.execute(f"SELECT 1 FROM `{biz}`.users LIMIT 1"))
        ok(code == 1142, f"kb_reader 업무 DB({biz}) 접근 거부 (1142)", f"결과 {code}")
    r.close()

    # kb_ingest: 적재 계정
    w = connect("kb_ingest", "KB_INGEST_PASSWORD")
    with w.cursor() as c:
        ext = f"__verify__{uuid.uuid4().hex[:8]}"
        c.execute(
            "INSERT INTO kb_sources (source_type, external_id, title, url, license, publisher) "
            "VALUES ('KOSHA_GUIDE', %s, '점검용', 'https://example.invalid', 'KOGL-1', '점검')", (ext,))
        src_id = c.lastrowid
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"KOSHA_GUIDE:{ext}:0"))
        c.execute(
            "INSERT INTO kb_chunks (source_id, chunk_index, collection_name, qdrant_point_id, text) "
            "VALUES (%s, 0, 'kb_guides', %s, '점검용 청크')", (src_id, pid))
        ok(True, "kb_ingest 원본·청크 적재 가능 (롤백)")
        w.rollback()

        code = mysql_error_code(lambda: c.execute(
            "INSERT INTO kb_sources (source_type, external_id, title) VALUES ('KOSHA_GUIDE', 'no-src', '출처 없음')"))
        ok(code == 3819, "출처 없는 공공 데이터 거부 (CHECK 3819)", f"결과 {code}")
        w.rollback()

        code = mysql_error_code(lambda: c.execute(
            "INSERT INTO kb_sources (source_type, external_id, site_id, title) VALUES ('NEAR_MISS', 'nm-0', 0, '현장 0')"))
        ok(code == 3819, "현장 id 0 거부 (Qdrant의 '공공'과 겹침, CHECK 3819)", f"결과 {code}")
        w.rollback()

        code = mysql_error_code(lambda: c.execute("CREATE TABLE _verify_tmp (id INT)"))
        if code is None:
            c.execute("DROP TABLE _verify_tmp")
        ok(code == 1142, "kb_ingest 테이블 생성(DDL) 거부 (1142)", f"결과 {code}")

        code = mysql_error_code(lambda: c.execute(f"SELECT 1 FROM `{biz}`.users LIMIT 1"))
        ok(code == 1142, f"kb_ingest 업무 DB({biz}) 접근 거부 (1142)", f"결과 {code}")
    w.close()


# ---------------------------------------------------------------- Qdrant
COMMON_INDEXES = {"kb_source_id", "kb_chunk_id", "source_type", "site_id", "hazard", "work_type", "cas", "year", "flagged"}
INDEXES = {
    "kb_accidents": COMMON_INDEXES | {"severity", "accident_type", "ghs"},
    "kb_guides": COMMON_INDEXES | {"guide_no"},
}


def verify_qdrant() -> None:
    url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    wkey = env("QDRANT_API_KEY")
    rkey = env("QDRANT_READ_ONLY_API_KEY")
    dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    print(f"\n=== Qdrant ({url})")

    ok(wkey != rkey, "쓰기 키와 읽기 키가 다름")

    def call(method: str, path: str, key: str | None, body: dict | None = None) -> requests.Response:
        headers = {"api-key": key} if key else {}
        return requests.request(method, f"{url}{path}", headers=headers, json=body, timeout=10)

    res = call("GET", "/collections", None)
    ok(res.status_code == 401, "키 없이 접근 거부 (401)", f"결과 {res.status_code}")

    for name, need in INDEXES.items():
        res = call("GET", f"/collections/{name}", rkey)
        if res.status_code != 200:
            ok(False, f"{name} 있음", f"결과 {res.status_code}. 먼저 python -m scripts.init_qdrant 실행")
            continue
        info = res.json()["result"]
        params = info["config"]["params"]
        dense = (params.get("vectors") or {}).get("dense") or {}
        sparse = (params.get("sparse_vectors") or {}).get("sparse")
        ok(dense.get("size") == dim and dense.get("distance") == "Cosine",
           f"{name} dense {dim} 코사인", f"현재 {dense}")
        ok(sparse is not None and sparse.get("modifier") in ("idf", "Idf"),
           f"{name} sparse(IDF)", f"현재 {sparse}")
        have = set((info.get("payload_schema") or {}).keys())
        ok(need <= have, f"{name} payload 색인 {len(need)}개", f"없음: {sorted(need - have)}")

    # 읽기 키: 검색은 되고 쓰기는 안 됨
    vec = [0.0] * dim
    vec[0] = 1.0
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "ACCIDENT_REPORT:__verify__:0"))
    point = {
        "id": pid,
        "vector": {"dense": vec, "sparse": {"indices": [1, 7], "values": [1.0, 0.5]}},
        "payload": {"kb_source_id": 0, "kb_chunk_id": 0, "source_type": "ACCIDENT_REPORT", "site_id": 0,
                    "title": "점검용", "severity": "FATAL", "accident_type": ["FIRE"], "hazard": ["CHEMICAL"],
                    "work_type": ["HOT"], "cas": ["108-88-3"], "year": 2024, "flagged": False},
    }
    query = {
        "query": vec, "using": "dense", "limit": 3,
        "filter": {"must": [
            {"key": "work_type", "match": {"any": ["HOT"]}},
            {"key": "site_id", "match": {"any": [0, 1]}},
            {"key": "flagged", "match": {"value": False}},
        ]},
    }
    res = call("POST", "/collections/kb_accidents/points/query", rkey, query)
    ok(res.status_code == 200, "읽기 키로 검색 가능", f"결과 {res.status_code}")
    res = call("PUT", "/collections/kb_accidents/points?wait=true", rkey, {"points": [point]})
    ok(res.status_code == 403, "읽기 키로 적재 거부 (403)", f"결과 {res.status_code}")
    res = call("DELETE", "/collections/kb_guides", rkey)
    ok(res.status_code == 403, "읽기 키로 컬렉션 삭제 거부 (403)", f"결과 {res.status_code}")

    # 쓰기 키: 적재 → 필터 검색 → 삭제
    res = call("PUT", "/collections/kb_accidents/points?wait=true", wkey, {"points": [point]})
    ok(res.status_code == 200, "쓰기 키로 적재 가능", f"결과 {res.status_code} {res.text[:200]}")
    res = call("POST", "/collections/kb_accidents/points/query", rkey, query)
    hits = [str(p["id"]) for p in res.json().get("result", {}).get("points", [])] if res.ok else []
    ok(pid in hits, "적재한 점이 5축 필터 검색에 걸림 (읽기 키)", f"결과 {res.status_code} {hits}")
    flagged_query = dict(query, filter={"must": [{"key": "flagged", "match": {"value": True}}]})
    res = call("POST", "/collections/kb_accidents/points/query", rkey, flagged_query)
    hits = [str(p["id"]) for p in res.json().get("result", {}).get("points", [])] if res.ok else []
    ok(pid not in hits, "flagged 조건이 걸러냄")
    res = call("POST", "/collections/kb_accidents/points/delete?wait=true", wkey, {"points": [pid]})
    ok(res.status_code == 200, "테스트 점 삭제", f"결과 {res.status_code}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mysql", action="store_true")
    ap.add_argument("--skip-qdrant", action="store_true")
    args = ap.parse_args()

    if not args.skip_mysql:
        try:
            verify_mysql()
        except Exception as e:  # 접속 실패 등
            ok(False, "지식 DB 점검 실행", repr(e))
    if not args.skip_qdrant:
        try:
            verify_qdrant()
        except Exception as e:
            ok(False, "Qdrant 점검 실행", repr(e))

    print(f"\n결과: 통과 {len(PASS)} / 실패 {len(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
