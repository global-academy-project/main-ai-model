"""법령 청크(05_chunks.jsonl)를 지식 DB 장부(agent_kb)와 Qdrant(kb_guides)에 적재한다.

실행 위치: 프로젝트 루트 (python -m scripts.ingest_law ...)

    python -m scripts.ingest_law --dry-run            # 계획만 보기 (DB·네트워크 접속 없음, 파일 쓰기 없음)
    python -m scripts.ingest_law --limit 5            # 앞 5건만 시험 적재
    python -m scripts.ingest_law                      # 전체 적재 (+ 끝에 검증)
    python -m scripts.ingest_law --verify-only        # 적재 없이 검증만
    python -m scripts.ingest_law --force              # 이미 EMBEDDED여도 다시 적재

흐름
  1. 청크 파일 검사 (06_validation.json의 ok, 필드, 중복)
  2. natural_key → chunk_index 대응표(chunk_index_map.json) 갱신  ← point_id가 바뀌지 않게 보관한다
  3. kb_sources upsert(PENDING) → kb_chunks upsert → kb_chunks.id 확보 → 상태 PARSED
  4. POST {kb-ingest}/ingest/chunks (32개씩) : 임베딩과 Qdrant upsert는 kb-ingest가 한다
  5. 전부 성공하면 EMBEDDED (실패하면 FAILED + error_message)
  6. 검증: 장부 행 수, Qdrant 점 수, 표본 점의 payload id·벡터 차원

id 규칙 (kb_chunks.qdrant_point_id 주석, scripts/init_qdrant.py의 point_id와 같은 공식)
  point_id = uuid5(NAMESPACE_URL, "LAW:{external_id}:{chunk_index}")
  external_id = 법령 식별자(natural_key.law_key). 개정돼도 바뀌지 않는다.
  chunk_index = 대응표로 고정한 번호. 처음엔 문서 순서(0부터), 개정 때 새 키는 이어서 번호를 받는다.

환경 변수 (값은 문서·저장소에 넣지 않는다)
  KB_DB_HOST(기본 127.0.0.1)  KB_DB_PORT(기본 60010)  KB_DATABASE(기본 agent_kb)
  KB_INGEST_PASSWORD          kb_ingest 계정 비밀번호
  KB_INGEST_URL(기본 http://127.0.0.1:8001)  KB_INGEST_API_TOKEN
  QDRANT_URL(기본 http://127.0.0.1:60011)    QDRANT_READ_ONLY_API_KEY (없으면 QDRANT_API_KEY)
  EMBEDDING_MODEL(기본 BAAI/bge-m3)          EMBEDDING_DIM(기본 1024)

종료 코드: 0 성공 / 1 적재·검증 실패 / 2 입력·설정 오류
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.chunking.law_ingest.fields import to_ingest_item

SOURCE_TYPE = "LAW"
COLLECTION = "kb_guides"
# 기본 요청 크기: 임베딩 서버 EMBEDDING_MAX_BATCH_SIZE(compose 기본 8)와 맞춘다
BATCH_SIZE = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "8"))
MAX_BATCH_SIZE = 32  # /ingest/chunks 한 요청의 상한 (그 이상은 서버가 422)
STATUS_EMBEDDED = "EMBEDDED"
SAMPLE_SIZE = 3


class InputError(Exception):
    """입력 파일·설정 문제 (종료 코드 2)."""


class IngestError(Exception):
    """적재·검증 실패 (종료 코드 1)."""


# ---------------------------------------------------------------- id 규칙


def point_id_for(external_id: str, chunk_index: int) -> str:
    """같은 청크는 항상 같은 id. init_qdrant.point_id와 같은 공식이다."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_TYPE}:{external_id}:{chunk_index}"))


def natural_key_str(record: dict[str, Any]) -> str:
    key = record["natural_key"]
    return f"{key['law_key']}|{key['article_key']}|{key['part']}"


# ---------------------------------------------------------------- 청크 파일 검사


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputError(f"청크 파일이 없습니다: {path}")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputError(f"{path} {number}번째 줄이 JSON이 아닙니다: {exc}") from exc
    return records


def check_records(records: Sequence[dict[str, Any]], validation_path: Path) -> str:
    """적재해도 되는 청크인지 검사하고 external_id(법령 식별자)를 돌려준다."""
    if not records:
        raise InputError("청크가 없습니다.")
    if not validation_path.is_file():
        raise InputError(f"검증 결과 파일이 없습니다: {validation_path} (청킹의 06_validation.json)")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("ok") is not True:
        raise InputError("06_validation.json의 ok가 true가 아닙니다. 검증을 통과한 청크만 적재합니다.")
    expected = validation.get("summary", {}).get("chunks")
    if expected is not None and expected != len(records):
        raise InputError(f"청크 수가 검증 결과와 다릅니다: 파일 {len(records)}, 검증 {expected}")

    law_keys = {record["natural_key"]["law_key"] for record in records}
    if len(law_keys) != 1:
        raise InputError(f"법령이 하나가 아닙니다: {sorted(law_keys)}")
    external_id = law_keys.pop()
    if not external_id or len(external_id) > 100:
        raise InputError("law_key는 1~100자여야 합니다 (kb_sources.external_id).")

    shas = {record["law_meta"]["source_sha256"] for record in records}
    if len(shas) != 1:
        raise InputError("원본 해시(source_sha256)가 하나가 아닙니다.")

    seen: set[str] = set()
    for record in records:
        key = natural_key_str(record)
        if key in seen:
            raise InputError(f"natural_key가 중복됩니다: {key}")
        seen.add(key)
        where = f"({key})"
        if record.get("source_type") != SOURCE_TYPE or record.get("collection") != COLLECTION:
            raise InputError(f"source_type/collection이 LAW/{COLLECTION}가 아닙니다 {where}")
        if record.get("site_id") != 0 or record.get("flagged") is not False:
            raise InputError(f"공공 데이터는 site_id=0, flagged=false여야 합니다 {where}")
        for field in ("text", "embed_text"):
            if not str(record.get(field, "")).strip():
                raise InputError(f"{field}가 비어 있습니다 {where}")
        if len(str(record.get("locator", ""))) > 100:
            raise InputError(f"locator가 100자를 넘습니다 {where}")
    return external_id


# ---------------------------------------------------------------- chunk_index 대응표


def load_index_map(path: Path, external_id: str) -> dict[str, int]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source_type") != SOURCE_TYPE or data.get("external_id") != external_id:
        raise InputError(f"대응표가 다른 법령의 것입니다: {path}")
    indexes = data.get("indexes")
    if not isinstance(indexes, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indexes.values()
    ):
        raise InputError(f"대응표 형식이 올바르지 않습니다: {path}")
    return dict(indexes)


def assign_indexes(records: Sequence[dict[str, Any]], existing: dict[str, int]) -> dict[str, int]:
    """대응표에 없는 natural_key에만 새 번호를 준다. 기존 번호는 바꾸거나 재사용하지 않는다."""
    result = dict(existing)
    used = set(result.values())
    if len(used) != len(result):
        raise InputError("대응표에 같은 chunk_index가 두 번 나옵니다.")
    next_index = max(used) + 1 if used else 0
    for record in records:
        key = natural_key_str(record)
        if key not in result:
            result[key] = next_index
            next_index += 1
    return result


def save_index_map(path: Path, external_id: str, indexes: dict[str, int]) -> None:
    body = {"source_type": SOURCE_TYPE, "external_id": external_id, "indexes": indexes}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


# ---------------------------------------------------------------- 장부(agent_kb)


@dataclass(frozen=True)
class SourceMeta:
    external_id: str
    title: str
    file_path: str | None
    content_sha256: str
    published_on: date | None
    publisher: str | None
    license: str | None


def build_source_meta(
    records: Sequence[dict[str, Any]], external_id: str, *, publisher: str | None, license_: str | None
) -> SourceMeta:
    first = records[0]
    promulgated = first["law_meta"].get("promulgation_date")
    return SourceMeta(
        external_id=external_id,
        title=str(first["title"])[:300],
        file_path=first.get("file_path"),
        content_sha256=first["law_meta"]["source_sha256"],
        published_on=date.fromisoformat(promulgated) if promulgated else None,
        publisher=publisher,
        license=license_,
    )


class Ledger:
    """kb_sources, kb_chunks 읽기·쓰기. 커밋은 호출하는 쪽이 정하는 지점에서 한다."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def find_source(self, external_id: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_sha256, status, chunk_count FROM kb_sources "
                "WHERE source_type = %s AND external_id = %s",
                (SOURCE_TYPE, external_id),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row is None:
            return None
        return {"id": row[0], "content_sha256": row[1], "status": row[2], "chunk_count": row[3]}

    def upsert_source(self, meta: SourceMeta) -> int:
        existing = self.find_source(meta.external_id)
        with self.conn.cursor() as cur:
            if existing is None:
                cur.execute(
                    "INSERT INTO kb_sources (source_type, external_id, site_id, title, url, license, publisher, "
                    "published_on, file_path, content_sha256, status, flagged) "
                    "VALUES (%s, %s, NULL, %s, NULL, %s, %s, %s, %s, %s, 'PENDING', FALSE)",
                    (SOURCE_TYPE, meta.external_id, meta.title, meta.license, meta.publisher,
                     meta.published_on, meta.file_path, meta.content_sha256),
                )
                source_id = cur.lastrowid
            else:
                source_id = existing["id"]
                cur.execute(
                    "UPDATE kb_sources SET title = %s, license = %s, publisher = %s, published_on = %s, "
                    "file_path = %s, content_sha256 = %s, status = 'PENDING', error_message = NULL "
                    "WHERE id = %s",
                    (meta.title, meta.license, meta.publisher, meta.published_on, meta.file_path,
                     meta.content_sha256, source_id),
                )
        self.conn.commit()
        return int(source_id)

    def upsert_chunks(self, source_id: int, rows: Sequence[dict[str, Any]]) -> dict[int, int]:
        """(source_id, chunk_index)로 upsert한다. 기존 행의 id는 유지된다. {chunk_index: kb_chunks.id}."""
        with self.conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    "INSERT INTO kb_chunks (source_id, chunk_index, collection_name, qdrant_point_id, locator, text) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE collection_name = %s, qdrant_point_id = %s, locator = %s, text = %s",
                    (source_id, row["chunk_index"], COLLECTION, row["point_id"], row["locator"], row["text"],
                     COLLECTION, row["point_id"], row["locator"], row["text"]),
                )
        self.conn.commit()
        return self.chunk_id_map(source_id)

    def set_status(
        self,
        source_id: int,
        status: str,
        *,
        chunk_count: int | None = None,
        embedding_model: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_sources SET status = %s, chunk_count = COALESCE(%s, chunk_count), "
                "embedding_model = COALESCE(%s, embedding_model), error_message = %s WHERE id = %s",
                (status, chunk_count, embedding_model, error, source_id),
            )
        self.conn.commit()

    def count_chunks(self, source_id: int) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM kb_chunks WHERE source_id = %s", (source_id,))
            (count,) = cur.fetchone()
        self.conn.commit()
        return int(count)

    def chunk_id_map(self, source_id: int) -> dict[int, int]:
        """{chunk_index: kb_chunks.id}"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_index, id FROM kb_chunks WHERE source_id = %s", (source_id,))
            found = {int(index): int(chunk_id) for index, chunk_id in cur.fetchall()}
        self.conn.commit()
        return found


# ---------------------------------------------------------------- 적재 요청


def batched(items: Sequence[Any], size: int = BATCH_SIZE) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def build_items(
    records: Sequence[dict[str, Any]],
    indexes: dict[str, int],
    external_id: str,
    source_id: int,
    chunk_ids: dict[int, int],
) -> list[dict[str, Any]]:
    """id가 채워진 /ingest/chunks 항목. kb_chunk_id는 장부(kb_chunks.id)의 값이다."""
    items = []
    for record in records:
        index = indexes[natural_key_str(record)]
        filled = dict(record)
        filled["kb_source_id"] = source_id
        filled["kb_chunk_id"] = chunk_ids[index]
        filled["point_id"] = point_id_for(external_id, index)
        items.append(to_ingest_item(filled))
    ids = [item["point_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise IngestError("point_id가 중복됩니다.")
    return items


def post_batch(
    session: Any,
    url: str,
    token: str,
    items: Sequence[dict[str, Any]],
    *,
    retries: int = 2,
    timeout: float = 300.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """한 요청(최대 32개)을 올린다. 서버·연결 오류는 재시도하고, 요청 오류(4xx)는 바로 실패한다.

    같은 point_id는 덮어쓰기라 재시도해도 안전하다.
    """
    endpoint = f"{url.rstrip('/')}/ingest/chunks"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"collection": COLLECTION, "chunks": list(items)}
    last = "알 수 없음"
    for attempt in range(retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=body, timeout=timeout)
        except OSError as exc:  # requests의 연결·시간 초과 예외도 OSError 계열이다
            last = f"연결 오류: {type(exc).__name__}"
        else:
            if response.status_code == 200:
                indexed = response.json().get("indexed")
                if indexed != len(items):
                    raise IngestError(f"적재 수가 다릅니다: 요청 {len(items)}, 응답 {indexed}")
                return int(indexed)
            last = f"HTTP {response.status_code} {response.text[:300]}"
            if response.status_code < 500:
                raise IngestError(f"적재 요청이 거부되었습니다: {last}")
        if attempt < retries:
            sleep(5.0 * (attempt + 1))
    raise IngestError(f"적재 요청이 계속 실패했습니다: {last}")


# ---------------------------------------------------------------- 검증


def sample_positions(count: int, size: int = SAMPLE_SIZE) -> list[int]:
    """처음·가운데·끝 위주로 표본 위치를 고른다."""
    if count <= size:
        return list(range(count))
    return sorted({0, count // 2, count - 1})[:size]


def qdrant_headers(key: str) -> dict[str, str]:
    return {"api-key": key, "Content-Type": "application/json"}


def verify(
    ledger: Ledger,
    session: Any,
    qdrant_url: str,
    qdrant_key: str,
    *,
    source_id: int,
    expected: int,
    full: bool,
    samples: Sequence[dict[str, Any]],
    embedding_dim: int,
) -> list[str]:
    """문제 목록을 돌려준다(비어 있으면 통과). samples: {point_id, chunk_id, locator} 목록."""
    problems: list[str] = []
    base = qdrant_url.rstrip("/")

    db_count = ledger.count_chunks(source_id)
    if (db_count != expected) if full else (db_count < expected):
        problems.append(f"장부 청크 수: 기대 {expected}, 실제 {db_count}")

    response = session.post(
        f"{base}/collections/{COLLECTION}/points/count",
        headers=qdrant_headers(qdrant_key),
        json={
            "exact": True,
            "filter": {
                "must": [
                    {"key": "source_type", "match": {"value": SOURCE_TYPE}},
                    {"key": "kb_source_id", "match": {"value": source_id}},
                ]
            },
        },
        timeout=30,
    )
    if response.status_code != 200:
        problems.append(f"Qdrant 점 수 조회 실패: HTTP {response.status_code}")
    else:
        qdrant_count = response.json()["result"]["count"]
        if (qdrant_count != expected) if full else (qdrant_count < expected):
            problems.append(f"Qdrant 점 수: 기대 {expected}, 실제 {qdrant_count}")

    if samples:
        response = session.post(
            f"{base}/collections/{COLLECTION}/points",
            headers=qdrant_headers(qdrant_key),
            json={"ids": [s["point_id"] for s in samples], "with_payload": True, "with_vector": ["dense"]},
            timeout=30,
        )
        if response.status_code != 200:
            problems.append(f"Qdrant 표본 조회 실패: HTTP {response.status_code}")
        else:
            found = {str(point["id"]): point for point in response.json()["result"]}
            for sample in samples:
                point = found.get(sample["point_id"])
                label = f"{sample['locator']} ({sample['point_id'][:8]})"
                if point is None:
                    problems.append(f"표본 점이 Qdrant에 없습니다: {label}")
                    continue
                payload = point.get("payload") or {}
                if payload.get("kb_source_id") != source_id or payload.get("kb_chunk_id") != sample["chunk_id"]:
                    problems.append(
                        f"표본 payload id가 장부와 다릅니다: {label} "
                        f"(payload {payload.get('kb_source_id')}/{payload.get('kb_chunk_id')}, "
                        f"장부 {source_id}/{sample['chunk_id']})"
                    )
                if payload.get("source_type") != SOURCE_TYPE or payload.get("flagged") is not False:
                    problems.append(f"표본 payload의 source_type/flagged가 이상합니다: {label}")
                if not str(payload.get("text", "")).strip():
                    problems.append(f"표본 payload.text가 비어 있습니다: {label}")
                vector = (point.get("vector") or {}).get("dense")
                if not isinstance(vector, list) or len(vector) != embedding_dim:
                    problems.append(f"표본 벡터 차원이 {embedding_dim}가 아닙니다: {label}")
    return problems


# ---------------------------------------------------------------- 실행


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise InputError(f"{name} 환경 변수가 필요합니다.")
    return value


def connect_ledger() -> Ledger:
    import pymysql  # 지연 import: 단위 시험과 --dry-run에는 필요 없다

    conn = pymysql.connect(
        host=os.getenv("KB_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("KB_DB_PORT", "60010")),
        user="kb_ingest",
        password=env("KB_INGEST_PASSWORD"),
        database=os.getenv("KB_DATABASE", "agent_kb"),
        charset="utf8mb4",
        autocommit=False,
    )
    return Ledger(conn)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="법령 청크를 agent_kb와 Qdrant에 적재한다.")
    parser.add_argument("--chunks", type=Path, default=Path("out/law/05_chunks.jsonl"))
    parser.add_argument("--validation", type=Path, default=None, help="기본: 청크 파일과 같은 폴더의 06_validation.json")
    parser.add_argument("--index-map", type=Path, default=None, help="기본: 청크 파일과 같은 폴더의 chunk_index_map.json")
    parser.add_argument("--limit", type=int, default=0, help="앞 N건만 적재 (시험용, 0이면 전체)")
    parser.add_argument("--dry-run", action="store_true", help="계획만 출력 (접속·파일 쓰기 없음)")
    parser.add_argument("--verify-only", action="store_true", help="적재 없이 검증만")
    parser.add_argument("--force", action="store_true", help="이미 EMBEDDED여도 다시 적재")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"한 요청의 청크 수 (기본 {BATCH_SIZE}, 1~{MAX_BATCH_SIZE}). 임베딩 서버의 EMBEDDING_MAX_BATCH_SIZE 이하여야 한다",
    )
    parser.add_argument("--publisher", default=None, help="kb_sources.publisher (기본 비움)")
    parser.add_argument("--license", dest="license_", default=None, help="kb_sources.license (기본 비움)")
    parser.add_argument("--ingest-url", default=os.getenv("KB_INGEST_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:60011"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다.")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size는 1~{MAX_BATCH_SIZE} 사이여야 합니다.")
    base = args.chunks.parent
    args.validation = args.validation or base / "06_validation.json"
    args.index_map = args.index_map or base / "chunk_index_map.json"
    return args


def run(args: argparse.Namespace) -> int:
    records = load_records(args.chunks)
    external_id = check_records(records, args.validation)
    saved_map = load_index_map(args.index_map, external_id)
    indexes = assign_indexes(records, saved_map)
    selected = records[: args.limit] if args.limit else list(records)
    full = len(selected) == len(records)
    meta = build_source_meta(records, external_id, publisher=args.publisher, license_=args.license_)
    new_keys = len(indexes) - len(saved_map)

    print(f"법령: {meta.title} (external_id={external_id})")
    print(f"청크 {len(records)}건 중 {len(selected)}건 대상 / 요청 {len(list(batched(selected, args.batch_size)))}번 ({args.batch_size}개씩)")
    print(f"chunk_index 대응표: 새 번호 {new_keys}개, 파일 {args.index_map}")

    if args.dry_run:
        first = selected[0]
        print("\n[dry-run] 접속·파일 쓰기 없이 종료합니다.")
        print(f"  첫 청크 {first['locator']}: chunk_index={indexes[natural_key_str(first)]}, "
              f"point_id={point_id_for(external_id, indexes[natural_key_str(first)])}")
        print(f"  source: sha256={meta.content_sha256[:12]}…, published_on={meta.published_on}, "
              f"file_path={meta.file_path}")
        return 0

    import requests  # 지연 import

    # 쓰기를 시작하기 전에 필요한 설정을 모두 확인한다 (중간에 빠져 상태가 어정쩡해지지 않게)
    token = None if args.verify_only else env("KB_INGEST_API_TOKEN")
    qdrant_key = os.getenv("QDRANT_READ_ONLY_API_KEY") or env("QDRANT_API_KEY")
    session = requests.Session()
    ledger = connect_ledger()

    existing = ledger.find_source(external_id)
    if args.verify_only:
        if existing is None:
            raise InputError("장부에 이 법령이 없습니다. 먼저 적재하세요.")
        source_id = existing["id"]
        chunk_ids = ledger.chunk_id_map(source_id)
    elif (
        existing is not None
        and existing["status"] == STATUS_EMBEDDED
        and existing["content_sha256"] == meta.content_sha256
        and existing["chunk_count"] == len(records)
        and not args.force
    ):
        print("이미 같은 원본이 EMBEDDED 상태입니다. 건너뛰고 검증만 합니다 (--force로 다시 적재).")
        source_id = existing["id"]
        chunk_ids = ledger.chunk_id_map(source_id)
    else:
        if existing is not None and existing["content_sha256"] != meta.content_sha256:
            print("원본 해시가 바뀌었습니다(개정). 같은 chunk_index는 덮어쓰고, 새 키는 새 번호를 받습니다.")
        save_index_map(args.index_map, external_id, indexes)
        source_id = ledger.upsert_source(meta)
        rows = [
            {
                "chunk_index": indexes[natural_key_str(r)],
                "point_id": point_id_for(external_id, indexes[natural_key_str(r)]),
                "locator": r["locator"],
                "text": r["text"],
            }
            for r in selected
        ]
        chunk_ids = ledger.upsert_chunks(source_id, rows)
        ledger.set_status(source_id, "PARSED")
        print(f"장부: kb_source_id={source_id}, kb_chunks {len(rows)}행 반영")
        current = {indexes[natural_key_str(r)] for r in records}
        stale = sorted(set(chunk_ids) - current)
        if stale:
            print(f"[주의] 이번 청크 파일에 없는 chunk_index가 장부에 {len(stale)}개 남아 있습니다 (삭제하지 않음): {stale[:10]}")

        items = build_items(selected, indexes, external_id, source_id, chunk_ids)
        started = time.monotonic()
        done = 0
        try:
            for number, batch in enumerate(batched(items, args.batch_size), start=1):
                done += post_batch(session, args.ingest_url, token, batch)
                print(f"  [{number}] {done}/{len(items)}건 적재 (경과 {time.monotonic() - started:.0f}초)")
        except (IngestError, OSError) as exc:
            message = f"적재 중단 ({done}/{len(items)}건 완료): {exc}"
            ledger.set_status(source_id, "FAILED", error=message[:1000])
            print(f"[실패] {message}")
            return 1
        if full:
            ledger.set_status(source_id, STATUS_EMBEDDED, chunk_count=len(records),
                              embedding_model=args.embedding_model)
            print(f"상태: EMBEDDED (chunk_count={len(records)}, model={args.embedding_model})")
        else:
            print(f"일부({len(selected)}건)만 적재해서 상태는 PARSED로 둡니다. 전체 적재 때 EMBEDDED가 됩니다.")

    sample_records = [selected[i] for i in sample_positions(len(selected))]
    samples = []
    for record in sample_records:
        index = indexes[natural_key_str(record)]
        if index in chunk_ids:
            samples.append(
                {"point_id": point_id_for(external_id, index), "chunk_id": chunk_ids[index], "locator": record["locator"]}
            )
    problems = verify(
        ledger, session, args.qdrant_url, qdrant_key,
        source_id=source_id, expected=len(selected), full=full,
        samples=samples, embedding_dim=args.embedding_dim,
    )
    if problems:
        print("\n[검증 실패]")
        for problem in problems:
            print(f"  - {problem}")
        print("점이 모자라면 --force로 다시 적재하세요 (같은 id로 덮어씁니다). 기대보다 많으면 개정으로 사라진 조의 옛 청크일 수 있습니다 (삭제는 이 스크립트가 하지 않습니다).")
        return 1
    print(f"\n[검증 통과] 장부와 Qdrant 모두 {len(selected)}건{'' if full else ' 이상'}, 표본 {len(samples)}건의 id·벡터 확인")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        sys.exit(run(parse_args(argv)))
    except InputError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
