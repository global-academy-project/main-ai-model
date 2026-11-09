from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .config import RagSettings
from .errors import RagBackendError
from .indexing import IndexableChunk

_PUBLIC_SOURCE_TYPES = frozenset(
    {"ACCIDENT_REPORT", "SIF", "KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"}
)


@dataclass(frozen=True)
class RegisteredChunk:
    chunk_index: int
    chunk_id: int
    point_id: str


@dataclass(frozen=True)
class RegisteredSource:
    source_id: int
    chunks: tuple[RegisteredChunk, ...]
    stale_point_ids: tuple[str, ...]
    stale_chunk_ids: tuple[int, ...]


def _mysql_driver():
    try:
        import pymysql
    except ImportError as exc:
        raise RagBackendError("agent_kb MySQL 드라이버가 설치되지 않았습니다.") from exc
    return pymysql


def _connect(settings: RagSettings, pymysql: Any):
    if not settings.kb_db_host or not settings.kb_ingest_password:
        raise RagBackendError("agent_kb 연결 설정이 없습니다.")
    try:
        return pymysql.connect(
            host=settings.kb_db_host,
            port=settings.kb_db_port,
            user="kb_ingest",
            password=settings.kb_ingest_password,
            database=settings.kb_database,
            charset="utf8mb4",
            autocommit=False,
            connect_timeout=max(1, int(settings.timeout_seconds)),
        )
    except pymysql.MySQLError as exc:
        raise RagBackendError("agent_kb 연결에 실패했습니다.") from exc


def validate_chunks(
    settings: RagSettings,
    collection: str,
    chunks: Sequence[IndexableChunk],
) -> None:
    managed = [
        chunk for chunk in chunks
        if chunk.payload.get("source_type") in _PUBLIC_SOURCE_TYPES
    ]
    if not managed:
        return

    chunk_ids = [chunk.payload.get("kb_chunk_id") for chunk in managed]
    if any(isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id <= 0 for chunk_id in chunk_ids):
        raise RagBackendError("agent_kb 청크 ID가 올바르지 않습니다.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise RagBackendError("요청에 중복된 agent_kb 청크 ID가 있습니다.")

    pymysql = _mysql_driver()
    connection = _connect(settings, pymysql)
    try:
        placeholders = ", ".join(["%s"] * len(chunk_ids))
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT c.id, c.source_id, c.collection_name, c.qdrant_point_id, "
                "c.text, c.locator, s.source_type, s.site_id, s.flagged "
                "FROM kb_chunks c JOIN kb_sources s ON s.id = c.source_id "
                f"WHERE c.id IN ({placeholders})",
                chunk_ids,
            )
            rows = {int(row[0]): row for row in cursor.fetchall()}

        if set(rows) != set(chunk_ids):
            raise RagBackendError("요청 청크가 agent_kb 장부에 모두 등록되어 있지 않습니다.")

        for chunk in managed:
            payload = chunk.payload
            row = rows[payload["kb_chunk_id"]]
            source_id = payload["kb_source_id"]
            row_site_id = 0 if row[7] is None else int(row[7])
            payload_text = payload.get("text")
            if (
                int(row[1]) != source_id
                or row[2] != collection
                or row[3] != str(chunk.point_id)
                or not isinstance(payload_text, str)
                or not payload_text.strip()
                or row[4] != payload_text
                or ("locator" in payload and row[5] != payload["locator"])
                or row[6] != payload["source_type"]
                or row_site_id != payload["site_id"]
                or bool(row[8]) != payload["flagged"]
            ):
                raise RagBackendError("요청 청크의 ID 또는 메타데이터가 agent_kb 장부와 다릅니다.")
    except RagBackendError:
        raise
    except pymysql.MySQLError as exc:
        raise RagBackendError("agent_kb 장부 검증에 실패했습니다.") from exc
    finally:
        connection.close()


def register_source_chunks(
    settings: RagSettings,
    collection: str,
    source: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
) -> RegisteredSource:
    source_type = source["source_type"]
    external_id = source["external_id"]
    if source_type not in _PUBLIC_SOURCE_TYPES:
        raise RagBackendError("agent_kb 원본 source_type이 올바르지 않습니다.")
    if not external_id.strip():
        raise RagBackendError("agent_kb 원본 external_id가 필요합니다.")
    if not chunks:
        raise RagBackendError("agent_kb 원본에는 청크가 하나 이상 필요합니다.")

    indexes = [chunk["chunk_index"] for chunk in chunks]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indexes):
        raise RagBackendError("chunk_index는 음이 아닌 정수여야 합니다.")
    if len(set(indexes)) != len(indexes):
        raise RagBackendError("요청에 중복된 chunk_index가 있습니다.")

    point_ids = {
        index: str(uuid5(NAMESPACE_URL, f"{source_type}:{external_id}:{index}"))
        for index in indexes
    }
    pymysql = _mysql_driver()
    connection = _connect(settings, pymysql)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO kb_sources "
                "(source_type, external_id, site_id, title, url, license, publisher, "
                "published_on, file_path, content_sha256, status, chunk_count, "
                "embedding_model, flagged, error_message) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PARSED', 0, NULL, %s, NULL) "
                "ON DUPLICATE KEY UPDATE site_id = VALUES(site_id), title = VALUES(title), "
                "url = VALUES(url), license = VALUES(license), publisher = VALUES(publisher), "
                "published_on = VALUES(published_on), file_path = VALUES(file_path), "
                "content_sha256 = VALUES(content_sha256), status = 'PARSED', chunk_count = 0, "
                "embedding_model = NULL, flagged = VALUES(flagged), error_message = NULL",
                (
                    source_type,
                    external_id,
                    source.get("site_id"),
                    source["title"],
                    source.get("url"),
                    source.get("license"),
                    source.get("publisher"),
                    source.get("published_on"),
                    source.get("file_path"),
                    source.get("content_sha256"),
                    source["flagged"],
                ),
            )
            cursor.execute(
                "SELECT id FROM kb_sources WHERE source_type = %s AND external_id = %s FOR UPDATE",
                (source_type, external_id),
            )
            source_row = cursor.fetchone()
            if source_row is None:
                raise RagBackendError("agent_kb 원본 등록 결과를 찾을 수 없습니다.")
            source_id = int(source_row[0])

            cursor.execute(
                "SELECT id, chunk_index, qdrant_point_id FROM kb_chunks WHERE source_id = %s FOR UPDATE",
                (source_id,),
            )
            old_rows = cursor.fetchall()
            old_by_index = {int(row[1]): row for row in old_rows}

            for chunk in chunks:
                chunk_index = chunk["chunk_index"]
                cursor.execute(
                    "INSERT INTO kb_chunks "
                    "(source_id, chunk_index, collection_name, qdrant_point_id, locator, text, token_count) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), "
                    "collection_name = VALUES(collection_name), "
                    "qdrant_point_id = VALUES(qdrant_point_id), locator = VALUES(locator), "
                    "text = VALUES(text), token_count = VALUES(token_count)",
                    (
                        source_id,
                        chunk_index,
                        collection,
                        point_ids[chunk_index],
                        chunk.get("locator"),
                        chunk["text"],
                        chunk.get("token_count"),
                    ),
                )

            placeholders = ", ".join(["%s"] * len(indexes))
            cursor.execute(
                "SELECT id, chunk_index FROM kb_chunks "
                f"WHERE source_id = %s AND chunk_index IN ({placeholders})",
                [source_id, *indexes],
            )
            new_ids = {int(row[1]): int(row[0]) for row in cursor.fetchall()}
            if set(new_ids) != set(indexes):
                raise RagBackendError("agent_kb 청크 등록 결과가 요청과 다릅니다.")

        connection.commit()
    except RagBackendError:
        connection.rollback()
        raise
    except pymysql.MySQLError as exc:
        connection.rollback()
        raise RagBackendError("agent_kb 원본·청크 등록에 실패했습니다.") from exc
    finally:
        connection.close()

    stale = [row for row in old_rows if int(row[1]) not in new_ids]
    return RegisteredSource(
        source_id=source_id,
        chunks=tuple(
            RegisteredChunk(chunk_index=index, chunk_id=new_ids[index], point_id=point_ids[index])
            for index in indexes
        ),
        stale_point_ids=tuple(str(row[2]) for row in stale),
        stale_chunk_ids=tuple(int(row[0]) for row in stale),
    )


def delete_stale_chunks(
    settings: RagSettings,
    source_id: int,
    chunk_ids: Sequence[int],
) -> None:
    if not chunk_ids:
        return
    pymysql = _mysql_driver()
    connection = _connect(settings, pymysql)
    try:
        with connection.cursor() as cursor:
            for offset in range(0, len(chunk_ids), 500):
                batch = chunk_ids[offset : offset + 500]
                placeholders = ", ".join(["%s"] * len(batch))
                cursor.execute(
                    f"DELETE FROM kb_chunks WHERE source_id = %s AND id IN ({placeholders})",
                    [source_id, *batch],
                )
        connection.commit()
    except pymysql.MySQLError as exc:
        connection.rollback()
        raise RagBackendError("agent_kb에서 이전 청크를 정리하지 못했습니다.") from exc
    finally:
        connection.close()


def mark_sources_embedded(
    settings: RagSettings,
    expected_chunk_counts: dict[int, int],
    embedding_model: str,
) -> None:
    if not expected_chunk_counts:
        return
    if len(embedding_model) > 60:
        raise RagBackendError("EMBEDDING_MODEL이 agent_kb 저장 길이를 초과합니다.")

    pymysql = _mysql_driver()
    connection = _connect(settings, pymysql)
    try:
        source_ids = list(expected_chunk_counts)
        placeholders = ", ".join(["%s"] * len(source_ids))
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT s.id, COUNT(c.id) "
                "FROM kb_sources s LEFT JOIN kb_chunks c ON c.source_id = s.id "
                f"WHERE s.id IN ({placeholders}) GROUP BY s.id",
                list(source_ids),
            )
            counts = {int(source_id): int(count) for source_id, count in cursor.fetchall()}
            if set(counts) != set(source_ids) or any(
                counts[source_id] != expected_chunk_counts[source_id]
                for source_id in source_ids
            ):
                raise RagBackendError("완료 처리할 청크 수가 agent_kb 장부와 다릅니다.")

            for source_id, count in counts.items():
                cursor.execute(
                    "UPDATE kb_sources SET status = 'EMBEDDED', chunk_count = %s, "
                    "embedding_model = %s, error_message = NULL WHERE id = %s",
                    (count, embedding_model, source_id),
                )
        connection.commit()
    except RagBackendError:
        connection.rollback()
        raise
    except pymysql.MySQLError as exc:
        connection.rollback()
        raise RagBackendError("agent_kb 완료 상태 저장에 실패했습니다.") from exc
    finally:
        connection.close()


def mark_sources_failed(
    settings: RagSettings,
    source_ids: Sequence[int],
    error_message: str,
) -> None:
    if not source_ids or not settings.kb_db_host:
        return
    message = error_message.strip() or "임베딩 또는 Qdrant 적재에 실패했습니다."
    pymysql = _mysql_driver()
    connection = _connect(settings, pymysql)
    try:
        placeholders = ", ".join(["%s"] * len(source_ids))
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE kb_sources SET status = 'FAILED', error_message = %s "
                f"WHERE id IN ({placeholders})",
                [message, *source_ids],
            )
        connection.commit()
    except pymysql.MySQLError as exc:
        connection.rollback()
        raise RagBackendError("agent_kb 실패 상태 저장에 실패했습니다.") from exc
    finally:
        connection.close()
