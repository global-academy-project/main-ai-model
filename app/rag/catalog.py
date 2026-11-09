from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import RagSettings
from .errors import RagBackendError
from .indexing import IndexableChunk

_PUBLIC_SOURCE_TYPES = frozenset(
    {"ACCIDENT_REPORT", "SIF", "KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"}
)


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
