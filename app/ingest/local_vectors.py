"""청크를 벡터 파일로 만들고, 별도 요청으로 Qdrant에 적재한다."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from app.rag.catalog import (
    mark_sources_embedded,
    mark_sources_failed,
    register_sources_chunks,
    validate_chunks,
)
from app.rag.config import RagSettings
from app.rag.embedding import embed_texts
from app.rag.errors import RagBackendError
from app.rag.filters import COLLECTIONS
from app.rag.indexing import IndexableChunk
from app.rag.qdrant import make_point, upsert_points
from .auth import authenticate
from .documents import _mark_failed_after_ingest
from .schemas import (
    LocalChunkFileRequest,
    LocalChunkRecord,
    LocalQdrantPoint,
    LocalVectorLoadRequest,
    LocalVectorSaveRequest,
)


logger = logging.getLogger(__name__)
LOCAL_VECTOR_BATCH_SIZE = 8
LOCAL_QDRANT_BATCH_SIZE = 32
router = APIRouter()


@router.post("/ingest/vectors/local")
def save_local_vectors(
    body: LocalVectorSaveRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    """요청 본문의 청크를 임베딩해 로컬 JSONL 파일로 저장한다."""
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    output_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser()
    destination = output_dir / f"{body.dataset_id}.jsonl"
    temp_path: Path | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{body.dataset_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for offset in range(0, len(body.chunks), LOCAL_VECTOR_BATCH_SIZE):
                batch = body.chunks[offset : offset + LOCAL_VECTOR_BATCH_SIZE]
                embeddings = embed_texts(settings, [chunk.text for chunk in batch])
                if len(embeddings) != len(batch):
                    raise RagBackendError("임베딩 서버가 청크 수와 다른 결과를 반환했습니다.")
                for chunk, embedding in zip(batch, embeddings):
                    point = make_point(chunk.point_id, embedding, chunk.payload)
                    output.write(
                        json.dumps(
                            {"collection": body.collection, "model": settings.embedding_model, **point},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        os.replace(temp_path, destination)
    except RagBackendError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local vector embedding failed")
        raise HTTPException(status_code=502, detail="임베딩 요청이 실패했습니다.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Local vector file save failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일 저장에 실패했습니다.") from exc

    return {
        "saved": len(body.chunks),
        "dataset_id": body.dataset_id,
        "file": destination.name,
        "model": settings.embedding_model,
    }


@router.post("/ingest/vectors/local/file")
def save_local_chunk_file_vectors(
    body: LocalChunkFileRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    """마운트된 청크 파일을 검증·장부 연결한 뒤 임베딩 파일을 만든다."""
    if body.data_type is None:
        input_dir = Path(os.getenv("LOCAL_CHUNK_DIR", "out/law")).expanduser().resolve()
    else:
        input_root = Path(os.getenv("LOCAL_CHUNK_ROOT", "out")).expanduser().resolve()
        input_dir = (input_root / body.data_type / body.subdir).resolve()
        try:
            input_dir.relative_to(input_root)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="청크 디렉터리 경로가 올바르지 않습니다.") from exc
    chunk_path = (input_dir / body.chunk_file).resolve()
    validation_file = body.validation_file
    if validation_file is None and body.data_type != "accidents":
        validation_file = "06_validation.json"
    validation_path = (input_dir / validation_file).resolve() if validation_file else None
    if chunk_path.parent != input_dir or (
        validation_path is not None and validation_path.parent != input_dir
    ):
        raise HTTPException(status_code=422, detail="청크 파일 경로가 올바르지 않습니다.")
    if not chunk_path.is_file():
        raise HTTPException(status_code=404, detail="로컬 청크 파일을 찾을 수 없습니다.")
    if validation_path is not None and not validation_path.is_file():
        if body.data_type == "accidents":
            validation_path = None
            validation_file = None
        else:
            raise HTTPException(status_code=422, detail=f"검증 파일을 찾을 수 없습니다: {validation_file}")

    validation: dict[str, Any] | None = None
    try:
        if validation_path is not None:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if validation.get("ok") is not True:
                raise HTTPException(status_code=422, detail="청크 검증을 통과하지 못했습니다.")
        records: list[LocalChunkRecord] = []
        for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(LocalChunkRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"청크 파일 {line_number}번째 줄을 읽을 수 없습니다.",
                ) from exc
            if len(records) > 5000:
                raise HTTPException(status_code=422, detail="청크 파일은 최대 5,000건까지 처리할 수 있습니다.")
    except OSError as exc:
        logger.exception("Local chunk file read failed")
        raise HTTPException(status_code=503, detail="로컬 청크 파일을 읽지 못했습니다.") from exc
    except (json.JSONDecodeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="06_validation.json 형식이 올바르지 않습니다.") from exc

    if not records:
        raise HTTPException(status_code=422, detail="청크 파일이 비어 있습니다.")
    if validation is not None:
        summary = validation.get("summary")
        expected_count = summary.get("chunks") if isinstance(summary, dict) else None
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(records)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"청크 수가 검증 결과와 일치하지 않습니다 (파일 {len(records)}, 검증 {expected_count}).",
            )
    collections = {record.collection for record in records}
    if len(collections) != 1:
        raise HTTPException(status_code=422, detail="청크 파일의 collection이 일치하지 않습니다.")
    collection = next(iter(collections))
    expected_collection = {
        "accidents": "kb_accidents",
        "guides": "kb_guides",
    }.get(body.data_type)
    if expected_collection is not None and collection != expected_collection:
        raise HTTPException(status_code=422, detail="data_type과 청크 collection이 일치하지 않습니다.")
    natural_keys = [tuple(sorted(record.natural_key.items())) for record in records]
    if len(set(natural_keys)) != len(natural_keys):
        raise HTTPException(status_code=422, detail="청크 파일에 중복된 natural_key가 있습니다.")
    indexed_records: list[tuple[LocalChunkRecord, str, dict[str, Any]]] = []
    sources_by_external_id: dict[str, dict[str, Any]] = {}
    chunks_by_external_id: dict[str, list[dict[str, Any]]] = {}
    ledger_key_by_point_id: dict[str, tuple[str, int]] = {}
    if collection == "kb_guides":
        law_keys = {record.natural_key.get("law_key") for record in records}
        source_types = {record.model_extra.get("source_type") if record.model_extra else None for record in records}
        if len(law_keys) != 1 or source_types != {"LAW"}:
            raise HTTPException(status_code=422, detail="법령 청크의 법령 ID 또는 source_type이 일치하지 않습니다.")
        external_id = next(iter(law_keys))

        index_map_path = input_dir / "chunk_index_map.json"
        if not index_map_path.is_file():
            raise HTTPException(status_code=422, detail="chunk_index_map.json이 필요합니다.")
        try:
            index_map_data = json.loads(index_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="chunk_index_map.json을 읽을 수 없습니다.") from exc
        indexes = index_map_data.get("indexes") if isinstance(index_map_data, dict) else None
        if (
            not isinstance(indexes, dict)
            or index_map_data.get("source_type") != "LAW"
            or index_map_data.get("external_id") != external_id
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in indexes.items()
            )
            or len(set(indexes.values())) != len(indexes)
        ):
            raise HTTPException(status_code=422, detail="chunk_index_map.json이 법령 또는 인덱스 형식과 맞지 않습니다.")

        for record in records:
            key = record.natural_key
            natural_key = f"{key['law_key']}|{key['article_key']}|{key['part']}"
            chunk_index = indexes.get(natural_key)
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
                raise HTTPException(status_code=422, detail="청크의 natural_key가 chunk_index_map.json에 없습니다.")
            row = record.model_extra or {}
            site_id = row.get("site_id")
            flagged = row.get("flagged")
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise HTTPException(status_code=422, detail="LAW payload.site_id가 올바르지 않습니다.")
            if not isinstance(flagged, bool):
                raise HTTPException(status_code=422, detail="LAW payload.flagged가 올바르지 않습니다.")
            if not isinstance(row.get("title"), str) or not row["title"].strip():
                raise HTTPException(status_code=422, detail="LAW payload.title이 비어 있습니다.")
            if not isinstance(row.get("locator"), str) or not row["locator"].strip():
                raise HTTPException(status_code=422, detail="LAW payload.locator가 비어 있습니다.")
            law_meta = row.get("law_meta")
            source = {
                "source_type": "LAW",
                "external_id": external_id,
                "site_id": site_id,
                "title": row["title"],
                "url": row.get("url"),
                "license": row.get("license"),
                "publisher": row.get("publisher"),
                "file_path": row.get("file_path"),
                "content_sha256": (
                    law_meta.get("source_sha256") if isinstance(law_meta, dict) else None
                ),
                "flagged": flagged,
            }
            existing_source = sources_by_external_id.get(external_id)
            if existing_source is not None and existing_source != source:
                raise HTTPException(status_code=422, detail="같은 LAW external_id의 원본 메타데이터가 일치하지 않습니다.")
            sources_by_external_id[external_id] = source
            chunks_by_external_id.setdefault(external_id, []).append(
                {
                    "chunk_index": chunk_index,
                    "text": record.text,
                    "locator": row["locator"],
                }
            )
            point_id = str(uuid5(NAMESPACE_URL, f"LAW:{external_id}:{chunk_index}"))
            ledger_key_by_point_id[point_id] = (external_id, chunk_index)
            indexed_records.append((record, point_id, record.model_dump()))
    else:
        rows = [record.model_dump() for record in records]
        source_types = {row.get("source_type") for row in rows}
        datasets = {record.natural_key.get("dataset") for record in records}
        if source_types != {"SIF"} or datasets != {"sif"}:
            raise HTTPException(status_code=422, detail="현재 사고 파일 벡터화는 SIF 데이터셋만 지원합니다.")
        if validation is not None and validation.get("dataset") != "sif":
            raise HTTPException(status_code=422, detail="사고 validation 파일의 dataset이 sif가 아닙니다.")

        seen_chunk_indexes: set[tuple[str, int]] = set()
        payload_fields = (
            "source_type", "site_id", "flagged", "title", "locator", "url", "text",
            "severity", "accident_type", "hazard", "work_type", "cas", "ghs", "year",
            "external_id", "chunk_index", "natural_key", "license", "publisher",
            "content_sha256", "tier", "cate", "disaster_type", "situation",
        )
        for record, row in zip(records, rows):
            external_id = row.get("external_id")
            chunk_index = row.get("chunk_index")
            if (
                not isinstance(external_id, str)
                or not external_id.strip()
                or isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
            ):
                raise HTTPException(status_code=422, detail="SIF 청크 external_id 또는 chunk_index가 올바르지 않습니다.")
            key = (external_id, chunk_index)
            if key in seen_chunk_indexes:
                raise HTTPException(status_code=422, detail="SIF 청크 external_id/chunk_index가 중복됩니다.")
            seen_chunk_indexes.add(key)

            site_id = row.get("site_id")
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise HTTPException(status_code=422, detail="SIF payload.site_id가 올바르지 않습니다.")
            if not isinstance(row.get("flagged"), bool):
                raise HTTPException(status_code=422, detail="SIF payload.flagged가 올바르지 않습니다.")
            if not isinstance(row.get("title"), str) or not row["title"].strip():
                raise HTTPException(status_code=422, detail="SIF payload.title이 비어 있습니다.")
            if not isinstance(row.get("locator"), str) or not row["locator"].strip():
                raise HTTPException(status_code=422, detail="SIF payload.locator가 비어 있습니다.")
            if row.get("year") is not None and (
                isinstance(row["year"], bool) or not isinstance(row["year"], int)
            ):
                raise HTTPException(status_code=422, detail="SIF payload.year가 올바르지 않습니다.")
            if row.get("severity") is not None and not isinstance(row["severity"], str):
                raise HTTPException(status_code=422, detail="SIF payload.severity가 올바르지 않습니다.")
            for field in ("accident_type", "hazard", "work_type", "cas", "ghs"):
                values = row.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise HTTPException(status_code=422, detail=f"SIF payload.{field} 형식이 올바르지 않습니다.")

            source = {
                "source_type": "SIF",
                "external_id": external_id,
                "site_id": site_id,
                "title": row["title"],
                "url": row.get("url"),
                "license": row.get("license"),
                "publisher": row.get("publisher"),
                "file_path": row.get("file_path"),
                "content_sha256": row.get("content_sha256"),
                "flagged": row["flagged"],
            }
            existing_source = sources_by_external_id.get(external_id)
            if existing_source is not None and existing_source != source:
                raise HTTPException(status_code=422, detail="같은 SIF external_id의 원본 메타데이터가 일치하지 않습니다.")
            sources_by_external_id[external_id] = source
            chunks_by_external_id.setdefault(external_id, []).append(
                {
                    "chunk_index": chunk_index,
                    "text": row["text"],
                    "locator": row["locator"],
                }
            )

            point_id = str(uuid5(NAMESPACE_URL, f"SIF:{external_id}:{chunk_index}"))
            ledger_key_by_point_id[point_id] = (external_id, chunk_index)
            payload = {field: row[field] for field in payload_fields if field in row}
            payload["kb_source_id"] = row.get("kb_source_id")
            payload["kb_chunk_id"] = row.get("kb_chunk_id")
            indexed_records.append((record, point_id, payload))

    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    registered_source_ids: list[int] = []
    if chunks_by_external_id:
        registered_ids: dict[tuple[str, int], tuple[int, int, str]] = {}
        try:
            source_entries = [
                (sources_by_external_id[external_id], chunks)
                for external_id, chunks in chunks_by_external_id.items()
            ]
            registrations = register_sources_chunks(settings, collection, source_entries)
            for external_id, registered in zip(chunks_by_external_id, registrations):
                registered_source_ids.append(registered.source_id)
                for registered_chunk in registered.chunks:
                    registered_ids[(external_id, registered_chunk.chunk_index)] = (
                        registered.source_id,
                        registered_chunk.chunk_id,
                        registered_chunk.point_id,
                    )
        except RagBackendError as exc:
            logger.exception("Public source or chunk registration failed")
            _mark_failed_after_ingest(settings, registered_source_ids)
            raise HTTPException(status_code=503, detail="원본·청크 장부 등록에 실패했습니다.") from exc

        indexed_records = [
            (
                record,
                registered_ids[ledger_key_by_point_id[point_id]][2],
                {
                    **payload,
                    "kb_source_id": registered_ids[ledger_key_by_point_id[point_id]][0],
                    "kb_chunk_id": registered_ids[ledger_key_by_point_id[point_id]][1],
                },
            )
            for record, point_id, payload in indexed_records
        ]

    output_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser()
    destination = output_dir / f"{body.dataset_id}.jsonl"
    temp_path: Path | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{body.dataset_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for offset in range(0, len(records), LOCAL_VECTOR_BATCH_SIZE):
                batch = records[offset : offset + LOCAL_VECTOR_BATCH_SIZE]
                embeddings = embed_texts(settings, [record.embed_text for record in batch])
                if len(embeddings) != len(batch):
                    raise RagBackendError("임베딩 서버가 청크 수와 다른 결과를 반환했습니다.")
                for (record, point_id, payload), embedding in zip(indexed_records[offset : offset + len(batch)], embeddings):
                    point = make_point(point_id, embedding, payload)
                    output.write(
                        json.dumps(
                            point,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        os.replace(temp_path, destination)
    except RagBackendError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        _mark_failed_after_ingest(settings, registered_source_ids)
        logger.exception("Local file vectorization failed")
        raise HTTPException(status_code=502, detail="임베딩 요청이 실패했습니다.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        _mark_failed_after_ingest(settings, registered_source_ids)
        logger.exception("Local vector file save failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일 저장에 실패했습니다.") from exc

    return {
        "saved": len(records),
        "dataset_id": body.dataset_id,
        "file": destination.name,
        "collection": next(iter(collections)),
        "model": settings.embedding_model,
        "validation": "pipeline" if validation is not None else "structural",
    }


@router.post("/ingest/vectors/local/load")
def load_local_vectors_to_qdrant(
    body: LocalVectorLoadRequest,
    _: Annotated[None, Depends(authenticate)],
) -> dict[str, str | int]:
    """벡터 파일을 검증해 Qdrant에 올리고 연결된 장부 원본을 완료 처리한다."""
    write_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not write_api_key:
        raise HTTPException(status_code=503, detail="Qdrant 쓰기 설정이 없습니다.")
    try:
        settings = RagSettings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="임베딩 설정이 올바르지 않습니다.") from exc

    vector_dir = Path(os.getenv("LOCAL_VECTOR_DIR", "data/vectors")).expanduser().resolve()
    vector_path = (vector_dir / f"{body.dataset_id}.jsonl").resolve()
    if vector_path.parent != vector_dir:
        raise HTTPException(status_code=422, detail="벡터 파일 경로가 올바르지 않습니다.")
    if not vector_path.is_file():
        raise HTTPException(status_code=404, detail="로컬 벡터 파일을 찾을 수 없습니다.")

    points: list[dict[str, Any]] = []
    point_ids: set[str | int] = set()
    unlinked_payloads = 0
    expected_chunk_counts: dict[int, int] = {}
    linked_chunks: list[IndexableChunk] = []
    try:
        for line_number, line in enumerate(vector_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            if len(points) >= 5000:
                raise HTTPException(status_code=422, detail="벡터 파일은 최대 5,000건까지 적재할 수 있습니다.")
            try:
                point = LocalQdrantPoint.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"벡터 파일 {line_number}번째 줄이 Qdrant 포인트 형식이 아닙니다.",
                ) from exc
            if point.id in point_ids:
                raise HTTPException(status_code=422, detail="벡터 파일에 중복된 point id가 있습니다.")
            point_ids.add(point.id)
            if len(point.vector["dense"]) != settings.embedding_dim:
                raise HTTPException(status_code=422, detail="dense 벡터 차원이 EMBEDDING_DIM과 다릅니다.")

            payload = point.payload
            if payload.get("collection", body.collection) != body.collection:
                raise HTTPException(status_code=422, detail="요청 collection과 payload collection이 다릅니다.")
            if payload.get("source_type") not in COLLECTIONS[body.collection]:
                raise HTTPException(status_code=422, detail="source_type이 collection과 맞지 않습니다.")
            site_id = payload.get("site_id")
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise HTTPException(status_code=422, detail="payload.site_id가 올바르지 않습니다.")
            if not isinstance(payload.get("flagged"), bool):
                raise HTTPException(status_code=422, detail="payload.flagged가 올바르지 않습니다.")
            if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                raise HTTPException(status_code=422, detail="payload.text가 비어 있습니다.")

            linked = True
            for id_field in ("kb_source_id", "kb_chunk_id"):
                value = payload.get(id_field)
                if value is None:
                    linked = False
                elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise HTTPException(status_code=422, detail=f"payload.{id_field}가 올바르지 않습니다.")
            if not linked:
                unlinked_payloads += 1
            else:
                source_id = payload["kb_source_id"]
                expected_chunk_counts[source_id] = expected_chunk_counts.get(source_id, 0) + 1
                linked_chunks.append(
                    IndexableChunk(
                        point_id=point.id,
                        text=payload["text"],
                        payload=payload,
                    )
                )
            points.append(point.model_dump())
    except OSError as exc:
        logger.exception("Local vector file read failed")
        raise HTTPException(status_code=503, detail="로컬 벡터 파일을 읽지 못했습니다.") from exc

    if not points:
        raise HTTPException(status_code=422, detail="벡터 파일이 비어 있습니다.")

    if linked_chunks:
        try:
            validate_chunks(settings, body.collection, linked_chunks)
        except RagBackendError as exc:
            logger.exception("agent_kb vector-file validation failed")
            raise HTTPException(status_code=503, detail="지식 DB 장부 확인에 실패했습니다.") from exc

    try:
        for offset in range(0, len(points), LOCAL_QDRANT_BATCH_SIZE):
            upsert_points(
                settings,
                body.collection,
                points[offset : offset + LOCAL_QDRANT_BATCH_SIZE],
                write_api_key=write_api_key,
            )
    except RagBackendError as exc:
        logger.exception("Local Qdrant upsert failed")
        _mark_failed_after_ingest(settings, list(expected_chunk_counts))
        raise HTTPException(
            status_code=502,
            detail="Qdrant 적재 요청이 실패했습니다. 같은 요청으로 재시도할 수 있습니다.",
        ) from exc

    if expected_chunk_counts:
        try:
            mark_sources_embedded(settings, expected_chunk_counts, settings.embedding_model)
        except RagBackendError as exc:
            logger.exception("agent_kb completion status update failed after local vector load")
            raise HTTPException(
                status_code=502,
                detail="Qdrant 적재 후 지식 DB 완료 상태 저장에 실패했습니다.",
            ) from exc

    return {
        "indexed": len(points),
        "collection": body.collection,
        "dataset_id": body.dataset_id,
        "unlinked_payloads": unlinked_payloads,
        "finalized_sources": len(expected_chunk_counts),
    }
