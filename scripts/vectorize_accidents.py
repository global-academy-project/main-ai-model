"""사고 데이터(SIF) 청크 파일을 벡터 파일로만 만든다. Qdrant에는 올리지 않고 장부(agent_kb)도 쓰지 않는다.

kb-ingest의 POST /ingest/vectors/local/file (data_type=accidents)를 호출한다. 서버가 청크 파일(05_chunks.jsonl)을
읽어 embed_text를 8개씩 임베딩하고 LOCAL_VECTOR_DIR/<dataset_id>.jsonl 에 {id, vector{dense,sparse}, payload}를 쓴다.
올리기는 따로 있는 적재 스크립트(scripts/upsert_qdrant_jsonl.py 등)가 한다.

  1. 입력 검사: 05_chunks.jsonl (서버의 SIF 규칙을 호출 전에 같은 기준으로 검사), 같은 폴더의 06_validation.json
  2. 사전 점검: kb-ingest /ready, /ingest/vectors/local/file 경로, 요청 스키마에 data_type이 있는지(옛 서버 감지)
  3. 호출: {dataset_id, data_type=accidents, subdir, chunk_file[, validation_file]}. 응답의 저장 건수·파일·컬렉션·검증 방식 확인
  4. 벡터 파일 검사(호스트에서 보이면): 기대 point id와 정확히 일치, dense 차원, sparse, payload가 05와 같은지
  5. 파일 크기와 SHA256을 출력 (적재 스크립트에 넘기기 전 확인용)

서버가 청크 파일을 읽는 위치
  <LOCAL_CHUNK_ROOT>/accidents/<subdir>/<chunk_file>   (LOCAL_CHUNK_ROOT 기본 "out", subdir 기본 "law")
  06_validation.json을 같은 폴더에서 읽으려면 validation_file을 보낸다 (이 스크립트는 06이 있으면 보낸다).
  06을 보내면 서버가 ok·dataset=sif·청크 수를 확인한다 ("pipeline"). 안 보내면 구조 검사만 한다 ("structural").

id 규칙 (서버와 같은 공식)
  point_id = uuid5(NAMESPACE_URL, "SIF:{external_id}:{chunk_index}")

환경 변수 (값은 문서·저장소에 넣지 않는다)
  KB_INGEST_API_TOKEN   kb-ingest 인증 (--dry-run·--check-only에는 불필요)
  KB_INGEST_URL         기본 http://127.0.0.1:8001
  EMBEDDING_DIM         기본 1024

종료 코드: 0 성공 / 1 서버 거부·벡터 파일 검사 실패 / 2 입력·설정 오류

scripts/ingest_law_vectors.py 의 공통 도우미(HTTP 재시도, 오류 종류, 대상 검사)를 가져다 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.ingest_law_vectors import (
    IngestError,
    InputError,
    _finite_numbers,
    _is_int,
    _post_json,
    _short,
    check_dataset_id,
    env,
    is_local_url,
    load_records,
    target_of,
)

SOURCE_TYPE = "SIF"
COLLECTION = "kb_accidents"
DATASET = "sif"
DATA_TYPE = "accidents"
VALIDATION_NAME = "06_validation.json"
SAVE_ROUTE = "/ingest/vectors/local/file"
MAX_RECORDS = 5000
LIST_FIELDS = ("accident_type", "hazard", "work_type", "cas", "ghs")
# 서버가 벡터 파일 payload에 담는 필드 (서버 payload_fields). 이 중 05에 있는 값은 그대로여야 한다.
COMPARE_FIELDS = (
    "source_type", "site_id", "flagged", "title", "locator", "url", "text", "severity", "year",
    "external_id", "chunk_index", "license", "publisher", "content_sha256", "tier", "cate",
    "disaster_type", "situation", *LIST_FIELDS,
)
SUBDIR_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
CHUNK_FILE_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jsonl$"


# ---------------------------------------------------------------- id 규칙


def point_id_for(external_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_TYPE}:{external_id}:{chunk_index}"))


@dataclass(frozen=True)
class ExpectedPoint:
    record: dict[str, Any]
    point_id: str
    label: str


# ---------------------------------------------------------------- 청크 파일 검사


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_records(records: Sequence[dict[str, Any]]) -> list[ExpectedPoint]:
    """서버의 SIF 규칙(LocalChunkRecord + 파일 단위 검사)을 같은 기준으로 먼저 본다. 기대 점 목록을 돌려준다."""
    if not records:
        raise InputError("청크가 없습니다.")
    if len(records) > MAX_RECORDS:
        raise InputError(f"청크가 {len(records)}건입니다. 서버는 한 파일에 최대 {MAX_RECORDS}건까지 처리합니다.")
    points: list[ExpectedPoint] = []
    seen_natural: set[tuple[Any, ...]] = set()
    seen_index: set[tuple[str, int]] = set()
    for number, record in enumerate(records, start=1):
        where = f"({number}번째 청크)"
        key = record.get("natural_key")
        if not isinstance(key, dict):
            raise InputError(f"natural_key가 없습니다 {where}")
        part = key.get("part")
        external_id = record.get("external_id")
        chunk_index = record.get("chunk_index")
        if (
            key.get("dataset") != DATASET
            or not _nonblank(key.get("record"))
            or not _nonblank(key.get("section"))
            or not _is_int(part)
            or part < 0
            or not _nonblank(external_id)
            or external_id != key.get("record")
            or not _is_int(chunk_index)
            or chunk_index < 0
            or record.get("source_type") != SOURCE_TYPE
        ):
            raise InputError(
                f"SIF 청크의 natural_key(dataset=sif·record·section·part), external_id(=record), chunk_index, "
                f"source_type=SIF가 올바르지 않습니다 {where}"
            )
        if record.get("collection", COLLECTION) != COLLECTION:
            raise InputError(f"collection이 {COLLECTION}이 아닙니다 {where}")
        if len(external_id) > 100:
            raise InputError(f"external_id가 100자를 넘습니다 {where}")
        natural = tuple(sorted(key.items()))
        if natural in seen_natural:
            raise InputError(f"natural_key가 중복됩니다 {where} {external_id}")
        seen_natural.add(natural)
        if (external_id, chunk_index) in seen_index:
            raise InputError(f"external_id/chunk_index가 중복됩니다 {where} {external_id}/{chunk_index}")
        seen_index.add((external_id, chunk_index))
        for field in ("text", "embed_text"):
            if not _nonblank(record.get(field)):
                raise InputError(f"{field}가 비어 있습니다 {where} {external_id}")
        site_id = record.get("site_id")
        if not _is_int(site_id) or site_id < 0:
            raise InputError(f"site_id가 올바르지 않습니다 {where} {external_id}")
        if not isinstance(record.get("flagged"), bool):
            raise InputError(f"flagged가 true/false가 아닙니다 {where} {external_id}")
        for field in ("title", "locator"):
            if not _nonblank(record.get(field)):
                raise InputError(f"{field} 값이 비어 있습니다 {where} {external_id}")
        if record.get("year") is not None and not _is_int(record["year"]):
            raise InputError(f"year가 정수가 아닙니다 {where} {external_id}")
        if record.get("severity") is not None and not isinstance(record["severity"], str):
            raise InputError(f"severity가 문자열이 아닙니다 {where} {external_id}")
        for field in LIST_FIELDS:
            values = record.get(field)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise InputError(f"{field}는 문자열 배열이어야 합니다 (태그가 없으면 []) {where} {external_id}")
        points.append(ExpectedPoint(record, point_id_for(external_id, chunk_index), f"{external_id}/{chunk_index}"))
    return points


def check_validation(path: Path, count: int) -> None:
    """서버가 06을 받으면 하는 검사와 같다 (ok, dataset=sif, summary.chunks). 파일이 있을 때만 부른다."""
    try:
        validation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{path}를 읽을 수 없거나 JSON이 아닙니다: {exc}") from exc
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        raise InputError(f"{VALIDATION_NAME}의 ok가 true가 아닙니다. 검증을 통과한 청크만 벡터로 만듭니다.")
    if validation.get("dataset") != DATASET:
        raise InputError(f"{VALIDATION_NAME}의 dataset이 {DATASET}가 아닙니다: {validation.get('dataset')!r}")
    summary = validation.get("summary")
    expected = summary.get("chunks") if isinstance(summary, dict) else None
    if not _is_int(expected) or expected != count:
        raise InputError(f"청크 수가 검증 결과와 다릅니다: 파일 {count}, 검증 {expected}")


def default_dataset_id(clock: Callable[[], datetime] = datetime.now) -> str:
    return clock().strftime("sif-%Y-%m")


# ---------------------------------------------------------------- 사전 점검


def preflight(session: Any, ingest_url: str, *, log: Callable[[str], None] = print) -> None:
    base = ingest_url.rstrip("/")
    try:
        ready = session.get(f"{base}/ready", timeout=10)
    except OSError as exc:
        raise InputError(f"kb-ingest에 접속할 수 없습니다 ({base}): {type(exc).__name__}") from exc
    if ready.status_code != 200:
        raise InputError(f"kb-ingest /ready가 {ready.status_code}를 돌려줬습니다 ({base})")
    try:
        spec = session.get(f"{base}/openapi.json", timeout=10)
    except OSError:
        spec = None
    data = None
    if spec is not None and spec.status_code == 200:
        try:
            data = spec.json()
        except ValueError:
            data = None
    paths = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(paths, dict):
        log(f"[주의] /openapi.json을 읽지 못해 {SAVE_ROUTE} 와 data_type 지원을 확인하지 못했습니다.")
        return
    if SAVE_ROUTE not in paths:
        raise InputError(
            f"이 kb-ingest에는 {SAVE_ROUTE} 가 없습니다. A의 최신 코드로 이미지를 다시 빌드하세요 "
            "(docker compose up -d --build kb-ingest)."
        )
    schemas = ((data.get("components") or {}).get("schemas") or {}) if isinstance(data, dict) else {}
    request_schema = schemas.get("LocalChunkFileRequest")
    properties = request_schema.get("properties") if isinstance(request_schema, dict) else None
    if isinstance(properties, dict) and "data_type" not in properties:
        raise InputError(
            "이 kb-ingest는 요청에 data_type이 없는 옛 판입니다 (사고 파일을 벡터화하지 못하고 법령 폴더를 읽습니다). "
            "A의 최신 ingest_main.py로 이미지를 다시 빌드하세요."
        )
    if not isinstance(properties, dict):
        log("[주의] 요청 스키마에서 data_type 지원을 확인하지 못했습니다.")


# ---------------------------------------------------------------- 호출


def save_vectors(
    session: Any,
    ingest_url: str,
    token: str,
    *,
    dataset_id: str,
    subdir: str,
    chunk_file: str,
    validation_file: str | None,
    count: int,
    retries: int = 2,
    timeout: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    request: dict[str, Any] = {"dataset_id": dataset_id, "data_type": DATA_TYPE, "subdir": subdir, "chunk_file": chunk_file}
    if validation_file is not None:
        request["validation_file"] = validation_file
    body = _post_json(
        session,
        f"{ingest_url.rstrip('/')}{SAVE_ROUTE}",
        token,
        request,
        what="벡터 파일 만들기",
        retries=retries,
        timeout=timeout,
        sleep=sleep,
        not_found_hint=(
            f"서버의 LOCAL_CHUNK_ROOT/{DATA_TYPE}/{subdir}/ 에 {chunk_file}가 없습니다. 이 폴더를 컨테이너에 마운트했는지, "
            "05_chunks.jsonl을 그 폴더에 복사했는지 확인하세요 (서버는 이 스크립트가 읽는 파일의 사본을 자기 폴더에서 읽습니다)."
        ),
    )
    if not _is_int(body.get("saved")) or body["saved"] != count:
        raise IngestError(f"벡터 파일 청크 수가 다릅니다: 파일 {count}, 서버 {body.get('saved')} (서버가 다른 청크 파일을 읽었을 수 있음)")
    if body.get("dataset_id") != dataset_id or body.get("file") != f"{dataset_id}.jsonl":
        raise IngestError(f"서버가 다른 파일에 저장했습니다: {body.get('dataset_id')} / {body.get('file')}")
    if body.get("collection") != COLLECTION:
        raise IngestError(f"서버가 {body.get('collection')} 컬렉션용으로 만들었습니다 (기대 {COLLECTION}).")
    want = "pipeline" if validation_file is not None else "structural"
    if body.get("validation") is not None and body.get("validation") != want:
        raise IngestError(f"서버의 검증 방식이 {body.get('validation')}입니다 (기대 {want}). 서버가 {VALIDATION_NAME}을 읽었는지 확인하세요.")
    return body


# ---------------------------------------------------------------- 벡터 파일 (호스트)


def _differs(expected: Any, actual: Any) -> bool:
    return expected != actual or type(expected) is not type(actual)


def check_vector_file(path: Path, points: Sequence[ExpectedPoint], dim: int) -> tuple[int, list[str]]:
    """서버가 만든 벡터 파일을 검사한다. (점 수, 문제 목록). 기대 point id와 정확히 일치해야 한다."""
    if not path.is_file():
        raise IngestError(
            f"서버가 만든 벡터 파일이 호스트에 없습니다: {path}. kb-ingest의 LOCAL_VECTOR_DIR이 이 폴더에 마운트됐는지 "
            "확인하거나 --vector-dir 을 맞추세요 (파일 검사를 건너뛰려면 --skip-file-check)."
        )
    expected = {p.point_id: p for p in points}
    problems: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{number}번째 줄"
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{where}: JSON이 아닙니다")
                continue
            if not isinstance(item, dict) or not {"id", "vector", "payload"} <= item.keys():
                problems.append(f"{where}: id·vector·payload가 필요합니다")
                continue
            pid = item["id"]
            point = expected.get(pid) if isinstance(pid, str) else None
            if point is None:
                problems.append(f"{where}: 기대하지 않은 id {str(pid)[:8]}")
                continue
            if pid in seen:
                problems.append(f"{where}: 중복 id {pid[:8]}")
                continue
            seen.add(pid)
            label = f"{point.label} ({pid[:8]})"
            vector, payload = item["vector"], item["payload"]
            if not isinstance(vector, dict) or not _finite_numbers(vector.get("dense")) or len(vector["dense"]) != dim:
                problems.append(f"dense 벡터가 없거나 차원이 {dim}가 아님: {label}")
            sparse = vector.get("sparse") if isinstance(vector, dict) else None
            if not (
                isinstance(sparse, dict)
                and isinstance(sparse.get("indices"), list)
                and sparse["indices"]
                and _finite_numbers(sparse.get("values"))
                and len(sparse["indices"]) == len(sparse["values"])
            ):
                problems.append(f"sparse 벡터가 없거나 비어 있음 (임베딩 이미지가 옛 버전?): {label}")
            if not isinstance(payload, dict):
                problems.append(f"payload가 객체가 아님: {label}")
                continue
            diffs = [
                field for field in COMPARE_FIELDS
                if field in point.record and _differs(point.record[field], payload.get(field))
            ]
            if diffs:
                problems.append(f"payload가 05와 다름 ({', '.join(diffs)}; 서버가 다른 청크 파일을 읽었을 수 있음): {label}")
            for field in ("kb_source_id", "kb_chunk_id"):
                if payload.get(field) is not None:
                    problems.append(f"payload.{field}가 null이 아님: {label}")
                    break
    missing = [expected[pid].label for pid in expected if pid not in seen]
    if missing:
        problems.append(_short("벡터 파일에 없는 청크", missing, 8))
    return len(seen), problems


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


# ---------------------------------------------------------------- 실행


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIF 청크 파일을 벡터 파일로만 만든다 (Qdrant·장부는 쓰지 않음).")
    parser.add_argument("--chunks", type=Path, default=Path("out/accidents/law/05_chunks.jsonl"),
                        help="청크 파일 (같은 폴더에 06_validation.json이 있으면 함께 보낸다)")
    parser.add_argument("--subdir", default="law",
                        help="서버가 읽는 하위 폴더: <LOCAL_CHUNK_ROOT>/accidents/<subdir>/ (기본 law)")
    parser.add_argument("--dataset-id", help="벡터 파일 이름 (기본: sif-<연-월>). 같은 이름이면 파일을 덮어쓴다")
    parser.add_argument("--vector-dir", type=Path, default=Path("out/vectors"),
                        help="서버의 LOCAL_VECTOR_DIR이 호스트에서 보이는 폴더 (벡터 파일 검사용)")
    parser.add_argument("--limit", type=int, help="앞의 N건만 벡터로 만든다 (시험용). 청크 파일 옆에 부분 파일을 쓰고 06은 보내지 않는다")
    parser.add_argument("--no-validation", action="store_true", help="06_validation.json이 있어도 보내지 않는다 (구조 검사만)")
    parser.add_argument("--dry-run", action="store_true", help="계획만 출력 (접속·파일 쓰기 없음)")
    parser.add_argument("--check-only", action="store_true", help="API를 부르지 않고 이미 있는 벡터 파일만 검사")
    parser.add_argument("--skip-file-check", action="store_true", help="호스트의 벡터 파일 검사를 건너뜀 (마운트가 없을 때)")
    parser.add_argument("--allow-remote", action="store_true", help="로컬이 아닌 kb-ingest 사용을 허용")
    parser.add_argument("--ingest-url", default=os.getenv("KB_INGEST_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    parser.add_argument("--save-timeout", type=float, default=3600.0, help="벡터 파일 만들기 요청 제한 시간(초)")
    args = parser.parse_args(argv)
    if args.embedding_dim <= 0 or args.save_timeout <= 0:
        parser.error("--embedding-dim과 --save-timeout은 양수여야 합니다")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit은 양수여야 합니다")
    if args.check_only and args.skip_file_check:
        parser.error("--check-only와 --skip-file-check는 함께 쓸 수 없습니다")
    return args


def limited_file(chunks: Path, records: Sequence[dict[str, Any]], limit: int) -> tuple[Path, int]:
    """앞의 N건을 청크 파일 옆의 부분 파일로 쓴다 (서버는 같은 폴더의 파일만 읽는다)."""
    count = min(limit, len(records))
    target = chunks.with_name(f"{chunks.stem}.limit{count}.jsonl")
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records[:count]), encoding="utf-8", newline="\n"
    )
    return target, count


def run(
    args: argparse.Namespace,
    *,
    session: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = datetime.now,
    log: Callable[[str], None] = print,
) -> int:
    if not re.match(SUBDIR_RE, args.subdir):
        raise InputError(f"--subdir이 서버 규칙(영문·숫자·._-, 100자 이하)에 맞지 않습니다: {args.subdir!r}")
    if not re.match(CHUNK_FILE_RE, args.chunks.name):
        raise InputError(f"청크 파일 이름이 서버가 받지 못하는 형식입니다 (영문·숫자·._-, .jsonl): {args.chunks.name}")
    if not is_local_url(args.ingest_url) and not args.allow_remote:
        raise InputError(f"kb-ingest 대상이 로컬이 아닙니다: {target_of(args.ingest_url)}. 공용 서버를 쓰려면 --allow-remote를 주세요.")

    records = load_records(args.chunks)
    check_points = check_records(records)  # 전체 파일을 먼저 검사한다
    full_count = len(records)
    validation_path = args.chunks.with_name(VALIDATION_NAME)
    use_validation = (not args.no_validation) and args.limit is None and validation_path.is_file()
    if use_validation:
        check_validation(validation_path, full_count)

    chunk_name = args.chunks.name
    points = check_points
    if args.limit is not None:
        count = min(args.limit, full_count)
        chunk_name = f"{args.chunks.stem}.limit{count}.jsonl"
        points = check_points[:count]
    dataset_id = check_dataset_id(args.dataset_id or (f"sif-limit{len(points)}" if args.limit is not None else default_dataset_id(clock)))
    vector_path = args.vector_dir / f"{dataset_id}.jsonl"

    log(f"SIF 청크 {len(points)}건 (파일 전체 {full_count}건) / 벡터 파일 {dataset_id}.jsonl / 컬렉션 {COLLECTION}")
    log(f"  서버가 읽는 위치: <LOCAL_CHUNK_ROOT>/{DATA_TYPE}/{args.subdir}/{chunk_name}")
    if use_validation:
        log(f"  {VALIDATION_NAME} 확인함 (ok, dataset=sif, 청크 수 {full_count}) → 서버에 validation_file 전달 (pipeline)")
    elif args.limit is not None:
        log(f"  --limit: 부분 파일이라 {VALIDATION_NAME}은 보내지 않음 (structural)")
    elif args.no_validation:
        log(f"  --no-validation: {VALIDATION_NAME}을 보내지 않음 (structural)")
    else:
        log(f"  [주의] {validation_path} 가 없어 구조 검사만 합니다 (structural)")
    log(f"  첫 점 {points[0].label} → {points[0].point_id}")
    log("Qdrant에는 올리지 않고 장부(agent_kb)도 쓰지 않습니다. 올리기는 별도 적재 스크립트로 하세요.")
    if args.dry_run:
        log("\n[dry-run] 접속·파일 쓰기 없이 종료합니다.")
        return 0

    if not args.check_only:
        token = env("KB_INGEST_API_TOKEN")
        if session is None:
            import requests

            session = requests.Session()
        preflight(session, args.ingest_url, log=log)
        if args.limit is not None:
            written, _ = limited_file(args.chunks, records, args.limit)
            log(f"부분 파일을 썼습니다: {written} (서버가 읽을 폴더에 있어야 합니다)")
        started = clock()
        log(f"벡터 파일 만드는 중 (서버가 8개씩 임베딩합니다. {len(points)}건은 몇 분 걸릴 수 있습니다)")
        body = save_vectors(
            session, args.ingest_url, token,
            dataset_id=dataset_id, subdir=args.subdir, chunk_file=chunk_name,
            validation_file=VALIDATION_NAME if use_validation else None,
            count=len(points), timeout=args.save_timeout, sleep=sleep,
        )
        log(f"  서버 응답: saved={body['saved']} file={body['file']} model={body.get('model')} validation={body.get('validation')} ({(clock() - started).total_seconds():.0f}초)")

    if args.skip_file_check:
        log("[참고] 호스트의 벡터 파일 검사를 건너뛰었습니다. 적재 전에 --check-only 로 검사하세요.")
        return 0
    seen, problems = check_vector_file(vector_path, points, args.embedding_dim)
    if problems:
        shown = "\n  - ".join(problems[:20]) + (f"\n  - … 외 {len(problems) - 20}건" if len(problems) > 20 else "")
        raise IngestError(f"벡터 파일 검사에서 문제가 나왔습니다 ({vector_path}):\n  - {shown}\n같은 명령을 다시 실행하면 파일을 덮어씁니다.")
    size, digest = file_digest(vector_path)
    log(f"\n[완료] 벡터 파일 검사 통과: {seen}점, dense {args.embedding_dim}차원, sparse 있음, payload가 05와 같음")
    log(f"  파일: {vector_path} ({size:,} 바이트)")
    log(f"  SHA256: {digest}")
    log("  kb_source_id·kb_chunk_id는 null입니다 (장부 연결 없음). Qdrant 올리기는 별도 적재 스크립트로 하세요.")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        code = run(args)
    except InputError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except IngestError as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 멈췄습니다. 같은 명령을 다시 실행하면 같은 파일을 덮어씁니다.", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
