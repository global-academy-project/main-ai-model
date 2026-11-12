"""법령 청크(05_chunks.jsonl)를 A의 '벡터 파일 → Qdrant' 경로로 적재하고 검증한다.

실행 위치: 프로젝트 루트 (python -m scripts.ingest_law_vectors ...)

    python -m scripts.ingest_law_vectors --dry-run                      # 계획만 (접속·파일 쓰기 없음)
    python -m scripts.ingest_law_vectors --env-file <infra>\\docker\\.env # 전체: 벡터 파일 만들기 → Qdrant 올리기 → 검증
    python -m scripts.ingest_law_vectors --verify-only                  # 올리기 없이 검증만
    python -m scripts.ingest_law_vectors --no-load                      # 벡터 파일만 만들고 멈춤
    python -m scripts.ingest_law_vectors --skip-save --env-file ...     # 이미 만든 벡터 파일을 다시 올림 (재임베딩 없음)
    python -m scripts.ingest_law_vectors --load-via api                 # Qdrant 올리기를 서버(/ingest/vectors/local/load)가 함

장부 id와 chunk_index 대응표 (예전 scripts/ingest_law.py 의 역할을 이어받았다)
  * 장부(agent_kb) 등록은 서버(API)가 하고, 벡터 파일 payload의 kb_source_id·kb_chunk_id가 채워져 있어야 검색 결과
    doc_id("LAW:{kb_source_id}")가 생긴다. 그래서 장부 id가 없는 점이 하나라도 있으면 올리지 않는다 (--no-require-ledger-ids로 끔).
  * 새 조문이 생기면(또는 대응표가 처음이면) --update-index-map 으로 새 번호를 배정한다. 기존 번호는 바꾸거나 재사용하지 않고
    새 키만 이어서 받는다. 파일을 덮어쓰기 전에 chunk_index_map.json.bak 으로 한 벌 남긴다. 서버가 같은 파일을 읽으므로
    서버를 부르기 전에 쓴다 (--dry-run과 --verify-only는 쓰지 않는다).
  * 임베딩은 서버가 한다: POST /ingest/vectors/local/file 이 서버의 <LOCAL_CHUNK_ROOT>/<data_type>/<subdir>/ (법령은 guides/law)에서
    05_chunks.jsonl, 06_validation.json, chunk_index_map.json 을 읽어 LOCAL_VECTOR_DIR/<dataset_id>.jsonl 을 만든다.
    이 스크립트는 그 세 파일이 --chunks 와 같은 폴더(서버가 읽는 폴더의 사본)에 있다고 본다.
    요청에 data_type·subdir를 싣는다. 서버 폴더에 파일이 없으면 404다. data_type이 없던 옛 서버는 --data-type none.

흐름
  1. 입력 검사 (06_validation.json ok·청크 수, 법령 하나, 필드, chunk_index_map.json이 모든 청크를 덮는지)
  2. 사전 점검: kb-ingest /ready, 필요한 경로, Qdrant kb_guides (dense 차원, sparse 설정)
  3. 벡터 파일 만들기: POST {kb-ingest}/ingest/vectors/local/file  (같은 dataset_id는 파일을 덮어쓴다)
  4. 벡터 파일 검사 (--load-via file): 기대 point id와 정확히 일치, dense 차원, sparse, 본문이 05와 같은지
  5. Qdrant 올리기: file = 호스트의 벡터 파일을 32개씩 PUT (쓰기 키 필요) / api = 서버가 올림
  6. 검증: Qdrant 점 수, 기대 point id 전수 조회(payload 대조), 표본 벡터(dense·sparse), 장부 연결 여부 보고

id 규칙 (서버·/init_qdrant.py가 같은 공식)
  point_id = uuid5(NAMESPACE_URL, "LAW:{external_id}:{chunk_index}"),  chunk_index = chunk_index_map.json[law_key|article_key|part]

환경 변수 (값은 문서·저장소에 넣지 않는다)
  KB_INGEST_API_TOKEN         kb-ingest 인증 (벡터 파일 만들기, --load-via api)
  KB_INGEST_URL(기본 http://127.0.0.1:8001)
  QDRANT_URL(기본 http://127.0.0.1:60011)
  QDRANT_READ_ONLY_API_KEY    검증용 (없으면 QDRANT_API_KEY, 그것도 없으면 --env-file의 쓰기 키)
  QDRANT_API_KEY 또는 --env-file    올리기용 쓰기 키 (--load-via file). --env-file을 주면 셸의 오래된 값 대신 이 파일의 값을 쓴다
  EMBEDDING_DIM(기본 1024)

종료 코드: 0 성공 / 1 적재·검증 실패 / 2 입력·설정 오류
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SOURCE_TYPE = "LAW"
COLLECTION = "kb_guides"
VALIDATION_NAME = "06_validation.json"  # 서버가 청크 폴더에서 이 이름으로 읽는다
INDEX_MAP_NAME = "chunk_index_map.json"  # 〃
SAVE_ROUTE = "/ingest/vectors/local/file"
DATA_TYPE = "guides"  # 요청의 data_type: 서버는 <LOCAL_CHUNK_ROOT>/<data_type>/<subdir>/ 에서 읽는다
SUBDIR = "law"  # 요청의 subdir
LOAD_ROUTE = "/ingest/vectors/local/load"
DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
CHUNK_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jsonl$")
MAX_RECORDS = 5000  # 서버 한도 (청크 파일·벡터 파일)
UPSERT_BATCH = 32  # Qdrant에 한 번에 보내는 점 수 (A의 scripts/upsert_qdrant_jsonl.py와 같다)
RETRIEVE_BATCH = 200
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_LOCATOR = 100


class InputError(Exception):
    """입력 파일·설정 문제 (종료 코드 2)."""


class IngestError(Exception):
    """적재·검증 실패 (종료 코드 1)."""


# ---------------------------------------------------------------- id 규칙


def point_id_for(external_id: str, chunk_index: int) -> str:
    """같은 청크는 항상 같은 id. 서버(/ingest/vectors/local/file)와 같은 공식이다."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_TYPE}:{external_id}:{chunk_index}"))


def natural_key_str(record: dict[str, Any]) -> str:
    key = record["natural_key"]
    return f"{key['law_key']}|{key['article_key']}|{key['part']}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------- 청크 파일 검사


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputError(f"청크 파일이 없습니다: {path}")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path} {number}번째 줄이 JSON이 아닙니다: {exc}") from exc
        if not isinstance(record, dict):
            raise InputError(f"{path} {number}번째 줄이 JSON 객체가 아닙니다.")
        records.append(record)
    return records


def check_records(records: Sequence[dict[str, Any]], validation_path: Path) -> str:
    """적재해도 되는 청크인지 검사하고 external_id(법령 식별자)를 돌려준다."""
    if not records:
        raise InputError("청크가 없습니다.")
    if len(records) > MAX_RECORDS:
        raise InputError(f"청크가 {len(records)}건입니다. 서버는 한 파일에 최대 {MAX_RECORDS}건까지 처리합니다.")
    if not validation_path.is_file():
        raise InputError(f"검증 결과 파일이 없습니다: {validation_path} (청킹의 {VALIDATION_NAME})")
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{validation_path}가 JSON이 아닙니다: {exc}") from exc
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        raise InputError(f"{VALIDATION_NAME}의 ok가 true가 아닙니다. 검증을 통과한 청크만 적재합니다.")
    summary = validation.get("summary")
    expected = summary.get("chunks") if isinstance(summary, dict) else None
    if not _is_int(expected) or expected != len(records):
        # 서버도 같은 검사를 한다 (chunks 필드가 정수가 아니거나 다르면 422)
        raise InputError(f"청크 수가 검증 결과와 다릅니다: 파일 {len(records)}, 검증 {expected}")

    try:
        law_keys = {record["natural_key"]["law_key"] for record in records}
    except (KeyError, TypeError) as exc:
        raise InputError("natural_key에 law_key·article_key·part가 필요합니다 (법령 청크 형식이 아님).") from exc
    if len(law_keys) != 1:
        raise InputError(f"법령이 하나가 아닙니다: {sorted(map(str, law_keys))}")
    external_id = law_keys.pop()
    if not isinstance(external_id, str) or not external_id.strip() or len(external_id) > 100:
        raise InputError("law_key는 1~100자 문자열이어야 합니다 (kb_sources.external_id).")

    seen: set[str] = set()
    collections: set[Any] = set()
    for record in records:
        key = record["natural_key"]
        article, part = key.get("article_key"), key.get("part")
        if not isinstance(article, str) or not article.strip() or not _is_int(part) or part < 0:
            raise InputError(f"natural_key의 article_key·part가 올바르지 않습니다: {key}")
        label = natural_key_str(record)
        if label in seen:
            raise InputError(f"natural_key가 중복됩니다: {label}")
        seen.add(label)
        where = f"({label})"
        if record.get("source_type") != SOURCE_TYPE or record.get("collection") != COLLECTION:
            raise InputError(f"source_type/collection이 {SOURCE_TYPE}/{COLLECTION}가 아닙니다 {where}")
        collections.add(record.get("collection"))
        if record.get("site_id") != 0 or record.get("flagged") is not False:
            raise InputError(f"공공 데이터는 site_id=0, flagged=false여야 합니다 {where}")
        for field in ("text", "embed_text"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise InputError(f"{field}가 비어 있습니다 {where}")
        if len(str(record.get("locator", ""))) > MAX_LOCATOR:
            raise InputError(f"locator가 {MAX_LOCATOR}자를 넘습니다 {where}")
    return external_id


def load_index_map(path: Path, external_id: str, *, allow_missing: bool = False) -> dict[str, int]:
    """서버는 이 파일이 없으면 422다. allow_missing(--update-index-map)일 때만 없는 파일을 빈 대응표로 본다."""
    if not path.is_file():
        if allow_missing:
            return {}
        raise InputError(
            f"{INDEX_MAP_NAME}가 없습니다: {path}. 번호의 유일한 보관처이니 백업본이 있으면 먼저 복원하세요. "
            "법령을 처음 올리는 거라면 --update-index-map 으로 새로 만들 수 있습니다."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}가 JSON이 아닙니다: {exc}") from exc
    if not isinstance(data, dict) or data.get("source_type") != SOURCE_TYPE or data.get("external_id") != external_id:
        raise InputError(f"대응표가 이 법령({external_id})의 것이 아닙니다: {path}")
    indexes = data.get("indexes")
    if not isinstance(indexes, dict) or any(
        not isinstance(key, str) or not _is_int(value) or value < 0 for key, value in indexes.items()
    ):
        raise InputError(f"대응표 형식이 올바르지 않습니다: {path}")
    if len(set(indexes.values())) != len(indexes):
        raise InputError("대응표에 같은 chunk_index가 두 번 나옵니다.")
    return dict(indexes)


def assign_indexes(records: Sequence[dict[str, Any]], existing: dict[str, int]) -> dict[str, int]:
    """대응표에 없는 natural_key에만 새 번호를 준다. 기존 번호는 바꾸거나 재사용하지 않는다 (예전 ingest_law와 같은 규칙)."""
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
    """대응표를 원자적으로 쓴다. 이미 파일이 있으면 .bak 으로 한 벌 남긴다 (번호의 유일한 보관처)."""
    body = {"source_type": SOURCE_TYPE, "external_id": external_id, "indexes": indexes}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    temp.replace(path)


@dataclass(frozen=True)
class ExpectedPoint:
    record: dict[str, Any]
    point_id: str
    label: str  # 사람이 읽는 이름 (locator)


def expected_points(records: Sequence[dict[str, Any]], indexes: dict[str, int], external_id: str) -> list[ExpectedPoint]:
    points: list[ExpectedPoint] = []
    missing: list[str] = []
    for record in records:
        key = natural_key_str(record)
        index = indexes.get(key)
        if index is None:
            missing.append(key)
            continue
        points.append(ExpectedPoint(record, point_id_for(external_id, index), str(record.get("locator") or key)))
    if missing:
        shown = ", ".join(missing[:5]) + (f" 외 {len(missing) - 5}건" if len(missing) > 5 else "")
        raise InputError(
            f"chunk_index_map.json에 없는 청크가 {len(missing)}건 있습니다 ({shown}). "
            "서버가 422를 냅니다. --update-index-map 으로 새 번호를 배정하세요 (기존 번호는 바뀌지 않습니다)."
        )
    ids = [p.point_id for p in points]
    if len(set(ids)) != len(ids):
        raise InputError("point_id가 중복됩니다.")
    return points


def default_dataset_id(records: Sequence[dict[str, Any]]) -> str:
    meta = records[0].get("law_meta")
    effective = meta.get("effective_date") if isinstance(meta, dict) else None
    if not isinstance(effective, str) or not effective.strip():
        raise InputError("law_meta.effective_date가 없어 dataset_id를 정할 수 없습니다. --dataset-id를 주세요.")
    return f"law-{effective.strip()}"


def check_dataset_id(value: str) -> str:
    if not DATASET_RE.match(value):
        raise InputError(f"dataset_id가 서버 규칙(영문·숫자·._-, 100자 이하)에 맞지 않습니다: {value!r}")
    return value


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset: offset + size]


def sample_positions(count: int, size: int = 5) -> list[int]:
    if count <= size:
        return list(range(count))
    return sorted({(i * (count - 1)) // (size - 1) for i in range(size)})


# ---------------------------------------------------------------- 대상·키


def target_of(url: str) -> str:
    return urlsplit(url).netloc or url


def is_local_url(url: str) -> bool:
    return (urlsplit(url).hostname or "") in LOCAL_HOSTS


def check_target(ingest_url: str, qdrant_url: str, allow_remote: bool) -> None:
    for label, url in (("kb-ingest", ingest_url), ("Qdrant", qdrant_url)):
        if not is_local_url(url) and not allow_remote:
            raise InputError(f"{label} 대상이 로컬이 아닙니다: {target_of(url)}. 공용 서버에 올리려면 --allow-remote를 주세요.")


def env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise InputError(f"{name} 환경 변수가 필요합니다.")
    return value


def _clean_key(value: str, where: str) -> str:
    if not value or any(ord(char) < 32 or ord(char) > 126 for char in value):
        raise InputError(f"쓰기 키가 비었거나 줄바꿈·제어 문자를 포함합니다 ({where}).")
    return value


def load_write_key(env_file: Path | None) -> str:
    """--env-file이 있으면 그 파일의 QDRANT_API_KEY (셸의 오래된 값·CRLF 문제를 피한다), 없으면 환경변수."""
    if env_file is None:
        value = os.getenv("QDRANT_API_KEY", "").strip()
        if not value:
            raise InputError("올리기에는 쓰기 키가 필요합니다: QDRANT_API_KEY 환경 변수 또는 --env-file 을 주세요.")
        return _clean_key(value, "환경 변수 QDRANT_API_KEY")
    try:
        lines = env_file.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"환경변수 파일을 읽을 수 없습니다: {env_file}") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "QDRANT_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return _clean_key(value.strip(), f"{env_file.name}의 QDRANT_API_KEY")
    raise InputError(f"환경변수 파일에 QDRANT_API_KEY가 없습니다: {env_file}")


def qdrant_headers(key: str) -> dict[str, str]:
    return {"api-key": key, "Content-Type": "application/json"}


# ---------------------------------------------------------------- 사전 점검


def preflight(
    session: Any,
    ingest_url: str,
    qdrant_url: str,
    qdrant_key: str | None,
    embedding_dim: int,
    *,
    routes: Sequence[str],
    log: Callable[[str], None] = print,
) -> None:
    """쓰기를 시작하기 전에 서버·경로·컬렉션을 확인한다 (routes가 비면 kb-ingest는 보지 않는다)."""
    if routes:
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
        paths = None
        if spec is not None and spec.status_code == 200:
            try:
                paths = spec.json().get("paths")
            except ValueError:
                paths = None
        if isinstance(paths, dict):
            absent = [route for route in routes if route not in paths]
            if absent:
                raise InputError(
                    f"이 kb-ingest에는 {', '.join(absent)} 가 없습니다. A의 최신 코드로 이미지를 다시 빌드하세요 "
                    "(docker compose up -d --build kb-ingest)."
                )
        else:
            log(f"[주의] /openapi.json을 읽지 못해 {', '.join(routes)} 존재를 확인하지 못했습니다.")

    if qdrant_key is None:
        return
    qbase = qdrant_url.rstrip("/")
    try:
        response = session.get(f"{qbase}/collections/{COLLECTION}", headers=qdrant_headers(qdrant_key), timeout=10)
    except OSError as exc:
        raise InputError(f"Qdrant에 접속할 수 없습니다 ({qbase}): {type(exc).__name__}") from exc
    if response.status_code == 404:
        raise InputError(f"Qdrant에 {COLLECTION} 컬렉션이 없습니다. python -m scripts.init_qdrant 를 먼저 실행하세요.")
    if response.status_code != 200:
        raise InputError(f"Qdrant 컬렉션 조회가 {response.status_code}를 돌려줬습니다 (키·주소 확인).")
    params = ((response.json().get("result") or {}).get("config") or {}).get("params") or {}
    vectors = params.get("vectors")
    if isinstance(vectors, dict) and isinstance(vectors.get("dense"), dict):
        size = vectors["dense"].get("size")
        if size != embedding_dim:
            raise InputError(f"{COLLECTION}의 dense 차원이 {size}입니다 (기대 {embedding_dim}).")
    else:
        log("[주의] 컬렉션의 dense 설정을 읽지 못해 차원을 확인하지 못했습니다.")
    sparse = params.get("sparse_vectors")
    if not isinstance(sparse, dict) or "sparse" not in sparse:
        raise InputError(f"{COLLECTION}에 sparse 벡터 설정이 없습니다. init_qdrant로 컬렉션을 하이브리드 설정으로 만드세요.")


# ---------------------------------------------------------------- kb-ingest 호출


def _post_json(
    session: Any,
    endpoint: str,
    token: str,
    body: dict[str, Any],
    *,
    what: str,
    retries: int,
    timeout: float,
    sleep: Callable[[float], None],
    not_found_hint: str,
) -> dict[str, Any]:
    """연결·시간 초과·5xx·408/429는 재시도, 401/403은 InputError, 그 밖의 4xx는 IngestError.

    같은 dataset_id로 다시 보내면 파일·점을 덮어쓰므로 재시도해도 안전하다.
    """
    headers = {"Authorization": f"Bearer {token}"}
    last = "알 수 없음"
    for attempt in range(retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=body, timeout=timeout)
        except OSError as exc:
            last = f"연결 오류: {type(exc).__name__}"
        else:
            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise IngestError(f"{what}: 응답이 JSON이 아닙니다.") from exc
                if not isinstance(data, dict):
                    raise IngestError(f"{what}: 응답이 JSON 객체가 아닙니다.")
                return data
            if status in (401, 403):
                raise InputError(f"kb-ingest가 인증을 거부했습니다 (HTTP {status}). KB_INGEST_API_TOKEN을 확인하세요.")
            last = f"HTTP {status} {response.text[:300]}"
            if status == 404:
                raise IngestError(f"{what}: {last}. {not_found_hint}")
            if status < 500 and status not in (408, 429):
                raise IngestError(f"{what}: 요청이 거부되었습니다: {last}")
        if attempt < retries:
            sleep(5.0 * (attempt + 1))
    raise IngestError(f"{what}: 요청이 계속 실패했습니다: {last}")


def save_vectors(
    session: Any,
    ingest_url: str,
    token: str,
    *,
    chunk_file: str,
    dataset_id: str,
    count: int,
    data_type: str | None = None,
    subdir: str | None = None,
    retries: int = 2,
    timeout: float = 1800.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST /ingest/vectors/local/file. 서버가 임베딩을 8개씩 하므로 청크 수에 비례해 오래 걸린다."""
    request: dict[str, Any] = {"dataset_id": dataset_id, "chunk_file": chunk_file}
    if data_type is not None:
        request["data_type"] = data_type
    if subdir is not None:
        request["subdir"] = subdir
    where = "LOCAL_CHUNK_DIR" if data_type is None else f"LOCAL_CHUNK_ROOT/{data_type}/{subdir or SUBDIR}/"
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
            f"kb-ingest의 {where}에 {chunk_file}가 없습니다. 05_chunks.jsonl·06_validation.json·chunk_index_map.json을 "
            "그 폴더에 복사했는지 확인하세요 (compose 기본: 호스트 out\\<data_type>\\<subdir>\\ = 컨테이너 /data/out/<data_type>/<subdir>/). "
            "서버는 이 스크립트가 읽는 파일의 사본을 자기 폴더에서 읽습니다."
        ),
    )
    if body.get("saved") != count or isinstance(body.get("saved"), bool):
        raise IngestError(f"벡터 파일 청크 수가 다릅니다: 파일 {count}, 서버 {body.get('saved')} (서버가 다른 청크 파일을 읽었을 수 있음)")
    if body.get("dataset_id") != dataset_id or body.get("file") != f"{dataset_id}.jsonl":
        raise IngestError(f"서버가 다른 파일에 저장했습니다: {body.get('dataset_id')} / {body.get('file')}")
    if body.get("collection", COLLECTION) != COLLECTION:
        raise IngestError(f"서버가 {body.get('collection')} 컬렉션용으로 만들었습니다 (기대 {COLLECTION}).")
    return body


def load_via_api(
    session: Any,
    ingest_url: str,
    token: str,
    *,
    dataset_id: str,
    count: int,
    retries: int = 2,
    timeout: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST /ingest/vectors/local/load. 서버의 LOCAL_VECTOR_DIR에 있는 파일을 서버가 Qdrant에 올린다."""
    body = _post_json(
        session,
        f"{ingest_url.rstrip('/')}{LOAD_ROUTE}",
        token,
        {"dataset_id": dataset_id, "collection": COLLECTION},
        what="Qdrant 올리기(서버)",
        retries=retries,
        timeout=timeout,
        sleep=sleep,
        not_found_hint="서버에 벡터 파일이 없습니다. --skip-save 없이 처음부터 실행하세요 (컨테이너를 다시 만들면 파일이 사라질 수 있음).",
    )
    if body.get("indexed") != count or isinstance(body.get("indexed"), bool):
        raise IngestError(f"올라간 점 수가 다릅니다: 기대 {count}, 서버 {body.get('indexed')}")
    return body


# ---------------------------------------------------------------- 벡터 파일 (호스트)


def _finite_numbers(values: Any) -> bool:
    return isinstance(values, list) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values
    )


def check_vector_file(
    path: Path, points: Sequence[ExpectedPoint], dim: int, *, require_ledger_ids: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """서버가 만든 벡터 파일을 올리기 전에 검사한다. (올릴 점 목록, 문제 목록).

    장부 id(kb_source_id·kb_chunk_id)는 둘 다 null이거나 둘 다 양의 정수이고 kb_chunk_id는 겹치지 않는다.
    require_ledger_ids 가 True 면 null 인 점도 문제로 본다 (검색 결과 doc_id가 비기 때문).

    기대 point id와 정확히 일치해야 한다. 본문(payload.text)이 05와 같은지도 본다 — 서버가 옛 청크 파일을
    읽었거나 다른 대응표를 썼다면 여기서 드러난다.
    """
    if not path.is_file():
        raise IngestError(
            f"서버가 만든 벡터 파일이 호스트에 없습니다: {path}. kb-ingest의 LOCAL_VECTOR_DIR이 이 폴더에 마운트됐는지 "
            "확인하세요. 마운트가 어렵다면 --load-via api 로 서버가 직접 올리게 할 수 있습니다."
        )
    expected = {p.point_id: p for p in points}
    problems: list[str] = []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    chunk_ids: dict[int, str] = {}
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
            vector, payload = item["vector"], item["payload"]
            label = f"{point.label} ({pid[:8]})"
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
            if payload.get("source_type") != SOURCE_TYPE or payload.get("site_id") != 0 or payload.get("flagged") is not False:
                problems.append(f"payload의 source_type/site_id/flagged가 이상함: {label}")
            if payload.get("text") != point.record["text"]:
                problems.append(f"payload.text가 05와 다름 (서버가 다른 청크 파일을 읽었을 수 있음): {label}")
            ledger = {name: payload.get(name) for name in ("kb_source_id", "kb_chunk_id")}
            bad = [name for name, value in ledger.items() if value is not None and not (type(value) is int and value > 0)]
            if bad:
                problems.append(f"장부 id가 양의 정수나 null이 아님 ({', '.join(bad)}): {label}")
            elif (ledger["kb_source_id"] is None) != (ledger["kb_chunk_id"] is None):
                problems.append(f"kb_source_id와 kb_chunk_id 중 하나만 있음: {label}")
            elif ledger["kb_chunk_id"] is None:
                if require_ledger_ids:
                    problems.append(f"장부 id(kb_source_id·kb_chunk_id)가 없음 (--require-ledger-ids): {label}")
            elif ledger["kb_chunk_id"] in chunk_ids:
                problems.append(f"kb_chunk_id {ledger['kb_chunk_id']}가 겹침: {label} / {chunk_ids[ledger['kb_chunk_id']]}")
            else:
                chunk_ids[ledger["kb_chunk_id"]] = label
            result.append({"id": pid, "vector": vector, "payload": payload})
    missing = [expected[pid].label for pid in expected if pid not in seen]
    if missing:
        problems.append(_short("벡터 파일에 없는 청크", missing, 8))
    return result, problems


def _short(label: str, items: list[str], limit: int) -> str:
    shown = "; ".join(items[:limit])
    more = f" 외 {len(items) - limit}건" if len(items) > limit else ""
    return f"{label} {len(items)}건: {shown}{more}"


def upsert_points(
    session: Any,
    qdrant_url: str,
    write_key: str,
    points: Sequence[dict[str, Any]],
    *,
    retries: int = 2,
    timeout: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """벡터 파일의 점을 32개씩 Qdrant에 올린다 (A의 upsert_qdrant_jsonl.py와 같은 요청). 같은 id는 덮어쓴다."""
    endpoint = f"{qdrant_url.rstrip('/')}/collections/{COLLECTION}/points?wait=true"
    total = len(points)
    done = 0
    for batch in batched(list(points), UPSERT_BATCH):
        last = "알 수 없음"
        for attempt in range(retries + 1):
            try:
                response = session.put(endpoint, headers=qdrant_headers(write_key), json={"points": list(batch)}, timeout=timeout)
            except OSError as exc:
                last = f"연결 오류: {type(exc).__name__}"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        ok = response.json().get("status") == "ok"
                    except (ValueError, AttributeError):
                        ok = False
                    if not ok:
                        raise IngestError(f"Qdrant가 올리기를 완료하지 않았습니다 (배치 {done // UPSERT_BATCH + 1}): {response.text[:300]}")
                    break
                if status in (401, 403):
                    raise InputError(f"Qdrant가 쓰기 키를 거부했습니다 (HTTP {status}). QDRANT_API_KEY(쓰기 키)를 확인하세요.")
                last = f"HTTP {status} {response.text[:300]}"
                if status < 500 and status not in (408, 429):
                    raise IngestError(
                        f"Qdrant 올리기가 거부되었습니다 (배치 {done // UPSERT_BATCH + 1}/{(total + UPSERT_BATCH - 1) // UPSERT_BATCH}): {last}. "
                        "앞선 배치는 반영됐을 수 있습니다 (같은 명령을 다시 실행하면 덮어씁니다)."
                    )
            if attempt < retries:
                sleep(5.0 * (attempt + 1))
        else:
            raise IngestError(
                f"Qdrant 올리기가 계속 실패했습니다 (배치 {done // UPSERT_BATCH + 1}): {last}. "
                "앞선 배치는 반영됐을 수 있습니다 (같은 명령을 다시 실행하면 덮어씁니다)."
            )
        done += len(batch)
        log(f"올리기: {done}/{total}")
    return done


# ---------------------------------------------------------------- 검증


def _count(session: Any, base: str, key: str, must: list[dict[str, Any]]) -> int | None:
    response = session.post(
        f"{base}/collections/{COLLECTION}/points/count",
        headers=qdrant_headers(key),
        json={"exact": True, "filter": {"must": must}},
        timeout=30,
    )
    if response.status_code != 200:
        return None
    return int(response.json()["result"]["count"])


def compare_payload(point: ExpectedPoint, payload: dict[str, Any]) -> list[str]:
    record = point.record
    diffs: list[str] = []
    if payload.get("source_type") != SOURCE_TYPE:
        diffs.append(f"source_type {payload.get('source_type')}")
    if payload.get("site_id") != 0:
        diffs.append(f"site_id {payload.get('site_id')}")
    if payload.get("flagged") is not False:
        diffs.append(f"flagged {payload.get('flagged')}")
    for name in ("text", "title", "locator", "year", "url"):
        if name in record and payload.get(name) != record[name]:
            diffs.append(f"{name} 다름")
    for axis in ("hazard", "work_type", "cas"):
        if axis in record and sorted(payload.get(axis) or []) != sorted(record[axis] or []):
            diffs.append(f"{axis} 다름")
    return diffs


def verify(
    session: Any,
    qdrant_url: str,
    qdrant_key: str,
    points: Sequence[ExpectedPoint],
    *,
    embedding_dim: int,
    max_report: int = 8,
    require_ledger_ids: bool = False,
) -> tuple[list[str], list[str]]:
    """(문제 목록, 참고 목록). 문제가 비어 있으면 통과. 참고는 실패가 아니다 (기대보다 많은 점 등).

    require_ledger_ids 가 True 면 장부 id가 없는 점을 문제로 올린다 (False면 참고).
    """
    problems: list[str] = []
    notes: list[str] = []
    base = qdrant_url.rstrip("/")
    expected = len(points)

    actual = _count(session, base, qdrant_key, [{"key": "source_type", "match": {"value": SOURCE_TYPE}}])
    if actual is None:
        problems.append("Qdrant 점 수 조회 실패")
    elif actual < expected:
        problems.append(f"Qdrant 점 수(source_type={SOURCE_TYPE}): 기대 {expected}, 실제 {actual}")
    elif actual > expected:
        notes.append(
            f"Qdrant의 {SOURCE_TYPE} 점이 {actual}개로 이번 청크({expected})보다 많습니다. 다른 법령이거나 개정으로 사라진 조의 옛 청크일 수 있습니다 "
            "(이 스크립트는 삭제하지 않습니다)."
        )

    found: dict[str, dict[str, Any]] = {}
    for chunk in batched([p.point_id for p in points], RETRIEVE_BATCH):
        response = session.post(
            f"{base}/collections/{COLLECTION}/points",
            headers=qdrant_headers(qdrant_key),
            json={"ids": list(chunk), "with_payload": True, "with_vector": False},
            timeout=60,
        )
        if response.status_code != 200:
            problems.append(f"Qdrant 점 조회 실패: HTTP {response.status_code}")
            continue
        for point in response.json()["result"]:
            found[str(point["id"])] = point
    missing: list[str] = []
    differ: list[str] = []
    linked = 0
    for point in points:
        got = found.get(point.point_id)
        if got is None:
            missing.append(point.label)
            continue
        payload = got.get("payload") or {}
        diffs = compare_payload(point, payload)
        if diffs:
            differ.append(f"{point.label}({', '.join(diffs)})")
        if _is_int(payload.get("kb_source_id")) and _is_int(payload.get("kb_chunk_id")):
            linked += 1
    if missing:
        problems.append(_short("Qdrant에 없는 점", missing, max_report))
    if differ:
        problems.append(_short("payload가 05와 다른 점", differ, max_report))
    present = expected - len(missing)
    if present and linked < present:
        message = (
            f"장부 연결: kb_source_id·kb_chunk_id가 있는 점 {linked}/{present}. "
            "없는 점은 검색 결과 doc_id가 비어 나옵니다 (서버가 장부 id를 채웠는지 확인)."
        )
        (problems if require_ledger_ids else notes).append(message)

    samples = [points[i] for i in sample_positions(expected)]
    if samples:
        response = session.post(
            f"{base}/collections/{COLLECTION}/points",
            headers=qdrant_headers(qdrant_key),
            json={"ids": [s.point_id for s in samples], "with_payload": False, "with_vector": ["dense", "sparse"]},
            timeout=60,
        )
        if response.status_code != 200:
            problems.append(f"Qdrant 표본 조회 실패: HTTP {response.status_code}")
        else:
            by_id = {str(p["id"]): p for p in response.json()["result"]}
            for sample in samples:
                got = by_id.get(sample.point_id)
                if got is None:
                    continue  # 없는 점은 위에서 이미 알렸다
                label = f"{sample.label} ({sample.point_id[:8]})"
                vector = got.get("vector") or {}
                dense = vector.get("dense")
                if not isinstance(dense, list) or len(dense) != embedding_dim:
                    problems.append(f"표본 dense 차원이 {embedding_dim}가 아닙니다: {label}")
                sparse = vector.get("sparse")
                if not isinstance(sparse, dict) or not sparse.get("indices"):
                    problems.append(f"표본 sparse 벡터가 비어 있습니다 (옛 dense-only 점이거나 임베딩 이미지가 옛 버전?): {label}")
    return problems, notes


# ---------------------------------------------------------------- 실행


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="법령 청크를 벡터 파일을 거쳐 Qdrant에 적재하고 검증한다 (장부 등록은 서버가 한다).")
    parser.add_argument("--chunks", type=Path, default=Path("out/law/05_chunks.jsonl"), help="청크 파일 (같은 폴더에 06_validation.json, chunk_index_map.json)")
    parser.add_argument("--dataset-id", default=None, help="벡터 파일 이름 (기본: law-<시행일>). 같은 이름이면 파일을 덮어쓴다")
    parser.add_argument("--vector-dir", type=Path, default=Path("out/vectors"),
                        help="서버의 LOCAL_VECTOR_DIR이 호스트에서 보이는 폴더 (--load-via file에서 읽는다)")
    parser.add_argument("--load-via", choices=("file", "api"), default="file",
                        help="Qdrant 올리기: file=호스트의 벡터 파일을 직접 (A 문서의 방식, 쓰기 키 필요) / api=서버의 /ingest/vectors/local/load")
    parser.add_argument("--env-file", type=Path, default=None, help="QDRANT_API_KEY를 읽을 .env (예: infra의 docker/.env). 없으면 환경변수")
    parser.add_argument("--dry-run", action="store_true", help="계획만 출력 (접속·파일 쓰기 없음)")
    parser.add_argument("--verify-only", action="store_true", help="올리기 없이 검증만")
    parser.add_argument("--skip-save", action="store_true", help="벡터 파일을 새로 만들지 않고 이미 있는 파일을 올린다 (재임베딩 없음)")
    parser.add_argument("--no-load", action="store_true", help="벡터 파일만 만들고 Qdrant에는 올리지 않는다 (장부 id 확인용으로도 쓴다)")
    parser.add_argument("--data-type", choices=("guides", "none"), default=DATA_TYPE,
                        help="요청의 data_type. 서버는 <LOCAL_CHUNK_ROOT>/<data_type>/<subdir>/ 에서 청크를 읽는다 (기본 guides). "
                             "none=보내지 않음(data_type이 없던 옛 서버)")
    parser.add_argument("--subdir", default=SUBDIR, help="요청의 subdir (기본 law). --data-type none에서는 쓰지 않는다")
    parser.add_argument("--update-index-map", action="store_true",
                        help="대응표에 없는 조문에 새 chunk_index를 배정해 chunk_index_map.json을 갱신 (없으면 새로 만듦, .bak 보관)")
    parser.add_argument("--require-ledger-ids", action=argparse.BooleanOptionalAction, default=True,
                        help="모든 점에 kb_source_id·kb_chunk_id가 있어야 올림 (기본 켬, 끄려면 --no-require-ledger-ids)")
    parser.add_argument("--allow-remote", action="store_true", help="로컬이 아닌 kb-ingest/Qdrant 사용을 허용")
    parser.add_argument("--ingest-url", default=os.getenv("KB_INGEST_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:60011"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    parser.add_argument("--save-timeout", type=float, default=1800.0, help="벡터 파일 만들기 요청 제한 시간(초). 서버가 8개씩 임베딩해 오래 걸린다")
    args = parser.parse_args(argv)
    if args.verify_only and (args.skip_save or args.no_load):
        parser.error("--verify-only는 --skip-save·--no-load와 함께 쓸 수 없습니다.")
    if args.skip_save and args.no_load:
        parser.error("--skip-save와 --no-load를 함께 쓰면 할 일이 없습니다.")
    if args.verify_only and args.update_index_map:
        parser.error("--verify-only는 대응표를 바꾸지 않습니다. --update-index-map과 함께 쓸 수 없습니다.")
    if args.embedding_dim <= 0 or args.save_timeout <= 0:
        parser.error("--embedding-dim과 --save-timeout은 양수여야 합니다.")
    if not DATASET_RE.match(args.subdir):
        parser.error("--subdir이 서버 규칙(영문·숫자·._-, 100자 이하)에 맞지 않습니다.")
    if args.data_type == "none" and args.subdir != SUBDIR:
        parser.error("--data-type none(옛 서버 방식)에서는 subdir를 보내지 않으므로 --subdir을 바꿀 수 없습니다.")
    base = args.chunks.parent
    args.validation = base / VALIDATION_NAME
    args.index_map = base / INDEX_MAP_NAME
    return args


def server_location(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """요청에 실을 (data_type, subdir). data_type이 none이면 둘 다 보내지 않는다(옛 서버). 속성이 없는 Namespace도 옛 방식으로 본다."""
    data_type = getattr(args, "data_type", "none")
    if data_type in (None, "none"):
        return None, None
    return data_type, getattr(args, "subdir", None) or SUBDIR


def print_plan(
    records: Sequence[dict[str, Any]],
    external_id: str,
    points: Sequence[ExpectedPoint],
    dataset_id: str,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> None:
    log(f"법령: {records[0].get('title')} (external_id={external_id})")
    log(f"청크 {len(records)}건 / 벡터 파일 {dataset_id}.jsonl / Qdrant {COLLECTION}")
    log(f"  첫 청크 {points[0].label}: point_id={points[0].point_id}")
    log(f"  마지막 청크 {points[-1].label}: point_id={points[-1].point_id}")
    log(f"대상: kb-ingest {target_of(args.ingest_url)}, Qdrant {target_of(args.qdrant_url)}")
    data_type, subdir = server_location(args)
    if data_type is None:
        log("서버가 읽는 위치: LOCAL_CHUNK_DIR (옛 서버 방식: data_type·subdir를 보내지 않음)")
    else:
        log(f"서버가 읽는 위치: <LOCAL_CHUNK_ROOT>/{data_type}/{subdir}/{args.chunks.name} (요청에 data_type={data_type}, subdir={subdir})")
    steps = []
    if args.verify_only:
        steps.append("검증만")
    else:
        if not args.skip_save:
            steps.append(f"벡터 파일 만들기 (POST {SAVE_ROUTE}, 서버가 8개씩 임베딩)")
        if not args.no_load:
            steps.append("Qdrant 올리기 " + ("(호스트의 벡터 파일을 32개씩, 쓰기 키 필요)" if args.load_via == "file" else f"(POST {LOAD_ROUTE})"))
            steps.append("검증 (점 수, 전수 조회, 표본 벡터)")
    log("할 일: " + " → ".join(steps))
    if getattr(args, "require_ledger_ids", False):
        log("장부 id(kb_source_id·kb_chunk_id): 없는 점이 있으면 올리지 않습니다 (검색 결과 doc_id에 필요)")
    else:
        log("장부 id(kb_source_id·kb_chunk_id): 필수로 보지 않습니다 (--no-require-ledger-ids)")


def run(
    args: argparse.Namespace,
    *,
    session: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> int:
    records = load_records(args.chunks)
    external_id = check_records(records, args.validation)
    if not CHUNK_FILE_RE.match(args.chunks.name):
        raise InputError(f"청크 파일 이름 {args.chunks.name!r}을 서버가 받지 못합니다 (영문·숫자·._- 와 .jsonl, 파일 이름만).")
    update_map = getattr(args, "update_index_map", False)
    saved_map = load_index_map(args.index_map, external_id, allow_missing=update_map)
    indexes = assign_indexes(records, saved_map) if update_map else saved_map
    added = len(indexes) - len(saved_map)
    points = expected_points(records, indexes, external_id)
    dataset_id = check_dataset_id(args.dataset_id or default_dataset_id(records))

    need_save = not (args.verify_only or args.skip_save)
    need_load = not (args.verify_only or args.no_load)
    need_verify = not args.no_load
    use_file = need_load and args.load_via == "file"
    use_api = need_load and args.load_via == "api"
    check_target(args.ingest_url, args.qdrant_url, args.allow_remote)

    print_plan(records, external_id, points, dataset_id, args, log)
    if update_map:
        log(f"chunk_index 대응표: 새 번호 {added}개 (파일 {args.index_map})")
    if args.dry_run:
        log("\n[dry-run] 접속·파일 쓰기 없이 종료합니다.")
        return 0

    # 쓰기를 시작하기 전에 필요한 설정을 모두 확인한다 (중간에 빠져 상태가 어정쩡해지지 않게)
    token = env("KB_INGEST_API_TOKEN") if (need_save or use_api) else None
    write_key = load_write_key(args.env_file) if use_file else None
    read_key: str | None = None
    if need_verify:
        read_key = os.getenv("QDRANT_READ_ONLY_API_KEY") or os.getenv("QDRANT_API_KEY") or write_key
        if not read_key:
            raise InputError("QDRANT_READ_ONLY_API_KEY(없으면 QDRANT_API_KEY) 환경 변수가 필요합니다.")
    if session is None:
        import requests  # 지연 import: 단위 시험과 --dry-run에는 필요 없다

        session = requests.Session()

    routes = ([SAVE_ROUTE] if need_save else []) + ([LOAD_ROUTE] if use_api else [])
    qdrant_key = (write_key or read_key) if need_load else read_key
    preflight(session, args.ingest_url, args.qdrant_url, qdrant_key, args.embedding_dim, routes=routes, log=log)

    if update_map and added:  # 서버가 이 파일을 읽으므로 서버를 부르기 전에 쓴다
        save_index_map(args.index_map, external_id, indexes)
        log(f"대응표를 갱신했습니다: 새 번호 {added}개 (이전 파일은 {INDEX_MAP_NAME}.bak)")

    started = clock()
    count = len(points)
    if need_save:
        log(f"\n벡터 파일 만드는 중… (서버가 {count}개를 8개씩 임베딩합니다. 몇 분 걸릴 수 있습니다)")
        data_type, subdir = server_location(args)
        body = save_vectors(session, args.ingest_url, token or "", chunk_file=args.chunks.name, dataset_id=dataset_id,
                            count=count, data_type=data_type, subdir=subdir, timeout=args.save_timeout, sleep=sleep)
        log(f"벡터 파일: {body['file']} ({body['saved']}건, 모델 {body.get('model', '?')}, {clock() - started:.0f}초)")
        if args.no_load:
            log(f"\n[완료] 벡터 파일만 만들었습니다. 올리려면: --skip-save{' --env-file …' if args.load_via == 'file' else ''}")
            return 0

    if use_file:
        vector_path = args.vector_dir / f"{dataset_id}.jsonl"
        raw_points, problems = check_vector_file(
            vector_path, points, args.embedding_dim, require_ledger_ids=getattr(args, "require_ledger_ids", False)
        )
        if problems:
            raise IngestError("벡터 파일 검사에서 문제가 나와 올리지 않았습니다:\n  - " + "\n  - ".join(problems[:10]))
        log(f"벡터 파일 검사 통과: {len(raw_points)}점 ({vector_path})")
        upsert_points(session, args.qdrant_url, write_key or "", raw_points, sleep=sleep, log=log)
    elif use_api:
        result = load_via_api(session, args.ingest_url, token or "", dataset_id=dataset_id, count=count, sleep=sleep)
        log(f"서버가 올림: {result['indexed']}점 (장부 미연결 {result.get('unlinked_payloads', '?')}점)")

    if need_verify:
        log("\n검증 중…")
        problems, notes = verify(
            session, args.qdrant_url, read_key or "", points, embedding_dim=args.embedding_dim,
            require_ledger_ids=getattr(args, "require_ledger_ids", False),
        )
        for note in notes:
            log(f"[참고] {note}")
        if problems:
            raise IngestError("검증에서 문제가 나왔습니다:\n  - " + "\n  - ".join(problems)
                              + "\n같은 명령을 다시 실행하면 같은 id를 덮어씁니다 (--skip-save: 이미 만든 벡터 파일만 다시 올림).")
        log(f"\n[검증 통과] Qdrant에 {count}건, 기대 point id 전수 확인, 표본의 dense·sparse 벡터 확인 ({clock() - started:.0f}초)")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        sys.exit(run(parse_args(argv)))
    except InputError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        sys.exit(2)
    except IngestError as exc:
        print(f"[중단] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 멈췄습니다. 같은 명령을 다시 실행하면 같은 id를 덮어씁니다.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
