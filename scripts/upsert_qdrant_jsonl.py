"""Qdrant-ready JSONL points를 컬렉션에 배치 업서트한다.

사용법:
    python -m scripts.upsert_qdrant_jsonl --file ~/Downloads/law-2026-03-02.jsonl --collection kb_guides --dry-run
    python -m scripts.upsert_qdrant_jsonl --file ~/Downloads/law-2026-03-02.jsonl --collection kb_guides --env-file ../main-infra-manage/docker/.env

실제 업서트에는 QDRANT_API_KEY 환경변수가 필요하다. 포인트 ID가 같으면 기존 점을 덮어쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COLLECTION_TYPES = {
    "kb_accidents": {"ACCIDENT_REPORT", "SIF", "NEAR_MISS"},
    "kb_guides": {"KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"},
}
MAX_BATCH_SIZE = 32


class InputError(ValueError):
    """Invalid input file or CLI configuration."""


def _valid_point_id(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False
    return False


def _validate_point(point: Any, line_number: int, collection: str, dimensions: int) -> dict[str, Any]:
    where = f"{line_number}번째 줄"
    if not isinstance(point, dict) or not {"id", "vector", "payload"} <= point.keys():
        raise InputError(f"{where}: id, vector, payload가 필요합니다.")
    if not _valid_point_id(point["id"]):
        raise InputError(f"{where}: id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.")

    vector = point["vector"]
    if not isinstance(vector, dict) or not isinstance(vector.get("dense"), list):
        raise InputError(f"{where}: vector.dense 배열이 필요합니다.")
    dense = vector["dense"]
    if len(dense) != dimensions or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in dense
    ):
        raise InputError(f"{where}: dense 벡터가 유효하지 않거나 차원이 {dimensions}이 아닙니다.")

    sparse = vector.get("sparse")
    if sparse is not None:
        if not isinstance(sparse, dict):
            raise InputError(f"{where}: vector.sparse 형식이 올바르지 않습니다.")
        indices, values = sparse.get("indices"), sparse.get("values")
        if (
            not isinstance(indices, list)
            or not isinstance(values, list)
            or len(indices) != len(values)
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != sorted(set(indices))
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            )
        ):
            raise InputError(f"{where}: sparse 벡터 형식이 올바르지 않습니다.")

    payload = point["payload"]
    if not isinstance(payload, dict):
        raise InputError(f"{where}: payload 객체가 필요합니다.")
    if payload.get("collection", collection) != collection:
        raise InputError(f"{where}: payload.collection이 요청 collection과 다릅니다.")
    if payload.get("source_type") not in COLLECTION_TYPES[collection]:
        raise InputError(f"{where}: source_type이 collection과 맞지 않습니다.")
    site_id = payload.get("site_id")
    if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
        raise InputError(f"{where}: payload.site_id가 올바르지 않습니다.")
    if not isinstance(payload.get("flagged"), bool):
        raise InputError(f"{where}: payload.flagged가 올바르지 않습니다.")
    if not isinstance(payload.get("text"), str) or not payload["text"].strip():
        raise InputError(f"{where}: payload.text가 비어 있습니다.")

    for id_field in ("kb_source_id", "kb_chunk_id"):
        value = payload.get(id_field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise InputError(f"{where}: payload.{id_field}가 올바르지 않습니다.")
    return point


def load_points(path: Path, collection: str, dimensions: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise InputError(f"파일이 없습니다: {path}")

    points = []
    seen_ids: set[str | int] = set()
    unlinked_payloads = 0
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    point = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{line_number}번째 줄이 JSON이 아닙니다: {exc.msg}") from exc
                point = _validate_point(point, line_number, collection, dimensions)
                point_id = point["id"]
                if point_id in seen_ids:
                    raise InputError(f"{line_number}번째 줄: 중복 point id입니다.")
                seen_ids.add(point_id)
                payload = point["payload"]
                if payload.get("kb_source_id") is None or payload.get("kb_chunk_id") is None:
                    unlinked_payloads += 1
                points.append(point)
    except UnicodeDecodeError as exc:
        raise InputError("JSONL 파일은 UTF-8이어야 합니다.") from exc

    if not points:
        raise InputError("업서트할 포인트가 없습니다.")
    return points, unlinked_payloads


def upsert_batch(url: str, api_key: str, points: list[dict[str, Any]], timeout: float) -> None:
    request = Request(
        url,
        data=json.dumps({"points": points}, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="PUT",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if result.get("status") != "ok":
        raise RuntimeError(f"Qdrant가 업서트를 완료하지 않았습니다: {result}")


def load_api_key(env_file: Path) -> str:
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
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
            return value.strip()
    raise InputError(f"환경변수 파일에 QDRANT_API_KEY가 없습니다: {env_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qdrant-ready JSONL 포인트를 Qdrant에 업서트합니다.")
    parser.add_argument("--file", required=True, type=Path, help="Qdrant 포인트 JSONL 파일")
    parser.add_argument("--collection", required=True, choices=sorted(COLLECTION_TYPES))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:60011"))
    parser.add_argument("--env-file", type=Path, help="../.env")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="검증만 수행하고 네트워크 요청은 보내지 않음")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size는 1~{MAX_BATCH_SIZE}이어야 합니다.")
    if args.embedding_dim <= 0 or args.timeout <= 0:
        parser.error("--embedding-dim과 --timeout은 양수여야 합니다.")

    try:
        points, unlinked_payloads = load_points(args.file.expanduser(), args.collection, args.embedding_dim)
    except InputError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        return 2

    total_batches = (len(points) + args.batch_size - 1) // args.batch_size
    print(f"검증 통과: {len(points)} 포인트, collection={args.collection}, batch={args.batch_size}")
    if unlinked_payloads:
        print(f"[주의] {unlinked_payloads} 포인트가 kb_source_id/kb_chunk_id 장부와 연결되지 않습니다.")
    if args.dry_run:
        print("[dry-run] Qdrant에는 쓰지 않았습니다.")
        return 0

    try:
        api_key = load_api_key(args.env_file) if args.env_file else os.getenv("QDRANT_API_KEY", "").strip()
    except InputError as exc:
        print(f"[설정 오류] {exc}", file=sys.stderr)
        return 2
    if not api_key or any(ord(char) < 32 or ord(char) > 126 for char in api_key):
        print("[설정 오류] QDRANT_API_KEY가 비었거나 줄바꿈을 포함합니다.", file=sys.stderr)
        return 2

    endpoint = f"{args.qdrant_url.rstrip('/')}/collections/{args.collection}/points?wait=true"
    completed = 0
    for offset in range(0, len(points), args.batch_size):
        batch = points[offset : offset + args.batch_size]
        try:
            upsert_batch(endpoint, api_key, batch, args.timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[Qdrant 오류] HTTP {exc.code}, 배치 {completed + 1}/{total_batches}: {detail}", file=sys.stderr)
            print("앞선 배치가 반영됐을 수 있습니다. 같은 파일을 재실행하면 동일 ID를 덮어씁니다.", file=sys.stderr)
            return 1
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"[Qdrant 오류] 배치 {completed + 1}/{total_batches}: {exc}", file=sys.stderr)
            print("앞선 배치가 반영됐을 수 있습니다. 같은 파일을 재실행하면 동일 ID를 덮어씁니다.", file=sys.stderr)
            return 1
        completed += 1
        print(f"업서트: {min(offset + len(batch), len(points))}/{len(points)}")

    print(f"완료: {len(points)} 포인트를 {args.collection}에 업서트했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
