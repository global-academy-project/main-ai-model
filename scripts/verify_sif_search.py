"""SIF를 Qdrant에 올린 뒤 "올라갔는가"와 "검색이 되는가"를 확인한다. 읽기만 하며 아무것도 쓰지 않는다.

  python -m scripts.verify_sif_search --dry-run              # 접속 없이 계획과 만든 검색 질문만 출력
  python -m scripts.verify_sif_search                         # Qdrant 확인 + 검색 확인
  python -m scripts.verify_sif_search --skip-search           # Qdrant만 (앱 코드·임베딩 서버 불필요)
  python -m scripts.verify_sif_search --limit 10              # 앞 10건만 올린 시험 적재 확인
  python -m scripts.verify_sif_search --out out/eval/sif_verify.json

프로젝트 루트에서 실행한다. 검색은 scripts/eval_retrieval.py와 같은 app.rag.tool_backend.build_search_backend를 쓰므로
같은 환경변수가 필요하다 (EMBEDDING_URL=http://127.0.0.1:60110/v1/embeddings 처럼 전체 경로, EMBEDDING_API_KEY,
QDRANT_URL, QDRANT_READ_ONLY_API_KEY).

1부 Qdrant 확인 (사고 데이터 적재 설계 9장 ②~⑥)
  - kb_accidents 컬렉션의 dense 차원, sparse 설정
  - source_type=SIF 점 수가 05의 청크 수와 같은가 (flagged=true는 따로 센다)
  - 기대 point id(uuid5("SIF:{external_id}:{chunk_index}")) 전수 조회, payload가 05와 같은가
  - 표본 벡터: dense 차원, sparse 비어 있지 않음
  - 태그 값(severity·accident_type·hazard·work_type)마다 Qdrant 필터 count가 05에서 센 값과 같은가
2부 검색 확인 (같은 9장 ⑦ + 품질)
  - 연기 시험: 대표 질문 4개의 결과가 규격에 맞고 관련 단어가 포함된 결과가 있는가
  - 자기 검색: 사례의 개요 앞부분으로 찾으면 그 사례가 상위에 나오는가 (인덱스·sparse·본문이 맞는지 보는 기초 시험)
  - 제목 검색: 작업명+재해형태로 찾으면 같은 제목의 사례가 상위에 나오는가 (참고용 지표, 실패 기준 없음)
  - 필터 검색: 태그 필터를 건 검색 결과가 모두 그 태그를 가진 사례인가 (필터 누수·빈 결과)
  - doc_id가 SIF:None처럼 비어 있지 않은가 (벡터 파일 방식은 kb_source_id가 null이다)

종료 코드: 0 통과 / 1 문제 있음 / 2 입력·설정 오류
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.ingest_law_vectors import (
    IngestError,
    InputError,
    _finite_numbers,
    _short,
    batched,
    is_local_url,
    load_records,
    target_of,
)
from scripts.vectorize_accidents import COMPARE_FIELDS, ExpectedPoint, _differs, check_records

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "kb_accidents"
SOURCE_TYPE = "SIF"
RETRIEVE_BATCH = 200
HIT_KS = (1, 3, 5, 10)
TAG_AXES = ("severity", "accident_type", "hazard", "work_type")
SearchFn = Callable[..., list[Any]]

SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    ("용접 불티 화재", r"화재|불티|화염|화상|폭발|용접"),
    ("밀폐공간 질식 사고", r"질식|밀폐|산소|가스|중독|맨홀|탱크"),
    ("고소 작업 추락", r"추락|떨어|고소|비계|사다리|안전대|작업발판|개구부"),
    ("전기 작업 감전", r"감전|전기|전선|전원|누전|충전"),
)
FILTER_CASES: tuple[tuple[str, str], ...] = (
    ("accident_type", "FALL"), ("accident_type", "CAUGHT"), ("accident_type", "FIRE"),
    ("work_type", "HOT"), ("work_type", "CONFINED"), ("work_type", "HEIGHT"),
    ("severity", "FATAL"), ("hazard", "ELECTRICAL"),
)
FILTER_QUERY = "작업 중 사고 예방 대책"
_OVERVIEW_RE = re.compile(r"\[재해개요\]\n(.*?)(?:\n\n\[|\Z)", re.S)
_TITLE_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")


# ---------------------------------------------------------------- 기대값 (05)


def tag_pairs(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    severity = record.get("severity")
    if isinstance(severity, str) and severity:
        pairs.append(("severity", severity))
    for axis in TAG_AXES[1:]:
        for value in record.get(axis) or []:
            pairs.append((axis, value))
    return pairs


def expected_tag_counts(points: Sequence[ExpectedPoint]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for point in points:
        if point.record.get("flagged") is True:
            continue
        for pair in set(tag_pairs(point.record)):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def overview_of(record: Mapping[str, Any]) -> str:
    match = _OVERVIEW_RE.search(record.get("text", ""))
    return " ".join((match.group(1) if match else record.get("text", "")).split())


def self_query(record: Mapping[str, Any]) -> str:
    return overview_of(record)[:80].strip()


def clean_title(record: Mapping[str, Any]) -> str:
    return _TITLE_YEAR_RE.sub("", str(record.get("title", ""))).strip()


def stride_sample(items: Sequence[Any], count: int) -> list[Any]:
    """처음부터 끝까지 고르게 count개 (결정적)."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    return [items[(i * len(items)) // count] for i in range(count)]


# ---------------------------------------------------------------- Qdrant (읽기)


def qdrant_headers(key: str) -> dict[str, str]:
    return {"api-key": key, "Content-Type": "application/json"}


def read_key() -> str:
    for name in ("QDRANT_READ_ONLY_API_KEY", "QDRANT_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            if any(ord(c) < 33 or ord(c) > 126 for c in value):
                raise InputError(f"{name}에 줄바꿈·공백·제어 문자가 있습니다.")
            return value
    raise InputError("Qdrant 읽기 키가 필요합니다: QDRANT_READ_ONLY_API_KEY 환경 변수 (Read-Host로 입력).")


def _qrequest(
    session: Any, method: str, url: str, key: str, body: dict[str, Any] | None, *, sleep: Callable[[float], None]
) -> tuple[Any, str | None]:
    """(JSON 또는 None, 오류 문구 또는 None). 연결·5xx는 2번 더 시도하고 401/403은 InputError."""
    last = "알 수 없음"
    for attempt in range(3):
        try:
            if method == "GET":
                response = session.get(url, headers=qdrant_headers(key), timeout=30)
            else:
                response = session.post(url, headers=qdrant_headers(key), json=body, timeout=60)
        except OSError as exc:
            last = f"연결 오류 {type(exc).__name__}"
        else:
            status = response.status_code
            if status in (401, 403):
                raise InputError(f"Qdrant가 읽기 키를 거부했습니다 (HTTP {status}). QDRANT_READ_ONLY_API_KEY를 확인하세요.")
            if status == 200:
                try:
                    return response.json(), None
                except ValueError:
                    return None, "응답이 JSON이 아님"
            last = f"HTTP {status}"
            if status < 500 and status not in (408, 429):
                return None, last
        if attempt < 2:
            sleep(5.0 * (attempt + 1))
    return None, last


def check_collection(session: Any, qdrant_url: str, key: str, dim: int, *, sleep: Callable[[float], None]) -> list[str]:
    data, error = _qrequest(session, "GET", f"{qdrant_url.rstrip('/')}/collections/{COLLECTION}", key, None, sleep=sleep)
    if error == "HTTP 404":
        raise InputError(f"Qdrant에 {COLLECTION} 컬렉션이 없습니다 (python -m scripts.init_qdrant).")
    if error is not None:
        return [f"컬렉션 조회 실패: {error}"]
    params = (((data or {}).get("result") or {}).get("config") or {}).get("params") or {}
    problems: list[str] = []
    vectors = params.get("vectors")
    size = vectors.get("dense", {}).get("size") if isinstance(vectors, dict) and isinstance(vectors.get("dense"), dict) else None
    if size != dim:
        problems.append(f"{COLLECTION}의 dense 차원이 {size}입니다 (기대 {dim})")
    sparse = params.get("sparse_vectors")
    if not isinstance(sparse, dict) or "sparse" not in sparse:
        problems.append(f"{COLLECTION}에 sparse 벡터 설정이 없습니다")
    return problems


def _count(session: Any, qdrant_url: str, key: str, must: list[dict[str, Any]], *, sleep: Callable[[float], None]) -> tuple[int | None, str | None]:
    data, error = _qrequest(
        session, "POST", f"{qdrant_url.rstrip('/')}/collections/{COLLECTION}/points/count", key,
        {"filter": {"must": must}, "exact": True}, sleep=sleep,
    )
    if error is not None:
        return None, error
    count = ((data or {}).get("result") or {}).get("count")
    return (count, None) if isinstance(count, int) and not isinstance(count, bool) else (None, "count가 정수가 아님")


def _match(field: str, value: Any) -> dict[str, Any]:
    return {"key": field, "match": {"value": value}}


def check_counts(
    session: Any, qdrant_url: str, key: str, points: Sequence[ExpectedPoint], *, partial: bool, sleep: Callable[[float], None]
) -> tuple[list[str], list[str]]:
    """점 수와 태그 필터 count를 05와 맞춰 본다. partial(--limit)이면 '이상'이면 통과."""
    problems: list[str] = []
    notes: list[str] = []
    expected_open = sum(1 for p in points if p.record.get("flagged") is not True)
    expected_flagged = len(points) - expected_open
    for label, flagged, expected in (("flagged=false", False, expected_open), ("flagged=true", True, expected_flagged)):
        actual, error = _count(session, qdrant_url, key, [_match("source_type", SOURCE_TYPE), _match("flagged", flagged)], sleep=sleep)
        if error is not None:
            problems.append(f"점 수 조회 실패({label}): {error}")
        elif actual < expected or (actual > expected and not partial):
            problems.append(f"SIF 점 수({label}): 기대 {expected}, 실제 {actual}")
        elif actual > expected:
            notes.append(f"SIF 점 수({label})가 {actual}개로 시험 범위({expected}개)보다 많습니다 (다른 SIF 점이 있음).")
    wrong: list[str] = []
    for (axis, value), expected in sorted(expected_tag_counts(points).items()):
        actual, error = _count(
            session, qdrant_url, key, [_match("source_type", SOURCE_TYPE), _match("flagged", False), _match(axis, value)], sleep=sleep
        )
        if error is not None:
            wrong.append(f"{axis}={value}: 조회 실패 {error}")
        elif actual < expected or (actual > expected and not partial):
            wrong.append(f"{axis}={value}: 기대 {expected}, 실제 {actual}")
    if wrong:
        problems.append(_short("태그 필터 count가 05와 다른 값", wrong, 10))
    return problems, notes


def check_points(
    session: Any, qdrant_url: str, key: str, points: Sequence[ExpectedPoint], dim: int, *, sleep: Callable[[float], None]
) -> tuple[list[str], list[str]]:
    """기대 point id를 전수 조회해 payload를 05와 대조하고, 표본의 dense·sparse 벡터를 본다."""
    problems: list[str] = []
    notes: list[str] = []
    base = f"{qdrant_url.rstrip('/')}/collections/{COLLECTION}/points"
    by_id = {p.point_id: p for p in points}
    found: set[str] = set()
    differing: list[str] = []
    linked = 0
    for chunk in batched([p.point_id for p in points], RETRIEVE_BATCH):
        data, error = _qrequest(session, "POST", base, key, {"ids": list(chunk), "with_payload": True, "with_vector": False}, sleep=sleep)
        if error is not None:
            problems.append(f"점 조회 실패: {error}")
            continue
        for item in (data or {}).get("result") or []:
            pid = item.get("id")
            point = by_id.get(pid)
            if point is None:
                continue
            found.add(pid)
            payload = item.get("payload") or {}
            diffs = [f for f in COMPARE_FIELDS if f in point.record and _differs(point.record[f], payload.get(f))]
            if diffs:
                differing.append(f"{point.label}({', '.join(diffs)})")
            if isinstance(payload.get("kb_source_id"), int) and isinstance(payload.get("kb_chunk_id"), int):
                linked += 1
    missing = [by_id[pid].label for pid in by_id if pid not in found]
    if missing:
        problems.append(_short("Qdrant에 없는 점", missing, 8))
    if differing:
        problems.append(_short("payload가 05와 다른 점", differing, 6))
    if found:
        notes.append(
            f"장부 연결(kb_source_id·kb_chunk_id): {linked}/{len(found)}"
            + (" — 0이면 벡터 파일 방식 적재입니다. 검색 결과의 doc_id를 아래 검색 확인에서 보세요." if linked == 0 else "")
        )
    samples = stride_sample([p for p in points if p.point_id in found], 5)
    if samples:
        data, error = _qrequest(
            session, "POST", base, key,
            {"ids": [s.point_id for s in samples], "with_payload": False, "with_vector": ["dense", "sparse"]}, sleep=sleep,
        )
        if error is not None:
            problems.append(f"표본 벡터 조회 실패: {error}")
        else:
            got = {item.get("id"): item for item in (data or {}).get("result") or []}
            for sample in samples:
                vector = (got.get(sample.point_id) or {}).get("vector") or {}
                dense = vector.get("dense")
                if not _finite_numbers(dense) or len(dense) != dim:
                    problems.append(f"표본 dense 차원이 {dim}가 아닙니다: {sample.label}")
                sparse = vector.get("sparse")
                if not isinstance(sparse, dict) or not sparse.get("indices"):
                    problems.append(f"표본 sparse 벡터가 비어 있습니다: {sample.label}")
    return problems, notes


# ---------------------------------------------------------------- 검색


def default_search_factory() -> SearchFn:
    sys.path.insert(0, str(ROOT))
    try:
        from app.rag.tool_backend import build_search_backend  # noqa: PLC0415  (Qdrant만 볼 때는 불러오지 않는다)
    except ImportError as exc:
        raise InputError(
            "검색 백엔드(app.rag.tool_backend)를 불러오지 못했습니다. 프로젝트 루트에서 실행하고 가상환경을 확인하세요: "
            f"{exc}"
        ) from exc
    return build_search_backend()


def field(hit: Any, name: str) -> Any:
    return hit.get(name) if isinstance(hit, Mapping) else getattr(hit, name, None)


def hit_problems(hit: Any) -> list[str]:
    """에이전트가 받는 검색 결과 규격(id uuid, doc_id, source, text, score)."""
    problems: list[str] = []
    ident = field(hit, "id")
    try:
        uuid.UUID(str(ident))
        if not isinstance(ident, str):
            raise ValueError
    except ValueError:
        problems.append(f"id가 uuid 문자열이 아님: {ident!r}")
    for name in ("doc_id", "source", "text"):
        value = field(hit, name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name}가 비어 있음")
    source = field(hit, "source")
    if isinstance(source, str) and source.strip() and not re.match(r"^.+ \(.+\)$", source):
        problems.append(f"source가 '제목 (위치)' 꼴이 아님: {source[:40]!r}")
    score = field(hit, "score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        problems.append(f"score가 유한한 숫자가 아님: {score!r}")
    return problems


def _timed(search: SearchFn, query: str, k: int, filters: Mapping[str, Any]) -> tuple[list[Any], int]:
    started = time.perf_counter()
    hits = search(query, k=k, filters=dict(filters))
    return list(hits), int((time.perf_counter() - started) * 1000)


def rank_of(ids: Sequence[str], targets: set[str]) -> int | None:
    return next((i for i, ident in enumerate(ids, 1) if ident in targets), None)


def summarize(ranks: Sequence[int | None]) -> dict[str, Any]:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for k in HIT_KS:
        out[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 4)
    return out


def _fmt_summary(name: str, s: Mapping[str, Any]) -> str:
    if not s.get("n"):
        return f"{name}: 질문 없음"
    return f"{name:<10} n={s['n']:<3} " + "  ".join(f"hit@{k} {s[f'hit@{k}']:.2f}" for k in HIT_KS) + f"  MRR {s['mrr']:.3f}"


def check_smoke(
    search: SearchFn, text_of: Mapping[str, str], *, k: int, min_relevant: int, strict: bool = True
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """strict=False(--limit 시험 적재)이면 관련 단어 부족은 문제가 아니라 참고로만 알린다 (몇 건뿐이라 주제가 없을 수 있다)."""
    problems: list[str] = []
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    for query, pattern in SMOKE_QUERIES:
        try:
            hits, ms = _timed(search, query, k, {"collection": COLLECTION, "source_type": SOURCE_TYPE})
        except Exception as exc:  # 검색 실패는 해당 질문만 오류로 보고 계속한다
            problems.append(f"연기 시험 '{query}': 검색 오류 {type(exc).__name__}: {exc}")
            rows.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not hits:
            problems.append(f"연기 시험 '{query}': 결과가 없습니다")
            rows.append({"query": query, "hits": 0, "ms": ms})
            continue
        bad = [p for hit in hits for p in hit_problems(hit)]
        if bad:
            problems.append(f"연기 시험 '{query}': 결과 규격 위반 {len(bad)}건 (예: {bad[0]})")
        ids = [str(field(h, "id")) for h in hits]
        unknown = [i for i in ids if i not in text_of]
        if unknown:
            notes.append(f"연기 시험 '{query}': 05에 없는 id {len(unknown)}건 (다른 적재분이거나 시험 범위 밖)")
        relevant = sum(1 for h in hits if re.search(pattern, str(field(h, "text")) or ""))
        if relevant < min_relevant:
            message = f"연기 시험 '{query}': 상위 {len(hits)}건 중 관련 단어가 있는 결과 {relevant}건 (기준 {min_relevant}건 이상)"
            (problems if strict else notes).append(message)
        rows.append({"query": query, "hits": len(hits), "relevant": relevant, "ms": ms, "top": [str(field(h, 'source'))[:60] for h in hits[:3]]})
    return problems, notes, rows


def check_doc_ids(search: SearchFn, queries: Sequence[str]) -> list[str]:
    notes: list[str] = []
    for query in queries:
        try:
            hits, _ = _timed(search, query, 3, {"collection": COLLECTION, "source_type": SOURCE_TYPE})
        except Exception:
            continue
        ids = {str(field(h, "doc_id")) for h in hits}
        if any(not doc_id.strip() or re.search(r"(^|:)(None|null)$", doc_id) for doc_id in ids):
            notes.append(
                f"doc_id가 비어 있거나 'SIF:None'입니다 (예: {sorted(ids)[:2]}). 벡터 파일 방식은 kb_source_id가 null이라 "
                "에이전트 인용(docId)에 그대로 나갑니다. A에게 확인이 필요합니다."
            )
            break
    return notes


def check_self_and_title(
    search: SearchFn, points: Sequence[ExpectedPoint], *, count_self: int, count_title: int, k: int
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    self_ranks: list[int | None] = []
    for point in stride_sample(points, count_self):
        query = self_query(point.record)
        try:
            hits, ms = _timed(search, query, k, {"collection": COLLECTION, "source_type": SOURCE_TYPE})
        except Exception as exc:
            errors.append(f"자기 검색 {point.label}: {type(exc).__name__}: {exc}")
            continue
        rank = rank_of([str(field(h, "id")) for h in hits], {point.point_id})
        self_ranks.append(rank)
        rows.append({"kind": "self", "target": point.label, "query": query, "rank": rank, "ms": ms})
    groups: dict[str, set[str]] = {}
    firsts: dict[str, ExpectedPoint] = {}
    for point in points:
        title = clean_title(point.record)
        groups.setdefault(title, set()).add(point.point_id)
        firsts.setdefault(title, point)
    title_ranks: list[int | None] = []
    for title in stride_sample(sorted(groups), count_title):
        try:
            hits, ms = _timed(search, title, k, {"collection": COLLECTION, "source_type": SOURCE_TYPE})
        except Exception as exc:
            errors.append(f"제목 검색 '{title[:30]}': {type(exc).__name__}: {exc}")
            continue
        rank = rank_of([str(field(h, "id")) for h in hits], groups[title])
        title_ranks.append(rank)
        rows.append({"kind": "title", "target": title, "query": title, "rank": rank, "ms": ms, "same_title_cases": len(groups[title])})
    return summarize(self_ranks), summarize(title_ranks), errors, rows


def search_with_filter(search: SearchFn, query: str, k: int, axis: str, value: str) -> tuple[list[Any], str]:
    base = {"collection": COLLECTION, "source_type": SOURCE_TYPE}
    try:
        return list(search(query, k=k, filters={**base, axis: value})), "문자열"
    except Exception as first:
        try:
            return list(search(query, k=k, filters={**base, axis: [value]})), "배열"
        except Exception:
            raise first from None


def check_filters(
    search: SearchFn, points: Sequence[ExpectedPoint], *, k: int
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    problems: list[str] = []
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    expected = expected_tag_counts(points)
    tags_of = {p.point_id: set(tag_pairs(p.record)) for p in points}
    forms: set[str] = set()
    for axis, value in FILTER_CASES:
        have = expected.get((axis, value), 0)
        if have == 0:
            continue  # 05에 없는 값은 확인할 수 없다
        try:
            hits, form = search_with_filter(search, FILTER_QUERY, k, axis, value)
        except Exception as exc:
            problems.append(f"필터 검색 {axis}={value}: 검색 오류 {type(exc).__name__}: {exc}")
            rows.append({"filter": f"{axis}={value}", "error": f"{type(exc).__name__}: {exc}"})
            continue
        forms.add(form)
        ids = [str(field(h, "id")) for h in hits]
        if not ids:
            problems.append(f"필터 검색 {axis}={value}: 05에 {have}건 있는데 결과가 없습니다 (필터가 동작하지 않거나 인덱스 문제)")
        leaked = [i for i in ids if i in tags_of and (axis, value) not in tags_of[i]]
        if leaked:
            problems.append(f"필터 검색 {axis}={value}: 상위 {len(ids)}건 중 {len(leaked)}건이 그 태그가 없는 사례입니다 (필터 누수)")
        unknown = sum(1 for i in ids if i not in tags_of)
        if unknown:
            notes.append(f"필터 검색 {axis}={value}: 05에 없는 id {unknown}건은 태그를 확인하지 못했습니다")
        if 0 < len(ids) < min(k, have):
            notes.append(f"필터 검색 {axis}={value}: 05에 {have}건이 있는데 {len(ids)}건만 돌아왔습니다")
        rows.append({"filter": f"{axis}={value}", "expected_in_05": have, "hits": len(ids), "leaked": len(leaked)})
    if forms:
        notes.append(f"태그 필터 값 형식: {'/'.join(sorted(forms))}로 호출이 통과했습니다")
    return problems, notes, rows


# ---------------------------------------------------------------- 실행


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIF 적재 후 Qdrant 상태와 검색 동작을 읽기 전용으로 확인한다.")
    parser.add_argument("--chunks", type=Path, default=Path("out/accidents/law/05_chunks.jsonl"), help="기대값으로 쓰는 05_chunks.jsonl")
    parser.add_argument("--limit", type=int, help="앞의 N건만 올린 시험 적재를 확인 (점 수는 '이상'이면 통과)")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:60011"))
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    parser.add_argument("--self-queries", type=int, default=50, help="자기 검색 질문 수")
    parser.add_argument("--title-queries", type=int, default=50, help="제목 검색 질문 수")
    parser.add_argument("--k-fetch", type=int, default=10, help="질문마다 가져올 결과 수 (hit@10까지 재려면 10 이상)")
    parser.add_argument("--min-self-hit5", type=float, default=0.9, help="자기 검색 hit@5 최소값 (미만이면 문제)")
    parser.add_argument("--min-relevant", type=int, default=2, help="연기 시험 질문마다 관련 단어가 있는 결과 최소 수")
    parser.add_argument("--skip-qdrant", action="store_true", help="1부(Qdrant 확인)를 건너뜀")
    parser.add_argument("--skip-search", action="store_true", help="2부(검색 확인)를 건너뜀 (앱 코드·임베딩 서버 불필요)")
    parser.add_argument("--dry-run", action="store_true", help="접속 없이 계획과 만든 질문만 출력")
    parser.add_argument("--allow-remote", action="store_true", help="로컬이 아닌 Qdrant 허용")
    parser.add_argument("--out", type=Path, help="결과를 JSON으로 저장")
    args = parser.parse_args(argv)
    if args.embedding_dim <= 0:
        parser.error("--embedding-dim은 양수여야 합니다")
    for name in ("limit", "self_queries", "title_queries", "k_fetch", "min_relevant"):
        value = getattr(args, name)
        if value is not None and value < (0 if name in ("self_queries", "title_queries", "min_relevant") else 1):
            parser.error(f"--{name.replace('_', '-')} 값이 올바르지 않습니다")
    if not 0 <= args.min_self_hit5 <= 1:
        parser.error("--min-self-hit5는 0~1 사이여야 합니다")
    if args.skip_qdrant and args.skip_search:
        parser.error("--skip-qdrant와 --skip-search를 함께 쓰면 확인할 것이 없습니다")
    return args


def run(
    args: argparse.Namespace,
    *,
    session: Any = None,
    search_factory: Callable[[], SearchFn] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    records = load_records(args.chunks)
    all_points = check_records(records)
    points = all_points[: args.limit] if args.limit is not None else all_points
    partial = args.limit is not None and len(points) < len(all_points)
    text_of = {p.point_id: str(p.record.get("text", "")) for p in points}
    log(f"기대값: 05의 SIF 청크 {len(points)}건" + (f" (파일 전체 {len(all_points)}건 중 앞부분)" if partial else ""))
    tags = expected_tag_counts(points)
    log("  태그 값 수: " + ", ".join(f"{axis} {sum(1 for a, _ in tags if a == axis)}종" for axis in TAG_AXES))
    log(f"  첫 점 {points[0].label} → {points[0].point_id}")
    if not is_local_url(args.qdrant_url) and not args.allow_remote:
        raise InputError(f"Qdrant 대상이 로컬이 아닙니다: {target_of(args.qdrant_url)}. 공용 서버를 쓰려면 --allow-remote를 주세요.")
    if args.dry_run:
        sample = stride_sample(points, 3)
        log("  자기 검색 질문 예: " + " | ".join(self_query(p.record)[:30] for p in sample))
        log("  제목 검색 질문 예: " + " | ".join(clean_title(p.record)[:30] for p in sample))
        log("  연기 시험 질문: " + ", ".join(q for q, _ in SMOKE_QUERIES))
        log("  필터 검색: " + ", ".join(f"{a}={v}" for a, v in FILTER_CASES if tags.get((a, v))))
        log("\n[dry-run] 접속 없이 종료합니다.")
        return 0

    problems: list[str] = []
    notes: list[str] = []
    report: dict[str, Any] = {"expected": len(points), "partial": partial}
    started = time.perf_counter()

    if not args.skip_qdrant:
        key = read_key()
        if session is None:
            import requests

            session = requests.Session()
        log("\n[1부] Qdrant 확인")
        found = check_collection(session, args.qdrant_url, key, args.embedding_dim, sleep=sleep)
        problems += found
        for step, run_step in (
            ("점 수·태그 필터 count", lambda: check_counts(session, args.qdrant_url, key, points, partial=partial, sleep=sleep)),
            ("전수 id 조회·payload 대조·표본 벡터", lambda: check_points(session, args.qdrant_url, key, points, args.embedding_dim, sleep=sleep)),
        ):
            step_problems, step_notes = run_step()
            problems += step_problems
            notes += step_notes
            log(f"  {step}: " + ("통과" if not step_problems else f"문제 {len(step_problems)}건"))
        report["qdrant_problems"] = list(problems)

    if not args.skip_search:
        log("\n[2부] 검색 확인")
        search = (search_factory or default_search_factory)()
        smoke_p, smoke_n, smoke_rows = check_smoke(search, text_of, k=5, min_relevant=args.min_relevant, strict=not partial)
        problems += smoke_p
        notes += smoke_n
        log("  연기 시험: " + ("통과" if not smoke_p else f"문제 {len(smoke_p)}건"))
        notes += check_doc_ids(search, [q for q, _ in SMOKE_QUERIES[:2]])
        self_s, title_s, errors, rows = check_self_and_title(
            search, points, count_self=args.self_queries, count_title=args.title_queries, k=args.k_fetch
        )
        problems += [f"검색 오류: {e}" for e in errors[:5]] + ([f"검색 오류 외 {len(errors) - 5}건"] if len(errors) > 5 else [])
        log("  " + _fmt_summary("자기 검색", self_s))
        log("  " + _fmt_summary("제목 검색", title_s) + "  (참고용)")
        if self_s.get("n") and self_s["hit@5"] < args.min_self_hit5:
            problems.append(f"자기 검색 hit@5 {self_s['hit@5']:.2f} < 기준 {args.min_self_hit5:.2f} — 인덱스·sparse·본문이 맞는지 확인")
        filter_p, filter_n, filter_rows = check_filters(search, points, k=args.k_fetch)
        problems += filter_p
        notes += filter_n
        log("  필터 검색: " + ("통과" if not filter_p else f"문제 {len(filter_p)}건"))
        all_ms = sorted(r["ms"] for r in smoke_rows + rows if "ms" in r)
        if all_ms:
            log(f"  검색 지연: 중앙값 {all_ms[len(all_ms) // 2]}ms, 최대 {all_ms[-1]}ms ({len(all_ms)}회)")
        report.update(smoke=smoke_rows, self_search=self_s, title_search=title_s, queries=rows, filters=filter_rows)

    log("")
    for note in notes:
        log(f"[참고] {note}")
    report.update(problems=problems, notes=notes, seconds=round(time.perf_counter() - started, 1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"저장: {args.out}")
    if problems:
        shown = "\n  - ".join(problems[:20]) + (f"\n  - … 외 {len(problems) - 20}건" if len(problems) > 20 else "")
        raise IngestError(f"확인에서 문제가 나왔습니다:\n  - {shown}")
    log(f"[통과] {report['seconds']}초")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        code = run(args)
    except InputError as exc:
        print(f"[입력 오류] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except IngestError as exc:
        print(f"[문제] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 멈췄습니다. 읽기만 하므로 다시 실행해도 안전합니다.", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
