#!/usr/bin/env python3
"""권장 경로 적재: {source, chunks} 요청 JSON을 검증하고 kb-ingest에 보낸다.

Swagger 문서의 권장 경로(POST /ingest/accidents, POST /ingest/guides)를 스크립트로 부른다.
서버가 장부(agent_kb) 등록, 임베딩, Qdrant upsert, 완료 상태 기록을 한 요청으로 처리한다.

- 입력: 파일 하나 = 요청 하나({"source": {...}, "chunks": [...]}). 원본 하나당 요청 하나.
- 기본은 검증만 한다. 실제로 보내려면 --send 를 준다(운영 자료를 실수로 덮어쓰지 않게).
- 모든 파일을 먼저 검증하고, 하나라도 어기면 아무것도 보내지 않는다(종료 코드 2).
- 토큰은 환경변수 KB_INGEST_API_TOKEN 으로만 받는다. 출력·보고 파일에 쓰지 않는다.
- 표준 라이브러리만 쓴다.

종료 코드: 0 성공, 1 전송·응답 검증 실패, 2 입력·설정 오류(아무것도 보내지 않음).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ACCIDENT_TYPES = ("ACCIDENT_REPORT", "SIF")
GUIDE_TYPES = ("KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW")
ENDPOINTS = {t: "/ingest/accidents" for t in ACCIDENT_TYPES}
ENDPOINTS.update({t: "/ingest/guides" for t in GUIDE_TYPES})

MAX_CHUNKS = 5000
# 임베딩 서버는 한 번에 8개까지만 받는 설정이었다(프로젝트 기록). 새 API가 나눠 처리하는지는 미확인이라 경고만 한다.
WARN_CHUNKS = 8

# Spring V5 tag_codes 스냅샷(11/10). 서버는 태그 값을 검증하지 않으므로 여기서 막는다. 코드가 바뀌면 갱신한다.
TAG_CODES = {
    "severity": {"NEAR_MISS", "MINOR", "SERIOUS", "FATAL"},
    "accident_type": {
        "FIRE", "EXPLOSION", "LEAK", "ASPHYXIATION", "FALL",
        "TRIP", "CAUGHT", "STRUCK", "COLLAPSE", "ELECTRIC",
    },
    "hazard": {"MECHANICAL", "ELECTRICAL", "CHEMICAL", "ENVIRONMENTAL", "HUMAN"},
    "work_type": {"GENERAL", "HOT", "CONFINED", "HEIGHT", "ELECTRICAL", "LIFTING"},
}
ARRAY_TAGS = ("accident_type", "hazard", "work_type", "cas", "ghs")
CAS_RE = re.compile(r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# 서버가 장부와 요청에서 채우므로 payload에 넣어도 덮어써진다.
SERVER_FIELDS = (
    "kb_source_id", "kb_chunk_id", "source_type", "site_id",
    "flagged", "title", "locator", "url", "text",
)
SIF_REQUIRED = ("natural_key", "tier", "cate", "disaster_type", "situation")
LAW_REQUIRED = ("natural_key", "article_sha256", "law_meta", "path", "hierarchy")

HINTS = {
    401: "Authorize 토큰(KB_INGEST_API_TOKEN) 누락·불일치",
    404: "경로 또는 /ready 확인",
    422: "source_type과 API 경로, 중복 chunk_index, 빈 본문, 태그 배열 형식 확인",
    502: "임베딩 서버/Qdrant 오류. kb-ingest, python-embedding-model, qdrant-server 로그 확인",
    503: "필수 키/URL, DB 연결, agent_kb 장부 등록·검증 오류 확인",
}


class InputError(Exception):
    """입력·설정 오류. 아무것도 보내지 않고 종료 코드 2."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_len(v, lo: int, hi: int) -> bool:
    return isinstance(v, str) and lo <= len(v) <= hi


def validate_request(req, tag_codes=None, allow_unknown_tags: bool = False):
    """요청 하나를 검사한다. (errors, warnings, info) 를 돌려준다."""
    tag_codes = tag_codes or TAG_CODES
    errors: list[str] = []
    warns: list[str] = []
    info = {"source_type": None, "external_id": None, "endpoint": None, "chunks": 0}

    if not isinstance(req, dict):
        return ["최상위가 객체가 아니다"], warns, info
    src = req.get("source")
    chunks = req.get("chunks")
    if not isinstance(src, dict):
        return ["source 객체가 없다"], warns, info
    if not isinstance(chunks, list):
        return ["chunks 배열이 없다"], warns, info

    stype = src.get("source_type")
    if stype not in ENDPOINTS:
        errors.append(f"source.source_type 이 허용 값이 아니다: {stype!r} (허용: {', '.join(ENDPOINTS)})")
        return errors, warns, info
    ext = src.get("external_id")
    info.update(source_type=stype, external_id=ext if isinstance(ext, str) else None,
                endpoint=ENDPOINTS[stype], chunks=len(chunks))

    if not _str_len(ext, 1, 100):
        errors.append("source.external_id 는 1~100자 문자열이어야 한다")
    elif ":" in ext or any(c.isspace() for c in ext):
        warns.append("source.external_id 에 ':' 또는 공백이 있다. point ID 이름에 섞이므로 피하는 것이 좋다")
    if not _str_len(src.get("title"), 1, 300):
        errors.append("source.title 은 1~300자 문자열이어야 한다")
    url, fpath = src.get("url"), src.get("file_path")
    if not ((isinstance(url, str) and url.strip()) or (isinstance(fpath, str) and fpath.strip())):
        errors.append("source.url 또는 source.file_path 중 하나는 있어야 한다")
    for key in ("url", "file_path"):
        if key in src and src[key] is not None and not _str_len(src[key], 0, 500):
            errors.append(f"source.{key} 는 500자 이하 문자열이어야 한다")
    if src.get("license") is not None and not _str_len(src["license"], 0, 50):
        errors.append("source.license 는 50자 이하 문자열이어야 한다")
    if src.get("publisher") is not None and not _str_len(src["publisher"], 0, 100):
        errors.append("source.publisher 는 100자 이하 문자열이어야 한다")
    sha = src.get("content_sha256")
    if sha is not None and not (isinstance(sha, str) and SHA256_RE.match(sha)):
        errors.append("source.content_sha256 은 64자리 소문자 hex 여야 한다")
    if "flagged" in src and not isinstance(src["flagged"], bool):
        errors.append("source.flagged 는 true/false 여야 한다")

    if not chunks:
        errors.append("chunks 가 비어 있다")
    if len(chunks) > MAX_CHUNKS:
        errors.append(f"chunks 가 요청당 최대 {MAX_CHUNKS}개를 넘는다: {len(chunks)}")
    elif len(chunks) > WARN_CHUNKS:
        warns.append(
            f"chunks 가 {len(chunks)}개다. 임베딩 서버가 한 번에 {WARN_CHUNKS}개까지만 받는 설정이면 "
            "502가 날 수 있다. 처음에는 적은 수로 시험한다"
        )

    seen_idx: set[int] = set()
    law_source_hashes: set[str] = set()
    for i, ch in enumerate(chunks):
        p = f"chunks[{i}]"
        if not isinstance(ch, dict):
            errors.append(f"{p} 가 객체가 아니다")
            continue
        idx = ch.get("chunk_index")
        if not _is_int(idx) or idx < 0:
            errors.append(f"{p}.chunk_index 는 0 이상의 정수여야 한다")
        elif idx in seen_idx:
            errors.append(f"{p}.chunk_index {idx} 가 중복이다")
        else:
            seen_idx.add(idx)
        text = ch.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{p}.text 가 비어 있다")
        elif len(text) > 32768:
            errors.append(f"{p}.text 가 32,768자를 넘는다")
        emb = ch.get("embedding_text")
        if emb is not None and (not isinstance(emb, str) or len(emb) > 32768):
            errors.append(f"{p}.embedding_text 는 32,768자 이하 문자열이어야 한다")
        if ch.get("locator") is not None and not _str_len(ch["locator"], 0, 100):
            errors.append(f"{p}.locator 는 100자 이하 문자열이어야 한다")
        tc = ch.get("token_count")
        if tc is not None and (not _is_int(tc) or tc <= 0):
            errors.append(f"{p}.token_count 는 0보다 큰 정수여야 한다")
        payload = ch.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            errors.append(f"{p}.payload 가 객체가 아니다")
            continue
        _check_payload(f"{p}.payload", payload, stype, ext, tag_codes, allow_unknown_tags,
                       errors, warns, law_source_hashes)

    if stype == "LAW" and len(law_source_hashes) > 1:
        errors.append("같은 요청의 law_meta.source_sha256 값이 서로 다르다")
    return errors, warns, info


def _check_payload(p, payload, stype, ext, tag_codes, allow_unknown, errors, warns, law_hashes):
    for key in SERVER_FIELDS:
        if key in payload:
            warns.append(f"{p}.{key} 는 서버가 채워 덮어쓴다. 넣지 않는 것이 좋다")
    for key in ARRAY_TAGS:
        if key not in payload:
            continue
        val = payload[key]
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            errors.append(f"{p}.{key} 는 문자열 배열이어야 한다")
            continue
        for x in val:
            if key == "cas":
                if not CAS_RE.match(x):
                    errors.append(f"{p}.cas 값 {x!r} 이 CAS 번호 형식이 아니다")
            elif key in tag_codes and x not in tag_codes[key]:
                _unknown_tag(f"{p}.{key}", x, allow_unknown, errors, warns)
    if "severity" in payload:
        sev = payload["severity"]
        if not isinstance(sev, str):
            errors.append(f"{p}.severity 는 문자열 하나여야 한다(모르면 키를 뺀다)")
        elif sev not in tag_codes["severity"]:
            _unknown_tag(f"{p}.severity", sev, allow_unknown, errors, warns)
    if "year" in payload and not _is_int(payload["year"]):
        errors.append(f"{p}.year 는 정수여야 한다")

    if stype in ACCIDENT_TYPES:
        missing = [k for k in ARRAY_TAGS if k not in payload]
        if missing:
            warns.append(f"{p} 에 {', '.join(missing)} 가 없다. 값이 없으면 빈 배열로 넣는 것이 문서 형식이다")
    if stype == "SIF":
        for key in SIF_REQUIRED:
            if key not in payload:
                errors.append(f"{p}.{key} 가 없다(SIF 요청 필수)")
    elif stype == "LAW":
        for key in LAW_REQUIRED:
            if key not in payload:
                errors.append(f"{p}.{key} 가 없다(LAW 요청 필수)")
        nk = payload.get("natural_key")
        if isinstance(nk, dict) and nk.get("law_key") != ext:
            errors.append(f"{p}.natural_key.law_key 가 source.external_id 와 다르다")
        meta = payload.get("law_meta")
        if isinstance(meta, dict):
            sh = meta.get("source_sha256")
            if not (isinstance(sh, str) and SHA256_RE.match(sh)):
                errors.append(f"{p}.law_meta.source_sha256 은 64자리 소문자 hex 여야 한다")
            else:
                law_hashes.add(sh)
        elif "law_meta" in payload:
            errors.append(f"{p}.law_meta 가 객체가 아니다")
        if "article_label" not in payload:
            warns.append(f"{p}.article_label 이 없다(조문 정보)")
    elif stype in GUIDE_TYPES:
        missing = [k for k in ("hazard", "work_type", "cas", "year") if k not in payload]
        if missing:
            warns.append(f"{p} 에 {', '.join(missing)} 가 없다")


def _unknown_tag(where, value, allow_unknown, errors, warns):
    msg = f"{where} 값 {value!r} 이 tag_codes 스냅샷에 없다(영문 코드만 허용, 한글 라벨 금지)"
    (warns if allow_unknown else errors).append(msg)


# ---------------------------------------------------------------- HTTP

def http_call(method, url, body=None, headers=None, timeout=30):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise InputError(f"서버에 연결할 수 없다: {url} ({e.reason})") from e
    except TimeoutError as e:
        raise InputError(f"서버 응답 시간 초과: {url}") from e


def auth_header(openapi: dict | None, token: str):
    """openapi.json 의 securitySchemes 로 인증 헤더를 정한다. 못 정하면 Bearer."""
    schemes = ((openapi or {}).get("components") or {}).get("securitySchemes") or {}
    for sch in schemes.values():
        if sch.get("type") == "http" and str(sch.get("scheme", "")).lower() == "bearer":
            return "Authorization", f"Bearer {token}", True
        if sch.get("type") == "apiKey" and sch.get("in") == "header" and sch.get("name"):
            return sch["name"], token, True
    return "Authorization", f"Bearer {token}", False


def expected_point_id(source_type, external_id, chunk_index):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_type}:{external_id}:{chunk_index}"))


def check_response(req, data):
    """200 응답을 요청과 대조한다. (errors, warnings)."""
    errors, warns = [], []
    sent = req["chunks"]
    if not isinstance(data, dict):
        return ["응답이 객체가 아니다"], warns
    if data.get("indexed") != len(sent):
        errors.append(f"indexed={data.get('indexed')!r} 가 보낸 청크 수 {len(sent)} 와 다르다")
    if not _is_int(data.get("source_id")):
        errors.append("응답에 source_id 가 없다")
    got = data.get("chunks")
    if not isinstance(got, list):
        errors.append("응답에 chunks 가 없다")
        return errors, warns
    by_idx = {c.get("chunk_index"): c for c in got if isinstance(c, dict)}
    sent_idx = {c["chunk_index"] for c in sent}
    if set(by_idx) != sent_idx:
        errors.append("응답의 chunk_index 집합이 보낸 것과 다르다")
    stype, ext = req["source"]["source_type"], req["source"]["external_id"]
    for idx, c in by_idx.items():
        if not _is_int(c.get("chunk_id")) or not c.get("point_id"):
            errors.append(f"chunk_index {idx}: chunk_id 또는 point_id 가 없다")
        elif c["point_id"] != expected_point_id(stype, ext, idx):
            warns.append(f"chunk_index {idx}: point_id 가 예상 UUID v5 공식과 다르다")
    return errors, warns


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as e:
        raise InputError(f"파일이 없다: {path}") from e
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InputError(f"JSON을 읽을 수 없다: {path} ({e})") from e


def add_common_options(ap):
    ap.add_argument("--base-url", default="http://localhost:8001", help="kb-ingest 주소(기본 http://localhost:8001)")
    ap.add_argument("--send", action="store_true", help="실제로 전송한다. 없으면 검증만 한다")
    ap.add_argument("--allow-remote", action="store_true", help="localhost 가 아닌 서버로 전송을 허용한다")
    ap.add_argument("--keep-going", action="store_true", help="전송 실패 뒤에도 다음 원본을 계속 보낸다")
    ap.add_argument("--allow-unknown-tags", action="store_true", help="스냅샷에 없는 태그 코드를 경고로 낮춘다")
    ap.add_argument("--token-env", default="KB_INGEST_API_TOKEN", help="토큰을 담은 환경변수 이름")
    ap.add_argument("--timeout", type=int, default=600, help="요청 하나의 응답 대기 초(기본 600)")
    ap.add_argument("--report", type=Path, help="결과를 JSON Lines로 저장할 경로(토큰은 쓰지 않는다)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="{source, chunks} 요청 JSON을 검증하고 kb-ingest 권장 경로로 적재한다.")
    ap.add_argument("files", nargs="+", type=Path, help="요청 JSON 파일(파일 하나 = 원본 하나)")
    add_common_options(ap)
    return ap.parse_args(argv)


def say(msg=""):
    print(msg, flush=True)


def run(args) -> int:
    return run_items([(path.name, load_json(path)) for path in args.files], args)


def run_items(items, args) -> int:
    """items = [(이름, 요청 dict)]. 전부 검증한 뒤(--send 면) 순차 전송한다."""
    base = args.base_url.rstrip("/")
    # 1. 전부 검증. 하나라도 어기면 아무것도 보내지 않는다.
    loaded = []
    bad = 0
    for name, req in items:
        errors, warns, info = validate_request(req, allow_unknown_tags=args.allow_unknown_tags)
        label = f"{name} [{info['source_type']} {info['external_id']!r}, 청크 {info['chunks']}]"
        if errors:
            bad += 1
            say(f"FAIL  {label}")
            for m in errors:
                say(f"      - {m}")
        else:
            say(f"OK    {label} -> {info['endpoint']}")
        for m in warns:
            say(f"      ! {m}")
        loaded.append((name, req, info, errors))
    if bad:
        say(f"\n검증 실패 {bad}건. 아무것도 보내지 않았다.")
        return 2
    if not args.send:
        say("\n검증만 수행했다(전송 안 함). 실제로 보내려면 --send 를 준다.")
        return 0

    # 2. 전송 전 점검
    host = urllib.parse.urlparse(base).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
        raise InputError(f"대상이 로컬이 아니다({host}). 공용 서버에 올리려면 --allow-remote 를 준다")
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise InputError(f"환경변수 {args.token_env} 가 비어 있다. Read-Host 로 이 터미널에 설정한다")
    status, body = http_call("GET", f"{base}/ready", timeout=15)
    if status != 200:
        raise InputError(f"/ready 가 {status} 를 돌려줬다. kb-ingest 상태를 확인한다")
    status, body = http_call("GET", f"{base}/openapi.json", timeout=15)
    openapi = None
    if status == 200:
        try:
            openapi = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            openapi = None
    if openapi is not None:
        paths = openapi.get("paths") or {}
        for ep in sorted({i["endpoint"] for _, _, i, _ in loaded}):
            if ep not in paths:
                raise InputError(f"배포된 kb-ingest 에 {ep} 경로가 없다. 이미지를 다시 빌드했는지 확인한다")
    hname, hvalue, detected = auth_header(openapi, token)
    if not detected:
        say("! openapi.json 에서 인증 방식을 못 찾아 Authorization: Bearer 를 쓴다")

    # 3. 전송. 파일 하나 = 요청 하나, 순차.
    results = []
    failed = 0
    for name, req, info, _ in loaded:
        url = f"{base}{info['endpoint']}"
        payload = json.dumps(req, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", hname: hvalue}
        status, body = http_call("POST", url, body=payload, headers=headers, timeout=args.timeout)
        rec = {"file": name, "source_type": info["source_type"], "external_id": info["external_id"],
               "endpoint": info["endpoint"], "http_status": status, "sent_chunks": info["chunks"]}
        ok = False
        if status == 200:
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                data = None
            errs, warns = check_response(req, data)
            if isinstance(data, dict):
                rec.update(indexed=data.get("indexed"), source_id=data.get("source_id"))
            for m in warns:
                say(f"      ! {m}")
            if errs:
                for m in errs:
                    say(f"      - {m}")
            else:
                ok = True
        else:
            text = body.decode("utf-8", errors="replace")[:800]
            hint = HINTS.get(status, "")
            say(f"      HTTP {status}: {text}")
            if hint:
                say(f"      확인: {hint}")
        rec["ok"] = ok
        results.append(rec)
        if ok:
            say(f"SENT  {name}: indexed={rec.get('indexed')}, source_id={rec.get('source_id')}")
        else:
            failed += 1
            say(f"FAIL  {name}: 전송 또는 응답 검증 실패")
            if not args.keep_going:
                break
    if args.report:
        with args.report.open("w", encoding="utf-8", newline="\n") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    say(f"\n전송 {len(results)}건 중 성공 {len(results) - failed}, 실패 {failed}")
    if failed:
        say("같은 입력을 다시 보낼 때 장부가 어떻게 반응하는지는 확인되지 않았다. 먼저 장부와 Qdrant 상태를 본다.")
    return 1 if failed else 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        return run(args)
    except InputError as e:
        say(f"입력·설정 오류: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
