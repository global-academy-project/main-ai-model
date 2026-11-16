#!/usr/bin/env python3
"""청크 JSONL 파일을 인자로 받아 적재한다. API 요청 형식을 직접 만들 필요가 없다.

사용자는 청크 파일 경로만 넘긴다. 스크립트가 같은 원본(source_type + external_id)의 줄을 묶어
{source, chunks} 요청으로 바꾸고, 검증한 뒤 kb-ingest 권장 경로로 보낸다
(검증·전송은 scripts/ingest_via_api.py 를 그대로 쓴다).

청크 파일 형식: 한 줄 = 청크 하나(JSON 객체).
  원본 정보  source_type, external_id, title, url 또는 file_path, publisher, license, content_sha256, flagged
  청크 정보  chunk_index, text, embedding_text, locator, token_count, payload
같은 원본의 줄마다 원본 정보를 반복해도 되고 첫 줄에만 적어도 된다(값이 다르면 오류).
payload 에는 API 문서와 같은 분류·유형별 필드(severity, accident_type, natural_key 등)를 넣는다.

기본은 검증만 한다. 실제로 보내려면 --send 를 준다. 종료 코드: 0 성공, 1 전송 실패, 2 입력·설정 오류.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ingest_via_api as core

SOURCE_KEYS = ("source_type", "external_id", "title", "url", "file_path",
               "publisher", "license", "content_sha256", "flagged")
CHUNK_KEYS = ("chunk_index", "text", "embedding_text", "locator", "token_count", "payload")
KNOWN_KEYS = set(SOURCE_KEYS) | set(CHUNK_KEYS)
# 청킹 산출물과 API 의 이름이 다른 것. 조용히 버리지 않고 알려 준다.
KEY_HINTS = {"embed_text": "embedding_text", "id": "external_id", "doc_id": "external_id"}


def read_lines(path: Path, errors: list[str]):
    """(줄 번호, 객체) 목록. 읽을 수 없는 줄은 errors 에 쌓는다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        errors.append(f"{path}: 파일이 없다")
        return []
    except (OSError, UnicodeDecodeError) as e:
        errors.append(f"{path}: 읽을 수 없다 ({e})")
        return []
    rows = []
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}:{n}: JSON 이 아니다 ({e.msg})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"{path.name}:{n}: 줄이 객체가 아니다")
            continue
        rows.append((n, obj))
    return rows


def build_requests(paths, defaults: dict, only=None):
    """청크 파일들 -> ([(이름, 요청)], errors). 원본은 첫 등장 순서, 청크는 파일 순서."""
    errors: list[str] = []
    groups: dict[tuple, dict] = {}
    for path in paths:
        for n, row in read_lines(path, errors):
            where = f"{path.name}:{n}"
            bad = False
            for key in row:
                if key not in KNOWN_KEYS:
                    hint = f" ({KEY_HINTS[key]} 인가?)" if key in KEY_HINTS else ""
                    errors.append(f"{where}: 알 수 없는 키 {key!r}{hint}")
                    bad = True
            merged = {k: v for k, v in defaults.items() if v is not None}
            merged.update(row)
            for key in ("source_type", "external_id"):
                if not isinstance(merged.get(key), str) or not merged[key].strip():
                    errors.append(f"{where}: {key} 가 없다")
                    bad = True
            if bad:
                continue
            gkey = (merged["source_type"], merged["external_id"])
            g = groups.setdefault(gkey, {"file": path.name, "source": {}, "chunks": []})
            for key in SOURCE_KEYS:
                if key not in merged:
                    continue
                if key in g["source"] and g["source"][key] != merged[key]:
                    errors.append(f"{where}: 같은 원본 {gkey[1]!r} 의 {key} 값이 앞 줄과 다르다")
                g["source"].setdefault(key, merged[key])
            chunk = {k: merged[k] for k in CHUNK_KEYS if k in merged}
            chunk["_where"] = where
            g["chunks"].append(chunk)

    items = []
    for (stype, ext), g in groups.items():
        if only and ext not in only:
            continue
        chunks = []
        for ch in g["chunks"]:
            ch = dict(ch)
            ch.pop("_where")
            chunks.append(ch)
        items.append((f"{g['file']}:{ext}", {"source": g["source"], "chunks": chunks}))
    if only:
        missing = sorted(set(only) - {k[1] for k in groups})
        if missing:
            errors.append(f"--only 로 지정한 external_id 가 파일에 없다: {', '.join(missing)}")
    return items, errors


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="청크 JSONL 파일을 인자로 받아 kb-ingest 권장 경로로 적재한다.")
    ap.add_argument("chunk_files", nargs="+", type=Path, help="청크 JSONL 파일(한 줄 = 청크 하나)")
    ap.add_argument("--source-type", help="줄에 source_type 이 없을 때 쓸 기본값")
    ap.add_argument("--publisher", help="줄에 publisher 가 없을 때 쓸 기본값")
    ap.add_argument("--license", help="줄에 license 가 없을 때 쓸 기본값")
    ap.add_argument("--url", help="줄에 url 이 없을 때 쓸 기본값")
    ap.add_argument("--only", action="append", metavar="EXTERNAL_ID",
                    help="이 external_id 의 원본만 처리한다(여러 번 줄 수 있다)")
    core.add_common_options(ap)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    defaults = {"source_type": args.source_type, "publisher": args.publisher,
                "license": args.license, "url": args.url}
    items, errors = build_requests(args.chunk_files, defaults, set(args.only or []))
    if errors:
        for m in errors:
            core.say(f"FAIL  {m}")
        core.say(f"\n입력 오류 {len(errors)}건. 아무것도 보내지 않았다.")
        return 2
    if not items:
        core.say("처리할 원본이 없다. 아무것도 보내지 않았다.")
        return 2
    core.say(f"원본 {len(items)}개, 청크 {sum(len(r['chunks']) for _, r in items)}개를 읽었다.")
    try:
        return core.run_items(items, args)
    except core.InputError as e:
        core.say(f"입력·설정 오류: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
