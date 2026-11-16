"""공단 MSDS API 실측 (읽기 전용). lookup_msds 실제 도구를 등록하기 전에 R3·R5·R6을 숫자로 확인한다.

무엇을 보나
  1. 연결 확인 (--ping-only 면 여기까지): 키 형식, 외부망, 공단 응답 코드.
  2. 에이전트 물질 사전(app/agent/text_rules.MATERIALS)의 표준 이름과 별칭을 하나씩 조회해
     찾았는지, 못 찾았으면 왜인지(후보 없음 / 이름 불일치 / 같은 이름에 CAS 여러 개).
  3. 호출 시간(물질당, API 호출당), 물질당 호출 횟수, 결과 크기, ghs_classes 모양.

쓰지 않는 것: 업무 DB, Spring, LLM, 에이전트 코드. 서버 설정도 읽지 않는다.
키는 환경변수 KOSHA_MSDS_SERVICE_KEY 에서만 읽고, 화면·파일에 쓰지 않는다 (URL도 출력하지 않는다).

사용 (저장소 루트에서)
    $env:KOSHA_MSDS_SERVICE_KEY = Read-Host "KOSHA_MSDS_SERVICE_KEY"
    python -B -m scripts.probe_msds --ping-only
    python -B -m scripts.probe_msds

옵션: --only-canonical (표준 이름만) --no-dictionary (--names만) --names 이름,이름 (추가) --limit N --sleep 초 --timeout 초 --out-dir 폴더
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from app.agent.text_rules import MATERIALS
from app.exceptions import ToolError
from app.publicdata import msds as msds_module
from app.publicdata.msds import KoshaMsdsClient

KEY_ENV = "KOSHA_MSDS_SERVICE_KEY"
DEFAULT_OUT_DIR = Path("out") / "msds_probe"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
OK_CODES = {"", "00", "0000"}
CANDIDATES_SHOWN = 5
SAMPLE_TEXT_CHARS = 120
WORST_CASE_MATERIALS = 6  # plan_brief.MAX_BRIEF_ITEMS: 계획이 넘기는 물질 후보 수(사용자가 말한 물질은 따로 더해진다)
# plan_brief.HAZARD_TERM_MARKERS와 같다 (임포트가 되면 그 값을 쓴다)
_FALLBACK_MARKERS = ("인화성", "급성 독성", "고압가스", "부식성", "발암성", "산화성")


def _hazard_markers() -> tuple[str, ...]:
    try:
        from app.agent.plan_brief import HAZARD_TERM_MARKERS

        return tuple(HAZARD_TERM_MARKERS)
    except Exception:  # noqa: BLE001 - 실측 스크립트: 임포트가 안 되면 같은 값을 직접 쓴다
        return _FALLBACK_MARKERS


# ── 조회 대상 ──


def build_names(*, only_canonical: bool, extra: list[str], limit: int | None, skip_dictionary: bool = False) -> list[tuple[str, str]]:
    """(조회어, 종류). 종류: 표준 · 별칭 · 추가. 같은 조회어(대소문자 무시)는 한 번만. skip_dictionary면 extra만."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, kind: str) -> None:
        key = name.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            rows.append((name.strip(), kind))

    if not skip_dictionary:
        for canonical in MATERIALS:
            add(canonical, "표준")
        if not only_canonical:
            for canonical, aliases in MATERIALS.items():
                for alias in aliases:
                    add(alias, "별칭")
    for name in extra:
        add(name, "추가")
    return rows[:limit] if limit is not None else rows


# ── 호출 기록용 클라이언트 (A의 코드는 고치지 않고 상속만 한다) ──


class ProbeClient(KoshaMsdsClient):
    """_request_items를 감싸 API 호출마다 시간·건수·글자 수를 남긴다. 조회 동작은 그대로다."""

    def __init__(self, service_key: str, *, timeout_seconds: float) -> None:
        super().__init__(service_key, timeout_seconds=timeout_seconds)
        self.calls: list[dict[str, object]] = []

    def _request_items(self, operation, parameters):  # type: ignore[override]
        started = time.perf_counter()
        try:
            items = super()._request_items(operation, parameters)
        except ToolError:
            self.calls.append({"operation": operation, "seconds": time.perf_counter() - started, "ok": False})
            raise
        chars = sum(len(value) for item in items for value in item.values())
        self.calls.append(
            {"operation": operation, "seconds": time.perf_counter() - started, "ok": True, "items": len(items), "chars": chars}
        )
        return items


def _norm(value: str) -> str:
    return "".join(value.split()).casefold()


def explain_miss(client: ProbeClient, query: str) -> dict[str, object]:
    """못 찾은 이유를 목록 API 후보로 설명한다. 호출 기록은 probe_one이 이 함수 전에 확정하므로 집계에 들어가지 않는다."""
    is_cas = msds_module._CAS_PATTERN.fullmatch(query.strip()) is not None
    try:
        candidates = client._request_items(
            "getChemList001",
            {"searchWrd": query.strip(), "searchCnd": "1" if is_cas else "0", "numOfRows": "100", "pageNo": "1"},
        )
    except ToolError as exc:
        return {"reason": f"후보 조회 실패({exc.code})", "candidates": []}
    shown = [{"name": c.get("chemNameKor", ""), "cas": c.get("casNo", "")} for c in candidates[:CANDIDATES_SHOWN]]
    if not candidates:
        return {"reason": "후보 0건", "candidates": shown, "candidate_count": 0}
    key = _norm(query)
    same_name = [c for c in candidates if (c.get("casNo", "").strip() == query.strip() if is_cas else _norm(c.get("chemNameKor", "")) == key)]
    if not same_name:
        return {"reason": f"이름 불일치(후보 {len(candidates)}건)", "candidates": shown, "candidate_count": len(candidates)}
    cas_numbers = sorted({c.get("casNo", "").strip() for c in same_name})
    if len(cas_numbers) > 1:
        return {"reason": f"같은 이름에 CAS {len(cas_numbers)}개", "candidates": shown, "candidate_count": len(candidates)}
    return {"reason": "일치 항목은 있으나 chemId 없음", "candidates": shown, "candidate_count": len(candidates)}


def probe_one(client: ProbeClient, query: str, kind: str, markers: tuple[str, ...]) -> dict[str, object]:
    client.calls.clear()
    started = time.perf_counter()
    result: dict[str, object] | None = None
    error: str | None = None
    try:
        result = client.lookup({"materials": [query]})["results"][0]  # type: ignore[index]
    except ToolError as exc:
        error = exc.code
    seconds = time.perf_counter() - started
    calls = list(client.calls)
    row: dict[str, object] = {
        "query": query,
        "kind": kind,
        "seconds": round(seconds, 3),
        "api_calls": len(calls),
        "call_seconds": [round(float(c["seconds"]), 3) for c in calls],
        "error": error,
        "found": bool(result and result.get("found")),
        "cached": bool(not calls and result is not None),
    }
    if result and result.get("found"):
        ghs = [str(g) for g in result.get("ghs_classes", [])]  # type: ignore[union-attr]
        terms = [g for g in ghs if any(m in g for m in markers)]
        row.update(
            name=result.get("name"),
            cas_no=result.get("cas_no"),
            name_equals_query=str(result.get("name") or "") == query.strip(),  # 글자 그대로 비교 (공백·대소문자 차이도 R4: 허가서 이름 매칭은 정확 일치)
            ghs_count=len(ghs),
            ghs_max_chars=max((len(g) for g in ghs), default=0),
            ghs_has_flammable=any("인화성" in g for g in ghs),
            query_terms_max_chars=max((len(t) for t in terms), default=0),
            ghs_classes=[g[:SAMPLE_TEXT_CHARS] for g in ghs[:5]],
            hazards_count=len(result.get("hazards", [])),  # type: ignore[arg-type]
            regulations_count=len(result.get("regulations", [])),  # type: ignore[arg-type]
            regulatory_tag_codes=[t.get("code") for t in result.get("regulatory_tags", [])],  # type: ignore[union-attr]
            first_aid_chars=len(str(result.get("first_aid") or "")),
            result_bytes=len(json.dumps(result, ensure_ascii=False).encode("utf-8")),
        )
    elif error is None:
        row.update(explain_miss(client, query))
    return row


# ── 연결 확인 ──


def _find(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == name and element.text:
            return element.text.strip()
    return ""


def ping(key: str, timeout: float) -> bool:
    """목록 API를 한 번 부른다. 실패 원인은 종류만 출력한다 (URL·키는 출력하지 않는다)."""
    if "%" in key:
        print("주의: 키에 '%'가 있습니다. 포털의 'Encoding 키'일 수 있습니다. 코드가 다시 인코딩하므로 'Decoding 키'를 쓰세요(추정).")
    query = urlencode({"serviceKey": key, "searchWrd": "67-64-1", "searchCnd": "1", "numOfRows": "1", "pageNo": "1"})
    request = Request(f"{msds_module._BASE_URL}/getChemList001?{query}", headers={"Accept": "application/xml"})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        print(f"연결 확인 실패: HTTP {exc.code} ({time.perf_counter() - started:.1f}초)")
        return False
    except URLError as exc:
        print(f"연결 확인 실패: {type(exc.reason).__name__} ({time.perf_counter() - started:.1f}초) - 외부망·프록시·DNS 확인")
        return False
    except (TimeoutError, OSError) as exc:
        print(f"연결 확인 실패: {type(exc).__name__} ({time.perf_counter() - started:.1f}초)")
        return False
    elapsed = time.perf_counter() - started
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        print(f"연결 확인 실패: XML이 아닌 응답 ({len(body)}바이트, {elapsed:.1f}초)")
        return False
    code, message = _find(root, "resultCode"), _find(root, "resultMsg")
    items = sum(1 for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "item")
    ok = code in OK_CODES
    print(f"연결 확인 {'성공' if ok else '실패'}: resultCode={code or '-'} resultMsg={message or '-'} 항목 {items}건 ({elapsed:.1f}초)")
    if ok and items == 0:
        print("주의: 아세톤(67-64-1)이 0건입니다. 키는 통했지만 서비스 범위를 확인하세요.")
    return ok


# ── 요약 ──


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)] if ordered else 0.0


def _stats(values: list[float]) -> str:
    if not values:
        return "자료 없음"
    return f"중앙값 {statistics.median(values):.2f}초 / p90 {_p90(values):.2f}초 / 최대 {max(values):.2f}초"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    live = [r for r in rows if not r["cached"] and r["error"] is None]
    seconds = [float(r["seconds"]) for r in live]  # type: ignore[arg-type]
    call_seconds = [float(s) for r in live for s in r["call_seconds"]]  # type: ignore[union-attr]
    found = [r for r in rows if r["found"]]
    by_kind = {}
    for kind in ("표준", "별칭", "추가"):
        group = [r for r in rows if r["kind"] == kind]
        if group:
            by_kind[kind] = {"total": len(group), "found": sum(1 for r in group if r["found"])}
    median_seconds = statistics.median(seconds) if seconds else 0.0
    return {
        "by_kind": by_kind,
        "errors": {str(r["query"]): r["error"] for r in rows if r["error"]},
        "lookup_seconds": {"median": median_seconds, "p90": _p90(seconds), "max": max(seconds, default=0.0)},
        "call_seconds": {"median": statistics.median(call_seconds) if call_seconds else 0.0, "max": max(call_seconds, default=0.0)},
        "api_calls_per_lookup": statistics.median([int(r["api_calls"]) for r in live]) if live else 0,  # type: ignore[arg-type]
        "worst_case_estimate_seconds": {
            "median": round(median_seconds * WORST_CASE_MATERIALS, 1),
            "max": round(max(seconds, default=0.0) * WORST_CASE_MATERIALS, 1),
        },
        "result_bytes": {
            "median": int(statistics.median([int(r["result_bytes"]) for r in found])) if found else 0,  # type: ignore[arg-type]
            "max": max((int(r["result_bytes"]) for r in found), default=0),  # type: ignore[arg-type]
        },
        "name_differs_from_query": [r["query"] for r in found if not r.get("name_equals_query")],
    }


def print_report(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    print()
    print("── 물질별 결과 ──")
    for r in rows:
        head = f"[{r['kind']}] {r['query']}"
        if r["error"]:
            print(f"{head}: 오류 {r['error']} ({r['seconds']}초)")
        elif r["found"]:
            flag = "" if r.get("name_equals_query") else f" ← 공단 이름 '{r['name']}'"
            print(
                f"{head}: 찾음 CAS {r['cas_no']}{flag} | 호출 {r['api_calls']}회 {r['seconds']}초 | "
                f"ghs {r['ghs_count']}개(최대 {r['ghs_max_chars']}자, 인화성 {'O' if r['ghs_has_flammable'] else 'X'}) | "
                f"규제 {r['regulations_count']}줄 | 응급조치 {r['first_aid_chars']}자 | {r['result_bytes']}바이트"
            )
        else:
            names = ", ".join(f"{c['name']}({c['cas']})" for c in r.get("candidates", [])[:3])  # type: ignore[union-attr]
            print(f"{head}: 못 찾음 - {r.get('reason')}" + (f" | 후보: {names}" if names else ""))

    print()
    print("── 요약 ──")
    for kind, g in summary["by_kind"].items():  # type: ignore[union-attr]
        print(f"찾음 {kind}: {g['found']}/{g['total']}")
    ls, cs = summary["lookup_seconds"], summary["call_seconds"]
    print(f"물질당 조회 시간(캐시·오류 제외): 중앙값 {ls['median']:.2f}초 / p90 {ls['p90']:.2f}초 / 최대 {ls['max']:.2f}초")  # type: ignore[index]
    print(f"API 호출당 시간: 중앙값 {cs['median']:.2f}초 / 최대 {cs['max']:.2f}초, 물질당 호출 {summary['api_calls_per_lookup']}회")  # type: ignore[index]
    est = summary["worst_case_estimate_seconds"]
    print(f"물질 {WORST_CASE_MATERIALS}개 순차 조회 추정: 중앙값 기준 {est['median']}초 / 최대값 기준 {est['max']}초 (Spring 제한 120초, 계획 LLM 22~35초 별도)")  # type: ignore[index]
    rb = summary["result_bytes"]
    print(f"찾은 결과 크기: 중앙값 {rb['median']}바이트 / 최대 {rb['max']}바이트 (물질 {WORST_CASE_MATERIALS}개면 최대 약 {rb['max'] * WORST_CASE_MATERIALS}바이트)")  # type: ignore[index]
    if summary["name_differs_from_query"]:
        print(f"공단 이름이 조회어와 다른 것(허가서 이름 매칭 R4): {', '.join(map(str, summary['name_differs_from_query']))}")  # type: ignore[arg-type]
    if summary["errors"]:
        print(f"오류가 난 조회: {summary['errors']}")
    long_terms = [r["query"] for r in rows if r.get("query_terms_max_chars", 0) and int(r["query_terms_max_chars"]) > 40]  # type: ignore[arg-type]
    if long_terms:
        print(f"사고사례 질의에 붙는 분류 문구가 40자를 넘는 물질(R5): {', '.join(map(str, long_terms))}")


# ── 실행 ──


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="공단 MSDS API 실측 (읽기 전용)")
    p.add_argument("--ping-only", action="store_true", help="연결 확인만 하고 끝낸다")
    p.add_argument("--only-canonical", action="store_true", help="사전의 표준 이름만 조회한다 (별칭 제외)")
    p.add_argument("--no-dictionary", action="store_true", help="사전은 건너뛰고 --names만 조회한다")
    p.add_argument("--names", default="", help="추가로 조회할 이름·CAS (쉼표로 구분)")
    p.add_argument("--limit", type=int, default=None, help="조회 개수 상한")
    p.add_argument("--sleep", type=float, default=0.2, help="물질 사이 대기(초)")
    p.add_argument("--timeout", type=float, default=10.0, help="API 호출 제한 시간(초). A의 기본값과 같다")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--base-url", default=None, help="시험 전용: 127.0.0.1·localhost 주소만 허용")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.base_url:
        host = urlparse(args.base_url).hostname
        if host not in LOCAL_HOSTS:
            print("--base-url은 127.0.0.1·localhost만 쓸 수 있습니다 (키가 다른 곳으로 나가는 것을 막습니다).")
            return 2
        msds_module._BASE_URL = args.base_url.rstrip("/")
    key = os.getenv(KEY_ENV, "").strip()
    if not key:
        print(f"환경변수 {KEY_ENV}가 필요합니다. 예: $env:{KEY_ENV} = Read-Host \"{KEY_ENV}\"")
        return 2
    if args.timeout <= 0 or args.sleep < 0 or (args.limit is not None and args.limit <= 0):
        print("--timeout·--limit은 양수, --sleep은 0 이상이어야 합니다.")
        return 2

    if not ping(key, args.timeout):
        return 1
    if args.ping_only:
        return 0

    extra = [n for n in (s.strip() for s in args.names.split(",")) if n]
    names = build_names(only_canonical=args.only_canonical, extra=extra, limit=args.limit, skip_dictionary=args.no_dictionary)
    if not names:
        print("조회할 물질이 없습니다 (--no-dictionary는 --names와 함께 쓰세요).")
        return 2
    client = ProbeClient(key, timeout_seconds=args.timeout)
    markers = _hazard_markers()
    print(f"조회 {len(names)}건 시작 (물질마다 API 4번 안팎, 간격 {args.sleep}초)")
    rows: list[dict[str, object]] = []
    for number, (query, kind) in enumerate(names, 1):
        row = probe_one(client, query, kind, markers)
        rows.append(row)
        state = "오류 " + str(row["error"]) if row["error"] else ("찾음" if row["found"] else "못 찾음")
        print(f"  {number}/{len(names)} {query}: {state} ({row['seconds']}초)", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    summary = summarize(rows)
    print_report(rows, summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n자세한 결과: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
