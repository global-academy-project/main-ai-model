"""기존 감사 대응 시험 3개 파일의 수정 — 올려 주신 구성기 라우터(`/audit-report`만 있고 `/compose` 별칭 없음)에 맞춤.

  [tests/test_audit_compose.py]
  1) 라우트 시험 2개가 `/api/v1/agent/audit-report/compose`(없어진 별칭)를 부르던 것을 `/api/v1/agent/audit-report`로
  [tests/test_audit_rules_switch.py]
  2) 구성기가 맡은 주소를 확인하는 시험에서 별칭 주소를 뺌
  [tests/test_audit_route.py]
  3) `AUDIT_ROUTE_ENABLED`를 안 줬을 때(None) 꺼져 있다고 못 박던 시험을 뺌 (작업본은 코드 기본값을 켜 둔 상태).
     대신 "환경변수가 없으면 Settings 기본값을 따른다"는 시험을 추가 (기본값을 바꿔도 시험이 따라감)

서비스 분리(apply_audit_compose_service.py)와는 독립이다. 어느 쪽을 먼저 적용해도 된다.

사용 (옵션 없이 실행하면 app 폴더를 스스로 찾는다):
  python apply_audit_tests_fix.py --check
  python apply_audit_tests_fix.py
  python apply_audit_tests_fix.py --revert
  (필요하면 --root <app 폴더가 들어 있는 폴더>)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROUTER_TEST = "tests/test_audit_compose.py"
SWITCH_TEST = "tests/test_audit_rules_switch.py"
ROUTE_TEST = "tests/test_audit_route.py"
MARKER = ROUTER_TEST

TARGETS = {
    ROUTER_TEST: [
        (
            "라우트 시험 1: 별칭 주소 -> 정식 주소",
            '    resp = client.post("/api/v1/agent/audit-report/compose", json=load("normal"))\n',
            '    resp = client.post("/api/v1/agent/audit-report", json=load("normal"))\n',
        ),
        (
            "라우트 시험 2: 별칭 주소 -> 정식 주소",
            '    resp = client.post("/api/v1/agent/audit-report/compose", json={"reportId": 1})\n',
            '    resp = client.post("/api/v1/agent/audit-report", json={"reportId": 1})\n',
        ),
    ],
    SWITCH_TEST: [
        (
            "별칭 주소 뺌",
            '    for path in ("/api/v1/agent/audit-report", "/api/v1/agent/audit-report/compose"):  # 뒤의 것은 호환 별칭\n',
            '    for path in ("/api/v1/agent/audit-report",):  # `/compose` 호환 별칭은 없어짐\n',
        ),
    ],
    ROUTE_TEST: [
        (
            "기본값 가정 시험 정리",
            '@pytest.mark.parametrize("route", [None, False])\n'
            "def test_switch_off_keeps_old_behaviour(monkeypatch, tmp_path, route):\n",
            '@pytest.mark.parametrize("route", [False])\n'
            "def test_switch_off_keeps_old_behaviour(monkeypatch, tmp_path, route):\n",
        ),
        (
            "기본값을 따르는 시험 추가",
            '    assert "전용 화면" in data["ask"]["message"]\n'
            '    assert data["slots"]["pendingText"] is None',
            '    assert "전용 화면" in data["ask"]["message"]\n'
            '    assert data["slots"]["pendingText"] is None\n'
            "\n"
            "\n"
            "def test_default_follows_the_setting_default(monkeypatch, tmp_path):\n"
            '    """환경변수가 없으면 Settings 기본값을 따른다 (코드 기본값을 바꿔도 시험이 따라간다)."""\n'
            "    from app.core import Settings\n"
            "\n"
            '    default_on = Settings.model_fields["audit_route_enabled"].default\n'
            '    data = _post(_client(monkeypatch, tmp_path, route=None), f"{PERIOD} 감사 보고서 만들어줘")\n'
            "    if default_on:\n"
            '        assert data["runStatus"] == "DONE" and data["route"] is not None\n'
            "    else:\n"
            '        assert data["runStatus"] == "WAITING_INPUT" and data["route"] is None',
        ),
    ],
}

def _find_root(explicit: str | None) -> tuple[Path | None, str]:
    """(app 폴더가 들어 있는 폴더, 찾은 방법). 현재 폴더 쪽을 먼저 보고, 마지막에 스크립트가 있는 폴더를 본다."""
    if explicit:
        return Path(explicit), "--root"
    here = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    candidates: list[tuple[Path, str]] = [(here, "현재 폴더")]
    try:
        candidates += [(d, "현재 폴더의 하위") for d in sorted(here.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        pass
    candidates += [(p, "현재 폴더의 상위") for p in list(here.parents)[:3]]
    candidates.append((script_dir, "스크립트가 있는 폴더"))
    seen: set[Path] = set()
    for c, how in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / MARKER).is_file():
            return c, how
    return None, ""


LABEL = {"BEFORE": "적용 전", "AFTER": "이미 적용됨", "MISMATCH": "안 맞음"}


def _edit_state(text: str, old: str, new: str) -> str:
    # 새 문구가 옛 문구를 포함할 수 있어 새 문구를 뺀 뒤에 옛 문구를 센다.
    if text.count(new) == 1 and text.replace(new, "").count(old) == 0:
        return "AFTER"
    if text.count(new) == 0 and text.count(old) == 1:
        return "BEFORE"
    return "MISMATCH"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="app 폴더가 들어 있는 폴더(생략하면 자동으로 찾음)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    root, how = _find_root(a.root)
    if root is None:
        print(f"{MARKER}를 찾지 못함. ai-model 저장소 폴더에서 실행하거나 --root <저장소 폴더>를 주세요.")
        return 2
    print(f"대상 폴더: {root.resolve()} ({how})")
    if how == "스크립트가 있는 폴더" and Path.cwd().resolve() != root.resolve():
        print("※ 현재 폴더 근처에서 app을 못 찾아 스크립트가 있는 폴더의 app을 대상으로 함. 저장소를 고치려면 저장소 폴더에서 실행하세요.")
    plan: list[tuple[Path, str, str]] = []  # (경로, 새 내용, 줄바꿈)
    bad = False
    for rel, edits in TARGETS.items():
        path = root / rel
        if not path.is_file():
            print(f"대상 파일이 없음: {path}")
            return 2
        raw = path.read_bytes()
        nl = "\r\n" if b"\r\n" in raw else "\n"
        text = raw.decode("utf-8")
        changed = False
        for name, o, n in edits:
            o, n = o.replace("\n", nl), n.replace("\n", nl)
            st = _edit_state(text, o, n)
            print(f"{rel} · {name}: {LABEL[st]}")
            if st == "MISMATCH":
                bad = True
                continue
            if a.revert and st == "AFTER":
                text, changed = text.replace(n, o), True
            elif not a.revert and st == "BEFORE":
                text, changed = text.replace(o, n), True
        if changed:
            plan.append((path, text, nl))

    if bad:
        print("맞지 않는 곳이 있어 아무것도 수정하지 않음 (시험 파일이 작업본과 다름 — 해당 파일을 보내 주세요)")
        return 1
    if a.check:
        return 0
    if not plan:
        print("변경할 것이 없음(이미 요청한 상태)")
        return 0
    for path, text, nl in plan:
        path.write_bytes(text.encode("utf-8"))  # 줄바꿈은 위에서 맞춰 둠(newline 변환 없음)
        print(f"{path}: {'되돌림' if a.revert else '적용'} 완료 (줄바꿈 {'CRLF' if nl == chr(13) + chr(10) else 'LF'} 유지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
