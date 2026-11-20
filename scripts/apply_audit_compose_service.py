"""감사 대응 구성기 라우터의 업무 로직을 서비스로 분리 — 기존 파일 1곳의 수정 (새 파일은 zip에 들어 있음).

기준: 2026-11-20에 올린 `app/api/v1/agent/audit_compose.py` (`/audit-report`만 맡고 `/compose` 별칭은 없는 판,
      모델을 못 얻으면 템플릿 문장으로 계속하는 try/except가 이미 들어 있는 판).

  [app/api/v1/agent/audit_compose.py]
  1) import 정리: logging·compose_audit_report·generate_sentence_candidates·logger를 빼고 서비스 import 한 줄
  2) 함수 본문: 문장 모델 선택·실패 처리를 app/services/audit_compose_service.py의 compose_report 호출로 바꿈

동작은 그대로다 (문장 LLM 꺼짐이 기본, 켜도 모델을 못 얻으면 템플릿 문장, 모델은 꺼져 있으면 얻지 않음).
새 파일(app/services/audit_compose_service.py, tests/test_audit_compose_service.py)은 zip의 app/, tests/ 아래에 그대로 풀어 넣는다.

사용 (옵션 없이 실행하면 app 폴더를 스스로 찾는다):
  python apply_audit_compose_service.py --check
  python apply_audit_compose_service.py
  python apply_audit_compose_service.py --revert     # 라우터만 되돌림. 새 파일은 README의 되돌리기 절을 본다.
  (필요하면 --root <app 폴더가 들어 있는 폴더>)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROUTER = "app/api/v1/agent/audit_compose.py"
MARKER = ROUTER

TARGETS = {
    ROUTER: [
        (
            "import 정리",
            "import logging\n"
            "\n"
            "from fastapi import APIRouter, Request\n"
            "\n"
            "from app.agent.audit_compose import compose_audit_report\n"
            "from app.agent.audit_narrative import generate_sentence_candidates\n"
            "from app.core import get_settings\n"
            "from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult\n"
            "from app.schemas.base_schemas import DataResponse\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n",
            "from fastapi import APIRouter, Request\n"
            "\n"
            "from app.core import get_settings\n"
            "from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult\n"
            "from app.schemas.base_schemas import DataResponse\n"
            "from app.services.audit_compose_service import AUDIT_NARRATIVE_SCENARIO, compose_report\n"
            "\n",
        ),
        (
            "함수 본문을 서비스 호출로",
            '    """서버 집계 항목을 구성한다. P2 LLM 문장은 설정이 켜진 경우에만 사용한다."""\n'
            "    settings = get_settings()\n"
            "    generator = None\n"
            "    if settings.audit_narrative_llm:\n"
            "        # 시나리오 E(audit_query)의 D 소유 모델을 사용한다. 기존 summary 엔진의\n"
            "        # 전용 메시지 후처리가 새 sentences JSON 형식에 끼어들지 않게 한다.\n"
            "        try:\n"
            '            llm = request.app.state.llm_provider.get("audit_query")\n'
            "            generator = lambda item: generate_sentence_candidates(item, llm)\n"
            "        except Exception as exc:  # 모델 설정 오류가 보고서 전체를 5xx로 만들지 않게 한다\n"
            '            logger.warning("감사 보고서 문장 모델을 못 얻어 템플릿 문장을 사용함 (%s)", type(exc).__name__)\n'
            "            generator = None\n"
            "    return DataResponse(data=compose_audit_report(body, narrative_generator=generator))",
            '    """서버 집계 항목을 구성한다. P2 LLM 문장은 설정이 켜진 경우에만 사용한다 (선택·실패 처리는 서비스)."""\n'
            "    return compose_report(\n"
            "        body,\n"
            "        narrative_llm_enabled=get_settings().audit_narrative_llm,\n"
            "        # 꺼져 있으면 호출되지 않는다 (모델 제공자가 없는 앱에서도 보고서는 만들어진다).\n"
            "        get_llm=lambda: request.app.state.llm_provider.get(AUDIT_NARRATIVE_SCENARIO),\n"
            "    )",
        ),
    ],
}

NEW_FILES = ["app/services/audit_compose_service.py"]


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
    for rel in NEW_FILES:
        exists = (root / rel).is_file()
        print(f"{rel}: {'있음' if exists else '없음 (zip의 app/ 아래 파일을 풀어 넣어야 함)'}")
        if not exists and not a.revert and not a.check:
            print("새 파일이 없으면 라우터가 import에 실패하므로 적용하지 않음")
            return 2
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
        print("맞지 않는 곳이 있어 아무것도 수정하지 않음 (라우터 파일이 올려 주신 판과 다름 — 해당 파일을 보내 주세요)")
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
