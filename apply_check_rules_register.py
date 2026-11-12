"""check_rules 실제 도구 등록 패치 (정확 일치, 전부 맞지 않으면 무적용).

    python apply_check_rules_register.py --root <main-ai-model 폴더> --check     # 적용 가능한지만 확인
    python apply_check_rules_register.py --root <main-ai-model 폴더>             # 적용
    python apply_check_rules_register.py --root <main-ai-model 폴더> --revert    # 되돌리기

고치는 파일: app/tools/tool_setup.py, app/agent/agent_deps.py (줄바꿈은 파일 그대로 유지)
새로 두는 파일: app/agent/check_rules_setup.py, tests/test_check_rules_registration.py (payload 폴더에서 복사)
끄기: .env의 CHECK_RULES_TOOL=MOCK (코드는 그대로 두고 변경 전과 같은 동작), 완전히 되돌리려면 --revert.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NEW_FILES = ["app/agent/check_rules_setup.py", "tests/test_check_rules_registration.py"]

# 원본 SHA256 (사용자가 올린 최신본, 11/12)
ORIGINAL_SHA256 = {
    "app/tools/tool_setup.py": "d989c841edb10b82ffe5ca3c74af640c4d1b2dee608446a33c1bada12f1aa4ed",
    "app/agent/agent_deps.py": "3d0a7e6d19aa86dbd45d4d7b588c3e478c678d28b7bed2b004b61c34102241db",
}

# (파일, [(이전, 이후), ...]) — 텍스트는 LF로 적고, 파일의 줄바꿈(CRLF)으로 바꿔 비교·기록한다
EDITS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "app/tools/tool_setup.py",
        [
            (
                "def build_tool_registry(\n"
                "    *, extra: Callable[[ToolRegistry], None] | None = None, rag_backend: str | None = None\n"
                ") -> ToolRegistry:\n",
                "CHECK_RULES_DESCRIPTION = \"가스농도·필수 배치·필수 항목 판정 (Spring 규칙 엔진 호출)\"\n"
                "\n"
                "def build_tool_registry(\n"
                "    *, extra: Callable[[ToolRegistry], None] | None = None, rag_backend: str | None = None,\n"
                "    check_rules_tool: tuple[Callable, dict] | None = None,\n"
                ") -> ToolRegistry:\n",
            ),
            (
                "    비어 있으면 mock 그대로다. 잘못된 값은 ValueError. 적용 순서는 mock → RAG → extra다.\n",
                "    비어 있으면 mock 그대로다. 잘못된 값은 ValueError. 적용 순서는 mock → 실제 check_rules → RAG → extra다.\n"
                "\n"
                "    check_rules_tool=(handler, parameters)가 있으면 check_rules를 C의 실제 도구로 바꾼다 (CHECK_RULES_TOOL=SPRING).\n"
                "    만드는 곳은 app/agent/check_rules_setup.py다. None이면 mock 그대로다.\n",
            ),
            (
                "    register_mock_tools(registry)\n",
                "    register_mock_tools(registry)\n"
                "    if check_rules_tool is not None:  # C의 실제 check_rules (mock과 같은 입출력)\n"
                "        handler, parameters = check_rules_tool\n"
                "        registry.register(\"check_rules\", handler, owner=\"C\", description=CHECK_RULES_DESCRIPTION,\n"
                "                          parameters=parameters, replace=True)\n",
            ),
        ],
    ),
    (
        "app/agent/agent_deps.py",
        [
            (
                "from .work_register import WorkRegisterAgent\n",
                "from .work_register import WorkRegisterAgent\n"
                "from .check_rules_setup import build_check_rules_tool\n",
            ),
            (
                "    return build_tool_registry(\n"
                "        rag_backend=get_settings().rag_backend_path,\n"
                "    )\n",
                "    # check_rules: CHECK_RULES_TOOL=SPRING이면 C의 실제 도구로 바꾼다. MOCK이면 None이라 mock 그대로다.\n"
                "    return build_tool_registry(\n"
                "        rag_backend=get_settings().rag_backend_path,\n"
                "        check_rules_tool=build_check_rules_tool(get_settings()),\n"
                "    )\n",
            ),
        ],
    ),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def _convert(text: str, newline: str) -> str:
    return text.replace("\n", newline)


def plan(root: Path, revert: bool) -> tuple[list[tuple[Path, bytes]], list[str]]:
    """(쓸 내용, 문제 목록). 문제가 하나라도 있으면 아무것도 쓰지 않는다."""
    writes: list[tuple[Path, bytes]] = []
    problems: list[str] = []
    for rel, edits in EDITS:
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: 파일이 없습니다.")
            continue
        raw = path.read_bytes()
        nl = _newline(raw)
        text = raw.decode("utf-8")
        for old, new in edits:
            before, after = (new, old) if revert else (old, new)
            before, after = _convert(before, nl), _convert(after, nl)
            count = text.count(before)
            if count != 1:
                state = "이미 적용됨 또는 다름" if count == 0 else f"{count}곳과 일치"
                problems.append(f"{rel}: 정확히 한 곳과 일치해야 합니다 ({state}): {before.splitlines()[0][:60]!r}")
                continue
            text = text.replace(before, after, 1)
        writes.append((path, text.encode("utf-8")))
    for rel in NEW_FILES:
        target = root / rel
        source = HERE / rel
        if not revert:
            if target.exists() and target.read_bytes() != source.read_bytes():
                problems.append(f"{rel}: 이미 다른 내용의 파일이 있습니다.")
            writes.append((target, source.read_bytes()))
        elif target.exists() and target.read_bytes() != source.read_bytes():
            problems.append(f"{rel}: 패치 때 둔 내용과 달라 지우지 않습니다.")
    return writes, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="main-ai-model 저장소 루트")
    parser.add_argument("--check", action="store_true", help="적용 가능한지만 확인 (파일을 쓰지 않음)")
    parser.add_argument("--revert", action="store_true", help="적용을 되돌림")
    args = parser.parse_args()
    root: Path = args.root.resolve()

    if not args.revert:
        for rel, expected in ORIGINAL_SHA256.items():
            path = root / rel
            if path.is_file() and _sha(path.read_bytes()) != expected:
                print(f"[알림] {rel}: 전달본과 SHA256이 다릅니다 (정확 일치 검사는 계속합니다).")

    writes, problems = plan(root, args.revert)
    if problems:
        print("[중단] 적용하지 않았습니다 (아무 파일도 바꾸지 않음):")
        for p in problems:
            print("  -", p)
        return 1
    if args.check:
        print("[확인] 적용 가능합니다." if not args.revert else "[확인] 되돌릴 수 있습니다.")
        return 0

    for path, data in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if args.revert:
        for rel in NEW_FILES:
            target = root / rel
            if target.exists():
                target.unlink()
    print("[완료]", "되돌렸습니다." if args.revert else "적용했습니다. 서버를 다시 띄우세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
