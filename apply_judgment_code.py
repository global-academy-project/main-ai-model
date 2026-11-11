"""work_register.py에 판단 '코드' 방식을 연결한다 (2곳, 줄바꿈 유지).

    python -B apply_judgment_code.py            적용
    python -B apply_judgment_code.py --revert   되돌리기
    python -B apply_judgment_code.py --check    적용 여부만 확인

저장소 루트에서 실행한다. 통째로 덮어쓰지 않고 정확히 일치하는 줄만 바꾼다.
judgment_pick 패치를 먼저 적용한 상태가 전제다 (pick 분기 앞에 code 분기를 한 줄 앞세운다).
찾는 줄이 없으면 아무것도 바꾸지 않고 멈춘다. 되돌릴 때는 이 패치를 먼저 되돌리고 judgment_pick을 되돌린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("app/agent/work_register.py")

IMPORT_OLD = "from .slots import ask_message, full_text, merge_slots, missing_fields, with_asked\n"
IMPORT_NEW = "from .judgment_code import JUDGMENT_CODE, judge_code\n" + IMPORT_OLD

BRANCH_OLD = '        if JUDGMENT_MODE == "pick":  # 위험요인은 코드, LLM은 근거 번호만 (judgment_pick.py). "full"이면 이전 방식\n'
BRANCH_NEW = (
    "        if JUDGMENT_CODE:  # 위험요인마다 다시 검색해 코드가 근거를 고른다. LLM 없음 (judgment_code.py). False면 아래 방식\n"
    "            hazards, judgment_source, dropped = judge_code(tools, inp)\n"
    '            judged_by = {"mode": "code"}\n'
    '        elif JUDGMENT_MODE == "pick":  # 위험요인은 코드, LLM은 근거 번호만 (judgment_pick.py). "full"이면 이전 방식\n'
)


def _read(path: Path) -> tuple[str, str]:
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def main(argv: list[str]) -> int:
    mode = "revert" if "--revert" in argv else "check" if "--check" in argv else "apply"
    if not TARGET.is_file():
        print(f"[중단] {TARGET}를 찾을 수 없습니다. 저장소 루트에서 실행하세요.")
        return 2
    text, newline = _read(TARGET)
    applied = "judge_code" in text
    if mode == "check":
        print("적용됨" if applied else "적용 안 됨")
        return 0
    if mode == "apply":
        if applied:
            print("이미 적용돼 있습니다. 바꾼 것이 없습니다.")
            return 0
        if text.count(IMPORT_OLD) != 1 or text.count(BRANCH_OLD) != 1:
            print("[중단] 바꿀 줄을 정확히 한 곳씩 찾지 못했습니다. 파일을 바꾸지 않았습니다.")
            print(f"  import 줄 {text.count(IMPORT_OLD)}곳, pick 분기 {text.count(BRANCH_OLD)}곳 (각 1곳이어야 함)")
            if "judge_pick" not in text:
                print("  judgment_pick 패치가 먼저 적용돼 있어야 합니다 (python -B apply_judgment_pick.py).")
            return 1
        new = text.replace(IMPORT_OLD, IMPORT_NEW).replace(BRANCH_OLD, BRANCH_NEW)
    else:
        if not applied:
            print("적용돼 있지 않습니다. 바꾼 것이 없습니다.")
            return 0
        if text.count(IMPORT_NEW) != 1 or text.count(BRANCH_NEW) != 1:
            print("[중단] 되돌릴 줄을 정확히 한 곳씩 찾지 못했습니다. 파일을 바꾸지 않았습니다.")
            return 1
        new = text.replace(IMPORT_NEW, IMPORT_OLD).replace(BRANCH_NEW, BRANCH_OLD)
    TARGET.write_bytes(new.replace("\n", newline).encode("utf-8"))
    print(("적용" if mode == "apply" else "되돌림") + f": {TARGET} (줄바꿈 {'CRLF' if newline == chr(13) + chr(10) else 'LF'} 유지)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
