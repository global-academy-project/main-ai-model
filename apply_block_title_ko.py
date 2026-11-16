"""TBM 첫 요점의 영어 규칙 코드를 한글로 (11/16). 새 파일 app/agent/rule_titles.py와 함께 쓴다.

기존 파일 1개를 정확 일치로만 고친다 (줄바꿈 그대로 보존).
  app/agent/work_register.py   rule_titles 임포트 1줄, _block_titles의 제목 목록 1줄

새 파일(app/agent/rule_titles.py, tests/test_rule_titles.py)은 압축에서 그대로 복사한다.

    python apply_block_title_ko.py --root . --check     # 상태만 본다
    python apply_block_title_ko.py --root .             # 적용
    python apply_block_title_ko.py --root . --revert    # 되돌리기 (rule_titles.py가 남아 있어도 쓰이지 않는다)

두 번 적용은 거부한다. 상태 판정: 새 문장이 모두 한 번씩 있으면 '적용됨', 옛 문장이 모두 한 번씩 있고 새 문장이 없으면 '원본'
(새 문장이 옛 문장을 품는 편집이 있어서 옛 문장만 보고 판정하지 않는다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EDITS: dict[str, list[tuple[str, str]]] = {'app/agent/work_register.py': [('from .plan_brief import MSDS_FIRST, PLAN_MODE, StagedCollector\n',
                                 'from .plan_brief import MSDS_FIRST, PLAN_MODE, StagedCollector\n'
                                 'from .rule_titles import blocking_titles\n'),
                                ('    titles = [e.title for e in rule_check.evaluations if e.severity == "BLOCK" and e.status != "PASS"] '
                                 'if rule_check else []\n',
                                 '    titles = blocking_titles(rule_check.evaluations) if rule_check else []\n')]}


def read(path: Path) -> tuple[str, str]:
    """(\\n으로 통일한 본문, 원래 줄바꿈). CRLF와 LF가 섞인 파일은 거부한다."""
    raw = path.read_bytes().decode("utf-8")
    crlf = raw.count("\r\n")
    bare_lf = raw.count("\n") - crlf
    if crlf and bare_lf:
        raise SystemExit(f"{path}: CRLF와 LF가 섞여 있습니다. 적용하지 않습니다.")
    return raw.replace("\r\n", "\n"), "\r\n" if crlf else "\n"


def state_of(text: str, edits: list[tuple[str, str]]) -> str:
    if all(text.count(new) == 1 for _, new in edits):
        return "적용됨"
    if all(text.count(old) == 1 and text.count(new) == 0 for old, new in edits):
        return "원본"
    return "알 수 없음"


def transform(text: str, edits: list[tuple[str, str]], *, forward: bool) -> str:
    for old, new in edits:
        src, dst = (old, new) if forward else (new, old)
        if text.count(src) != 1:
            raise SystemExit("편집 위치를 한 곳으로 정할 수 없습니다.")
        text = text.replace(src, dst, 1)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TBM 규칙 제목 한글 보강 패치 적용기")
    parser.add_argument("--root", required=True, help="저장소 루트 (app 폴더가 있는 곳)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="상태만 본다")
    mode.add_argument("--revert", action="store_true", help="되돌린다")
    args = parser.parse_args(argv)

    root = Path(args.root)
    loaded: dict[str, tuple[str, str]] = {}
    states: dict[str, str] = {}
    for rel, edits in EDITS.items():
        path = root / rel
        if not path.is_file():
            print(f"{rel}: 파일이 없습니다.")
            return 2
        loaded[rel] = read(path)
        states[rel] = state_of(loaded[rel][0], edits)
        print(f"{rel}: {states[rel]} ({'CRLF' if loaded[rel][1] == chr(13) + chr(10) else 'LF'})")

    if args.check:
        return 0 if len(set(states.values())) == 1 and "알 수 없음" not in states.values() else 1

    want = "적용됨" if args.revert else "원본"
    if any(s != want for s in states.values()):
        if all(s == "적용됨" for s in states.values()):
            print("이미 적용되어 있습니다. 아무것도 바꾸지 않았습니다.")
        elif all(s == "원본" for s in states.values()):
            print("적용되어 있지 않습니다. 아무것도 바꾸지 않았습니다.")
        else:
            print("파일 상태가 기대와 다릅니다 (원본이 아니거나 일부만 적용됨). 아무것도 바꾸지 않았습니다.")
        return 1

    results = {rel: transform(loaded[rel][0], EDITS[rel], forward=not args.revert) for rel in EDITS}
    for rel, text in results.items():
        newline = loaded[rel][1]
        (root / rel).write_bytes(text.replace("\n", newline).encode("utf-8"))
    print("되돌렸습니다." if args.revert else "적용했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
