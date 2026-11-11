"""시나리오 판단 규칙(scenario_policy.py)을 분류기·디스패처에 연결한다 (2개 파일, 4곳, 줄바꿈 유지).

    python -B apply_scenario_policy.py            적용
    python -B apply_scenario_policy.py --revert   되돌리기
    python -B apply_scenario_policy.py --check    적용 여부만 확인

저장소 루트에서 실행한다. 통째로 덮어쓰지 않고 정확히 일치하는 줄만 바꾼다.
찾는 줄이 없으면 어느 파일도 바꾸지 않고 멈춘다 (두 파일을 함께 바꾸거나, 아예 바꾸지 않는다).
"""

from __future__ import annotations

import sys
from pathlib import Path

CLASSIFIER = Path("app/agent/scenario/scenario_classifier.py")
DISPATCH = Path("app/agent/scenario/scenario_dispatch.py")

C_IMPORT_OLD = "from .scenario_router import MAX_REFERENCES_IN_PROMPT, Scenario, ScenarioDecision, ScenarioRouter\n"
C_IMPORT_NEW = "from .scenario_policy import decide\n" + C_IMPORT_OLD
C_SIG_OLD = "    def classify(self, message: str, *, use_rag: bool = True) -> ClassificationResult:\n"
C_SIG_NEW = "    def classify(self, message: str, *, use_rag: bool = True, mode: str | None = None) -> ClassificationResult:\n"
C_BODY_OLD = (
    '        text = (message or "").strip()\n'
    "        rag = use_rag and self._retriever is not None and bool(text)\n"
)
C_BODY_NEW = (
    '        text = (message or "").strip()\n'
    "        if (ruled := decide(text, mode=mode, allowed=self._chat_scenarios)) is not None:\n"
    "            # 규칙(scenario_policy.py)이 정했으면 검색도 모델도 부르지 않는다\n"
    "            return ClassificationResult(\n"
    "                decision=ruled,\n"
    "                classifier_model=self._provider.binding_for(ROUTER_KEY),\n"
    "                scenario_model=(\n"
    "                    None if ruled.scenario is Scenario.UNKNOWN else self._provider.binding_for(ruled.scenario.value)\n"
    "                ),\n"
    "                rag_used=False,\n"
    "            )\n"
    "        rag = use_rag and self._retriever is not None and bool(text)\n"
)
D_OLD = "        return self._classifier.classify(text).decision, text"  # 파일의 마지막 줄이라 줄바꿈이 없을 수 있다
D_NEW = (
    "        mode = request.payload.get(_MODE_PAYLOAD_KEY)  # 현장 모드는 시나리오 판단 규칙이 쓴다 (scenario_policy.py)\n"
    '        options = {} if mode is None else {"mode": mode}\n'
    "        return self._classifier.classify(text, **options).decision, text"
)

EDITS: tuple[tuple[Path, tuple[tuple[str, str], ...]], ...] = (
    (CLASSIFIER, ((C_IMPORT_OLD, C_IMPORT_NEW), (C_SIG_OLD, C_SIG_NEW), (C_BODY_OLD, C_BODY_NEW))),
    (DISPATCH, ((D_OLD, D_NEW),)),
)


def _read(path: Path) -> tuple[str, str]:
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def main(argv: list[str]) -> int:
    mode = "revert" if "--revert" in argv else "check" if "--check" in argv else "apply"
    missing = [p for p, _ in EDITS if not p.is_file()]
    if missing:
        print(f"[중단] {', '.join(str(p) for p in missing)}를 찾을 수 없습니다. 저장소 루트에서 실행하세요.")
        return 2
    texts = {p: _read(p) for p, _ in EDITS}
    applied = {p: "scenario_policy" in texts[p][0] for p, _ in EDITS}
    if mode == "check":
        print("적용됨" if all(applied.values()) else "일부만 적용됨" if any(applied.values()) else "적용 안 됨")
        return 0
    if mode == "apply" and all(applied.values()):
        print("이미 적용돼 있습니다. 바꾼 것이 없습니다.")
        return 0
    if mode == "revert" and not any(applied.values()):
        print("적용돼 있지 않습니다. 바꾼 것이 없습니다.")
        return 0
    if mode == "apply" and any(applied.values()):
        print("[중단] 일부만 적용된 상태입니다. 파일을 바꾸지 않았습니다. --revert로 되돌린 뒤 다시 적용하세요.")
        return 1
    results: dict[Path, str] = {}
    for path, pairs in EDITS:
        text = texts[path][0]
        for old, new in pairs:
            src, dst = (old, new) if mode == "apply" else (new, old)
            if text.count(src) != 1:
                print(f"[중단] {path}에서 바꿀 줄을 정확히 한 곳 찾지 못했습니다 ({text.count(src)}곳). 어느 파일도 바꾸지 않았습니다.")
                print(f"  찾은 줄의 첫머리: {src.splitlines()[0].strip()[:70]}")
                return 1
            text = text.replace(src, dst)
        results[path] = text
    for path, text in results.items():
        newline = texts[path][1]
        path.write_bytes(text.replace("\n", newline).encode("utf-8"))
        print(("적용" if mode == "apply" else "되돌림") + f": {path} (줄바꿈 {'CRLF' if newline == chr(13) + chr(10) else 'LF'} 유지)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
