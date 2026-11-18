"""어댑터 내장 위치 1판 → 2판 업그레이드 적용기 (B, 2026-11-18, T32).

1판(adapters/plan_brief_0.6b, 환경변수가 이김)을 이미 적용했는데 2판 적용기가 "정확히 일치하지 않습니다"라고 할 때 쓴다.
1판이 적용된 파일 4개를 2판(adapters/work_register, 환경변수 안 봄, 설정+가중치 둘 다 필요) 문장으로 바꾼다.
  python apply_adapter_builtin_upgrade.py --root <저장소 루트> --check    바꿀 수 있는지만 본다 (파일을 쓰지 않는다)
  python apply_adapter_builtin_upgrade.py --root <저장소 루트>            바꾼다
  python apply_adapter_builtin_upgrade.py --root <저장소 루트> --revert   2판 → 1판으로 되돌린다
- 새 파일 app/llm/adapter_paths.py, tests/test_adapter_builtin.py 는 zip 이 이미 2판으로 덮어썼어야 한다 (그 파일들은 이 적용기가 만지지 않는다).
- 1판을 적용하지 않았다면 이 적용기가 아니라 apply_adapter_builtin.py 를 쓴다.
- --root 는 필수다. 전부 맞을 때만 쓴다. 하나라도 안 맞으면 아무 파일도 바꾸지 않는다. 이미 2판이면 거부한다. 파일의 원래 줄바꿈(CRLF/LF)을 따른다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EDITS = json.loads(r'''[
 {
  "file": "app/llm/llm_transformers.py",
  "name": "llm_transformers: 머리말에 내장 위치 설명 (1판→2판)",
  "old": "  어댑터 폴더는 `Settings.llm_adapters`(env LLM_ADAPTERS, 이름 → 로컬 폴더)에서 찾고, 처음 쓸 때 한 번만 붙인다 (머지하지 않음).\n  환경변수 없이도 쓰는 내장 위치는 `adapter_paths.BUILTIN_ADAPTERS`(11/18, 계획 단계 plan_brief): 폴더가 있으면 이름이 더해지고 LLM_ADAPTERS 가 이긴다.\n",
  "new": "  어댑터 폴더는 `Settings.llm_adapters`(env LLM_ADAPTERS, 이름 → 로컬 폴더)에서 찾고, 처음 쓸 때 한 번만 붙인다 (머지하지 않음).\n  내장 위치는 `adapter_paths.BUILTIN_ADAPTERS`(11/18, 계획 단계 plan_brief, adapters/work_register): 폴더의 어댑터 파일로만 정하고 LLM_ADAPTERS 는 보지 않는다.\n"
 },
 {
  "file": "app/llm/llm_transformers.py",
  "name": "llm_transformers: 어댑터 폴더 표에 내장 위치 더하기 (1판→2판)",
  "old": "    \"\"\"어댑터 이름 → 로컬 폴더 (Settings.llm_adapters, env LLM_ADAPTERS + 내장 위치 adapter_paths). 설정을 읽지 못하면 비어 있음.\"\"\"\n    try:\n        from app.core import get_settings\n\n        from .adapter_paths import adapter_folders\n\n        return adapter_folders(get_settings())\n",
  "new": "    \"\"\"어댑터 이름 → 로컬 폴더 (Settings.llm_adapters(env LLM_ADAPTERS) + 내장 위치 adapter_paths, 내장 이름은 폴더로만). 설정을 읽지 못하면 비어 있음.\"\"\"\n    try:\n        from app.core import get_settings\n\n        from .adapter_paths import adapter_folders\n\n        return adapter_folders(get_settings())\n"
 },
 {
  "file": "app/llm/engines/llm_work_register.py",
  "name": "llm_work_register: 어댑터 설명·스위치 (1판→2판)",
  "old": "# 계획 단계 LoRA 어댑터 (11/17, 11/18 갱신). 어댑터 폴더가 있을 때만 클라이언트에 어댑터 이름을 붙인다 (build_work_register_client).\n# 폴더는 환경변수 없이 adapters/plan_brief_0.6b (app/llm/adapter_paths.py 의 내장 위치)이고, LLM_ADAPTERS 에 plan_brief 가 있으면 그 경로가 이긴다.\n# 폴더가 없으면 지금과 같다 (어댑터 없음, 긴 프롬프트, 경고 로그). 쓰는 순서와 짧은 프롬프트의 원본은 app/agent/plan_adapter.py.\n# 어댑터는 학습한 기본 모델(같은 model_id·같은 버전)에만 붙는다. 이 엔진의 모델을 바꾸면 어댑터도 다시 학습한다.\nADAPTER_NAME = \"plan_brief\"\nADAPTER_ENABLED = True  # False면 폴더가 있어도 붙이지 않는다\n",
  "new": "# 계획 단계 LoRA 어댑터 (11/17, 11/18 갱신). 어댑터 파일이 있을 때만 클라이언트에 어댑터 이름을 붙인다 (build_work_register_client).\n# 폴더는 환경변수 없이 adapters/work_register (app/llm/adapter_paths.py 의 내장 위치)이고, 어댑터 파일(adapter_config.json + 가중치)이 있을 때만 붙인다. LLM_ADAPTERS 는 보지 않는다.\n# 어댑터 파일이 없으면 지금과 같다 (어댑터 없음, 긴 프롬프트, 경고 로그). 쓰는 순서와 짧은 프롬프트의 원본은 app/agent/plan_adapter.py.\n# 어댑터는 학습한 기본 모델(같은 model_id·같은 버전)에만 붙는다. 이 엔진의 모델을 바꾸면 어댑터도 다시 학습한다.\nADAPTER_NAME = \"plan_brief\"\nADAPTER_ENABLED = True  # False면 어댑터 파일이 있어도 붙이지 않는다\n"
 },
 {
  "file": "app/llm/engines/llm_work_register.py",
  "name": "llm_work_register: 빌드 함수 (1판→2판)",
  "old": "    if ADAPTER_ENABLED:\n        folder = adapter_paths.adapter_folders(settings).get(ADAPTER_NAME)  # LLM_ADAPTERS 가 이기고, 없으면 내장 위치(adapter_config.json 이 있을 때만)\n        if folder:\n            client.adapter = ADAPTER_NAME  # 이 인스턴스만. router·B·E·F 는 같은 기본 모델에서 어댑터를 끄고 생성한다\n            logger.info(\"계획 어댑터 %s 를 씁니다 (짧은 프롬프트). 폴더=%s\", ADAPTER_NAME, folder)\n        else:\n            logger.warning(\"계획 어댑터 폴더가 없어 어댑터 없이 긴 프롬프트로 계속합니다. 두는 곳: %s (또는 LLM_ADAPTERS 의 %s)\",\n                           adapter_paths.BUILTIN_ADAPTERS.get(ADAPTER_NAME), ADAPTER_NAME)\n    return client\n",
  "new": "    if ADAPTER_ENABLED:\n        folder = adapter_paths.adapter_folders(settings).get(ADAPTER_NAME)  # 내장 위치 adapters/work_register 에 어댑터 파일이 있을 때만 (환경변수 안 봄)\n        if folder:\n            client.adapter = ADAPTER_NAME  # 이 인스턴스만. router·B·E·F 는 같은 기본 모델에서 어댑터를 끄고 생성한다\n            logger.info(\"계획 어댑터 %s 를 씁니다 (짧은 프롬프트). 폴더=%s\", ADAPTER_NAME, folder)\n        else:\n            logger.warning(\"계획 어댑터 파일이 없어 어댑터 없이 긴 프롬프트로 계속합니다. 두는 곳: %s (adapter_config.json + adapter_model.safetensors)\",\n                           adapter_paths.BUILTIN_ADAPTERS.get(ADAPTER_NAME))\n    return client\n"
 },
 {
  "file": "app/agent/plan_adapter.py",
  "name": "plan_adapter: 머리말(엔진·설정) (1판→2판)",
  "old": "  - 엔진: app/llm/engines/llm_work_register.py 가 어댑터 폴더가 있을 때만 클라이언트에 `ADAPTER_NAME` 을 붙인다.\n",
  "new": "  - 엔진: app/llm/engines/llm_work_register.py 가 어댑터 파일이 있을 때만 클라이언트에 `ADAPTER_NAME` 을 붙인다.\n"
 },
 {
  "file": "app/agent/plan_adapter.py",
  "name": "plan_adapter: 머리말(설정) (1판→2판)",
  "old": "  - 설정: 환경변수 없이 adapters/plan_brief_0.6b (app/llm/adapter_paths.py 의 내장 위치). `llm_adapters` (env LLM_ADAPTERS)에 같은 이름이 있으면 그 경로가 이긴다.\n",
  "new": "  - 설정: 환경변수 없이 adapters/work_register (app/llm/adapter_paths.py 의 내장 위치). 어댑터 파일이 있는지로만 정하고 LLM_ADAPTERS 는 보지 않는다.\n"
 },
 {
  "file": "app/agent/plan_adapter.py",
  "name": "plan_adapter: 머리말(끄기) (1판→2판)",
  "old": "끄기: llm_work_register.py 의 `ADAPTER_ENABLED = False`, 또는 어댑터 폴더를 치운다 (LLM_ADAPTERS 에 \"plan_brief\" 가 있으면 그것도 뺀다).\n",
  "new": "끄기: llm_work_register.py 의 `ADAPTER_ENABLED = False`, 또는 어댑터 폴더(adapters/work_register)를 치운다.\n"
 },
 {
  "file": "app/agent/plan_adapter.py",
  "name": "plan_adapter: adapter_ready 설명 (1판→2판)",
  "old": "    - 클라이언트에 어댑터 이름이 붙어 있고 (엔진이 어댑터 폴더가 있을 때만 붙인다),\n",
  "new": "    - 클라이언트에 어댑터 이름이 붙어 있고 (엔진이 어댑터 파일이 있을 때만 붙인다),\n"
 },
 {
  "file": "app/agent/plan_adapter.py",
  "name": "plan_adapter: 폴더 찾기 (1판→2판)",
  "old": "        from app.core import get_settings\n        from app.llm.adapter_paths import adapter_folders\n\n        path = adapter_folders(get_settings()).get(name)  # LLM_ADAPTERS 가 이기고, 없으면 내장 위치 (adapters/plan_brief_0.6b)\n",
  "new": "        from app.core import get_settings\n        from app.llm.adapter_paths import adapter_folders\n\n        path = adapter_folders(get_settings()).get(name)  # 내장 위치 adapters/work_register 의 어댑터 파일로만 정한다 (환경변수 안 봄)\n"
 },
 {
  "file": "tests/test_plan_adapter.py",
  "name": "test_plan_adapter: 실제 어댑터 폴더와 상관없게 (1판→2판)",
  "old": "\n\n@pytest.fixture(autouse=True)\ndef _no_builtin_adapter(monkeypatch):\n    \"\"\"이 파일의 시험은 PC·서버에 실제 어댑터 폴더(adapters/plan_brief_0.6b)가 있든 없든 같아야 한다. 내장 위치 시험은 test_adapter_builtin.py.\"\"\"\n    monkeypatch.setattr(adapter_paths, \"BUILTIN_ADAPTERS\", {})\n\n\ndef make_work(**kw) -> WorkContext:\n",
  "new": "\n\n@pytest.fixture(autouse=True)\ndef _no_builtin_adapter(monkeypatch):\n    \"\"\"이 파일의 시험은 PC·서버에 실제 어댑터 폴더(adapters/work_register)가 있든 없든 같아야 한다. 내장 위치 시험은 test_adapter_builtin.py.\"\"\"\n    monkeypatch.setattr(adapter_paths, \"BUILTIN_ADAPTERS\", {})\n\n\ndef make_work(**kw) -> WorkContext:\n"
 }
]''')
REQUIRED_NEW_FILES = json.loads(r'''["app/llm/adapter_paths.py"]''')


def _newline(raw: bytes, path: str) -> str:
    crlf, lf = raw.count(b"\r\n"), raw.count(b"\n") - raw.count(b"\r\n")
    if crlf and lf:
        raise SystemExit(f"[중단] {path}: 줄바꿈이 CRLF와 LF로 섞여 있어 안전하게 고칠 수 없습니다.")
    return "\r\n" if crlf else "\n"


def _state(text: str, old: str, new: str) -> str:
    """'todo'(적용 전) | 'done'(적용됨) | 'bad'. 새 문장이 옛 문장을 품어도 새 문장이 한 번 있으면 적용된 것이다."""
    n_new, n_old = text.count(new), text.count(old)
    if n_new == 1:
        return "done"
    if n_new == 0 and n_old == 1:
        return "todo"
    return "bad"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="저장소 루트 (app/ 폴더가 있는 곳)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="적용 가능한지만 본다")
    mode.add_argument("--revert", action="store_true", help="되돌린다")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not (root / "app").is_dir():
        print(f"[중단] {root} 아래에 app/ 폴더가 없습니다. --root 를 저장소 루트로 지정하세요.")
        return 2
    missing = [f for f in REQUIRED_NEW_FILES if not (root / f).is_file()]
    if missing and not args.revert:
        print("[중단] 새 파일이 없습니다. zip을 이 루트에 먼저 풀어 주세요:", ", ".join(missing))
        return 2

    texts: dict[str, str] = {}
    raws: dict[str, bytes] = {}
    newlines: dict[str, str] = {}
    problems: list[str] = []
    plan: list[tuple[dict, str]] = []
    for edit in EDITS:
        rel = edit["file"]
        path = root / rel
        if rel not in texts:
            if not path.is_file():
                problems.append(f"{rel}: 파일이 없습니다")
                continue
            raws[rel] = path.read_bytes()
            newlines[rel] = _newline(raws[rel], rel)
            texts[rel] = raws[rel].decode("utf-8")
        nl = newlines[rel]
        old, new = edit["old"].replace("\n", nl), edit["new"].replace("\n", nl)
        state = _state(texts[rel], old, new)
        want = "done" if args.revert else "todo"
        if state != want:
            if state == "done" and not args.revert:
                problems.append(f"{rel} [{edit['name']}]: 이미 적용돼 있습니다 (두 번 적용하지 않습니다)")
            elif state == "todo" and args.revert:
                problems.append(f"{rel} [{edit['name']}]: 적용된 흔적이 없어 되돌릴 수 없습니다")
            else:
                problems.append(f"{rel} [{edit['name']}]: 기대한 문장과 정확히 일치하지 않습니다 (파일이 다른 판일 수 있습니다)")
            continue
        if not args.check:
            a, b = (new, old) if args.revert else (old, new)
            texts[rel] = texts[rel].replace(a, b, 1)
        plan.append((edit, state))

    if problems:
        print("[중단] 아무 파일도 바꾸지 않았습니다.")
        for p in problems:
            print("  -", p)
        return 1

    for edit, _ in plan:
        print(f"  {'되돌릴 수 있음' if args.revert else '적용할 수 있음'}: {edit['file']} [{edit['name']}]" if args.check
              else f"  {'되돌림' if args.revert else '적용'}: {edit['file']} [{edit['name']}]")
    if args.check:
        print(f"[확인] 편집 {len(plan)}곳 모두 {'되돌릴' if args.revert else '적용할'} 수 있습니다. 파일은 바꾸지 않았습니다.")
        return 0
    for rel, text in texts.items():
        (root / rel).write_bytes(text.encode("utf-8"))
        print(f"  SHA256 {rel}: {hashlib.sha256((root / rel).read_bytes()).hexdigest()}")
    print(f"[완료] 편집 {len(plan)}곳을 {'되돌렸습니다' if args.revert else '적용했습니다'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
