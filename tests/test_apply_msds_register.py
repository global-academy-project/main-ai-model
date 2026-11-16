"""apply_msds_register.py 시험. 패치된 저장소 파일 4개를 임시 폴더에 복사해 되돌리기 → 다시 적용 → 원래 바이트 확인."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "apply_msds_register.py"
FILES = ["app/core/config.py", "app/tools/tool_setup.py", "app/agent/agent_deps.py", "app/agent/drafting.py"]

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="apply_msds_register.py가 저장소 루트에 없음")


def run(root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(root), *flags], capture_output=True, text=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tree(tmp_path):
    """패치된 4개 파일의 사본."""
    for rel in FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    return tmp_path


def snapshot(root: Path) -> dict[str, str]:
    return {rel: sha(root / rel) for rel in FILES}


def test_repo_files_are_in_applied_state():
    done = run(ROOT, "--check")
    assert done.returncode == 0, done.stdout
    assert done.stdout.count("적용됨") == 4


def test_revert_then_apply_round_trip(tree):
    patched = snapshot(tree)
    assert run(tree, "--revert").returncode == 0
    reverted = snapshot(tree)
    assert all(reverted[rel] != patched[rel] for rel in FILES)
    assert "kosha_msds_service_key" not in (tree / "app/core/config.py").read_text(encoding="utf-8")
    assert "msds_tool" not in (tree / "app/tools/tool_setup.py").read_text(encoding="utf-8")
    assert "build_msds_tool" not in (tree / "app/agent/agent_deps.py").read_text(encoding="utf-8")
    assert "register_publicdata_msds" in (tree / "app/agent/agent_deps.py").read_text(encoding="utf-8")
    check = run(tree, "--check")
    assert check.returncode == 0 and check.stdout.count("원본") == 4

    assert run(tree).returncode == 0
    assert snapshot(tree) == patched  # 바이트까지 같다


def test_newlines_are_preserved(tree):
    before = {rel: (tree / rel).read_bytes() for rel in FILES}
    run(tree, "--revert")
    for rel in FILES:
        raw = (tree / rel).read_bytes()
        crlf = before[rel].count(b"\r\n")
        assert (raw.count(b"\r\n") > 0) == (crlf > 0)
        if crlf:
            assert raw.count(b"\n") == raw.count(b"\r\n")  # CRLF 파일에 맨 LF가 끼지 않는다
        else:
            assert b"\r" not in raw


def test_double_apply_and_double_revert_are_refused(tree):
    first = snapshot(tree)
    again = run(tree)  # 이미 적용된 상태에서 적용
    assert again.returncode == 1 and "이미 적용" in again.stdout
    assert snapshot(tree) == first

    assert run(tree, "--revert").returncode == 0
    reverted = snapshot(tree)
    twice = run(tree, "--revert")
    assert twice.returncode == 1 and "적용되어 있지 않습니다" in twice.stdout
    assert snapshot(tree) == reverted


def test_partial_state_writes_nothing(tree):
    # 한 파일만 되돌려 섞인 상태를 만든다: 하나라도 다르면 아무것도 쓰지 않아야 한다
    assert run(tree, "--revert").returncode == 0
    assert run(tree).returncode == 0
    drafting = tree / "app/agent/drafting.py"
    raw = drafting.read_bytes().replace(b"**{m.query: m for m in msds}", b"**{m.query: m for m in msds if True}")
    drafting.write_bytes(raw)
    before = snapshot(tree)
    for flags in ((), ("--revert",)):
        done = run(tree, *flags)
        assert done.returncode == 1
        assert snapshot(tree) == before
    assert run(tree, "--check").returncode == 1


def test_mixed_newlines_refused(tree):
    target = tree / "app/tools/tool_setup.py"
    raw = target.read_bytes()
    target.write_bytes(raw.replace(b"\r\n", b"\n", 1) if b"\r\n" in raw else raw.replace(b"\n", b"\r\n", 1))
    before = snapshot(tree)
    done = run(tree, "--revert")
    assert done.returncode != 0 and snapshot(tree) == before


def test_missing_file_is_reported(tree):
    (tree / "app/agent/drafting.py").unlink()
    done = run(tree, "--check")
    assert done.returncode == 2 and "파일이 없습니다" in done.stdout


def test_root_is_required():
    done = subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 2


def test_half_applied_file_is_not_treated_as_original(tree):
    """config.py에서 첫 편집만 들어간 상태(새 문장이 옛 문장을 품는 편집)는 '원본'이 아니다 → 적용도 되돌리기도 거부."""
    assert run(tree, "--revert").returncode == 0
    import importlib.util

    spec = importlib.util.spec_from_file_location("apply_msds_register", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tree / "app/core/config.py"
    text, newline = module.read(config)
    old, new = module.EDITS["app/core/config.py"][0]
    assert text.count(old) == 1 and new.startswith(old)  # 새 문장이 옛 문장을 품는 편집
    config.write_bytes(text.replace(old, new, 1).replace("\n", newline).encode("utf-8"))
    before = snapshot(tree)
    for flags in ((), ("--revert",), ("--check",)):
        assert run(tree, *flags).returncode != 0
        assert snapshot(tree) == before
