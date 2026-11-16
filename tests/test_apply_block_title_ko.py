"""apply_block_title_ko.py 시험 (work_register.py 사본으로 되돌리기 → 다시 적용 → 원래 바이트)."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "apply_block_title_ko.py"
REL = "app/agent/work_register.py"
OLD_LINE = b'    titles = [e.title for e in rule_check.evaluations if e.severity == "BLOCK" and e.status != "PASS"] if rule_check else []'

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="apply_block_title_ko.py가 저장소 루트에 없음")


def run(root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(root), *flags], capture_output=True, text=True)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / REL).parent.mkdir(parents=True)
    shutil.copy2(ROOT / REL, tmp_path / REL)
    return tmp_path


def sha(root: Path) -> str:
    return hashlib.sha256((root / REL).read_bytes()).hexdigest()


def test_repo_is_in_applied_state(tree):
    done = run(tree, "--check")
    assert done.returncode == 0 and "적용됨" in done.stdout


def test_round_trip_and_newlines(tree):
    patched = sha(tree)
    before = (tree / REL).read_bytes()
    assert run(tree, "--revert").returncode == 0
    reverted = (tree / REL).read_bytes()
    assert b"blocking_titles" not in reverted and OLD_LINE in reverted
    assert (reverted.count(b"\r\n") > 0) == (before.count(b"\r\n") > 0)
    if b"\r\n" in reverted:
        assert reverted.count(b"\n") == reverted.count(b"\r\n")
    assert run(tree, "--check").returncode == 0 and "원본" in run(tree, "--check").stdout
    assert run(tree).returncode == 0
    assert sha(tree) == patched


def test_double_apply_and_double_revert_refused(tree):
    first = sha(tree)
    again = run(tree)
    assert again.returncode == 1 and "이미 적용" in again.stdout and sha(tree) == first
    run(tree, "--revert")
    reverted = sha(tree)
    twice = run(tree, "--revert")
    assert twice.returncode == 1 and sha(tree) == reverted


def test_half_applied_is_refused(tree):
    run(tree, "--revert")
    spec = importlib.util.spec_from_file_location("apply_block_title_ko", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tree / REL
    text, newline = module.read(path)
    old, new = module.EDITS[REL][0]  # 새 문장이 옛 문장을 품는 편집(임포트 줄)만 먼저 넣은 상태
    assert new.startswith(old)
    path.write_bytes(text.replace(old, new, 1).replace("\n", newline).encode("utf-8"))
    before = sha(tree)
    for flags in ((), ("--revert",), ("--check",)):
        assert run(tree, *flags).returncode != 0
        assert sha(tree) == before


def test_root_is_required():
    done = subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 2


def test_missing_file(tmp_path):
    assert run(tmp_path).returncode == 2


def test_mixed_newlines_are_refused_even_for_original_text(tree):
    run(tree, "--revert")
    path = tree / REL
    raw = path.read_bytes()
    assert raw.count(b"\r\n") > 0 and OLD_LINE in raw.replace(b"\r\n", b"\n")
    mixed = raw.replace(b"\r\n", b"\n", 1)  # 줄 하나만 LF로: 본문은 원본이지만 줄바꿈이 섞임
    path.write_bytes(mixed)
    before = sha(tree)
    done = run(tree)
    assert done.returncode != 0 and "섞여" in (done.stdout + done.stderr) and sha(tree) == before
