"""apply_first_aid_inhalation.py 시험 (draft_extras.py 사본으로 되돌리기 → 다시 적용 → 원래 바이트)."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "apply_first_aid_inhalation.py"
REL = "app/agent/draft_extras.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="apply_first_aid_inhalation.py가 저장소 루트에 없음")


def run(root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(root), *flags], capture_output=True, text=True)


SEGMENT = ROOT / "apply_first_aid_segment.py"


def to_first_patch_state(root: Path) -> None:
    """2차 패치(apply_first_aid_segment.py)가 들어 있으면 1차 상태로 되돌린다. 1차 적용기는 1차 상태의 파일만 다룬다."""
    if SEGMENT.is_file():
        done = subprocess.run([sys.executable, "-B", str(SEGMENT), "--root", str(root), "--revert"], capture_output=True, text=True)
        assert done.returncode in (0, 1), done.stdout  # 1: 2차가 적용되어 있지 않음


@pytest.fixture
def tree(tmp_path):
    (tmp_path / REL).parent.mkdir(parents=True)
    shutil.copy2(ROOT / REL, tmp_path / REL)
    to_first_patch_state(tmp_path)
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
    assert b"PREFER_INHALATION_FIRST_AID" not in reverted and b"_first_aid_text" not in reverted
    assert (reverted.count(b"\r\n") > 0) == (before.count(b"\r\n") > 0)
    if b"\r\n" in reverted:
        assert reverted.count(b"\n") == reverted.count(b"\r\n")
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
    spec = importlib.util.spec_from_file_location("apply_first_aid_inhalation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tree / REL
    text, newline = module.read(path)
    old, new = module.EDITS[REL][2]  # 새 문장이 옛 문장을 품는 편집(상수 추가)만 먼저 넣은 상태
    assert new.startswith(old)
    path.write_bytes(text.replace(old, new, 1).replace("\n", newline).encode("utf-8"))
    before = sha(tree)
    for flags in ((), ("--revert",), ("--check",)):
        assert run(tree, *flags).returncode != 0
        assert sha(tree) == before


def test_missing_file_and_root_required(tmp_path):
    assert run(tmp_path, "--check").returncode == 2
    assert subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, text=True).returncode == 2
