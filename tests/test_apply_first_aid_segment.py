"""apply_first_aid_segment.py 시험. 1차 패치가 들어간 draft_extras.py 사본으로 되돌리기 → 다시 적용 → 원래 바이트."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "apply_first_aid_segment.py"
FIRST = ROOT / "apply_first_aid_inhalation.py"
REL = "app/agent/draft_extras.py"

pytestmark = pytest.mark.skipif(not (SCRIPT.is_file() and FIRST.is_file()), reason="적용기가 저장소 루트에 없음")


def run(script: Path, root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(script), "--root", str(root), *flags], capture_output=True, text=True)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / REL).parent.mkdir(parents=True)
    shutil.copy2(ROOT / REL, tmp_path / REL)
    return tmp_path


def sha(root: Path) -> str:
    return hashlib.sha256((root / REL).read_bytes()).hexdigest()


def test_repo_has_both_patches(tree):
    assert run(SCRIPT, ROOT, "--check").returncode == 0
    assert "적용됨" in run(SCRIPT, ROOT, "--check").stdout


def test_revert_gives_exactly_first_patch_state_and_reapply_restores(tree):
    patched = sha(tree)
    before = (tree / REL).read_bytes()
    assert run(SCRIPT, tree, "--revert").returncode == 0
    reverted = (tree / REL).read_bytes()
    assert b"FIRST_AID_FIRST_SEGMENT_ONLY" not in reverted and b"_first_segment" not in reverted
    assert b"_first_aid_text" in reverted  # 1차 패치는 남아 있다
    assert (reverted.count(b"\r\n") > 0) == (before.count(b"\r\n") > 0)
    if b"\r\n" in reverted:
        assert reverted.count(b"\n") == reverted.count(b"\r\n")
    check_first = run(FIRST, tree, "--check")
    assert check_first.returncode == 0 and "적용됨" in check_first.stdout  # 1차 적용기가 보기에도 1차 상태
    assert run(SCRIPT, tree).returncode == 0
    assert sha(tree) == patched


def test_second_patch_refuses_original_file(tree):
    run(SCRIPT, tree, "--revert")
    assert run(FIRST, tree, "--revert").returncode == 0  # 원본(1차도 없음)
    before = sha(tree)
    done = run(SCRIPT, tree)
    assert done.returncode == 1 and sha(tree) == before  # 1차 없이 2차만 넣지 않는다
    assert run(SCRIPT, tree, "--check").returncode == 1


def test_double_apply_and_double_revert_refused(tree):
    first = sha(tree)
    again = run(SCRIPT, tree)
    assert again.returncode == 1 and "이미 적용" in again.stdout and sha(tree) == first
    run(SCRIPT, tree, "--revert")
    reverted = sha(tree)
    twice = run(SCRIPT, tree, "--revert")
    assert twice.returncode == 1 and sha(tree) == reverted


def test_half_applied_is_refused(tree):
    run(SCRIPT, tree, "--revert")
    spec = importlib.util.spec_from_file_location("apply_first_aid_segment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tree / REL
    text, newline = module.read(path)
    old, new = module.EDITS[REL][2]  # 새 문장이 옛 문장을 품는 편집(상수 추가)만 먼저 넣은 상태
    assert new.startswith(old)
    path.write_bytes(text.replace(old, new, 1).replace("\n", newline).encode("utf-8"))
    before = sha(tree)
    for flags in ((), ("--revert",), ("--check",)):
        assert run(SCRIPT, tree, *flags).returncode != 0
        assert sha(tree) == before


def test_missing_file_and_root_required(tmp_path):
    assert run(SCRIPT, tmp_path, "--check").returncode == 2
    assert subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, text=True).returncode == 2
