"""scripts/download_models.py: 내려받기·변환 없이 돌 수 있는 부분만 시험한다 (네트워크·torch 불필요)."""

from __future__ import annotations

from scripts import download_models as dm


def _fake_converted(folder):
    folder.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"):
        (folder / name).write_bytes(b"x")


def test_stt_default_is_korean_model(tmp_path, monkeypatch, capsys):
    _fake_converted(tmp_path / "whisper-small-ko")   # 이미 변환된 폴더 → 다시 받지 않는다
    monkeypatch.setattr(dm, "download_stt_base", lambda dest: (_ for _ in ()).throw(AssertionError("원본을 받으면 안 됨")))

    assert dm.main(["--dest", str(tmp_path), "--only", "stt"]) == 0
    out = capsys.readouterr().out
    assert f"STT_MODEL_PATH={(tmp_path / 'whisper-small-ko').as_posix()}" in out
    assert "STT_MODEL_NAME=whisper-small-ko" in out


def test_ready_needs_all_files(tmp_path):
    folder = tmp_path / "whisper-small-ko"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"x")
    (folder / "config.json").write_bytes(b"x")
    assert not dm.stt_ko_ready(folder)   # tokenizer.json 이 없으면 덜 된 변환
    (folder / "tokenizer.json").write_bytes(b"x")
    assert dm.stt_ko_ready(folder)


def test_stt_base_prints_base_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "download_stt_base",
                        lambda dest: [f"STT_MODEL_PATH={(dest / 'whisper-small').as_posix()}", "STT_MODEL_NAME=whisper-small"])
    assert dm.main(["--dest", str(tmp_path), "--only", "stt-base"]) == 0
    assert "STT_MODEL_NAME=whisper-small\n" in capsys.readouterr().out
