"""C 담당 모델 파일을 미리 받아 둔다 (인터넷 되는 PC 에서 한 번). 서버는 실행 중에 모델을 내려받지 않는다 (폐쇄망).

    python -m scripts.download_models --dest <폴더>                  # 사진 인식 + 음성(한국어)
    python -m scripts.download_models --dest <폴더> --only vision    # 사진 인식 YOLOX-Nano
    python -m scripts.download_models --dest <폴더> --only stt       # 음성: SungBeom/whisper-small-ko 를 받아 변환 (기본)
    python -m scripts.download_models --dest <폴더> --only stt-base  # 음성: 원본 Whisper small (변환 없음, 한국어는 덜 정확)

받은 뒤 출력된 줄을 그대로 .env 에 붙인다 (경로는 --dest 기준으로 나온다):
    VISION_MODEL_PATH=<폴더>/yolox/yolox_nano.onnx
    STT_MODEL_PATH=<폴더>/whisper-small-ko
    STT_MODEL_NAME=whisper-small-ko

음성(한국어) 변환에는 torch 가 필요하다: python -m pip install -r requirements.release.txt
폐쇄망 서버에는 --dest 폴더를 통째로 복사하고 경로만 서버 기준으로 바꿔 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

# YOLOX 공식 배포 (Megvii-BaseDetection/YOLOX, Apache-2.0). COCO 80종, 입력 416×416, 출력 (1, 3549, 85)
YOLOX_NANO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"

# Whisper small 을 faster-whisper(CTranslate2) 형식으로 바꿔 둔 공개 저장소 (MIT). 한국어는 아래 SungBeom 보다 덜 정확
WHISPER_SMALL_REPO = "Systran/faster-whisper-small"

# Whisper small 을 한국어로 추가 학습한 모델 (11/10 결정, 음성 입력 기본). transformers 형식이라 CTranslate2 int8 로 바꿔 쓴다
SUNGBEOM_REPO = "SungBeom/whisper-small-ko"
SUNGBEOM_REVISION = "refs/pr/2"   # 11/10 실제 변환에 쓴 버전. 바꾸면 받아 적기 결과를 다시 확인할 것
STT_KO_NAME = "whisper-small-ko"
# 변환 결과에 같이 둘 파일 (원본 저장소에 있는 것만 복사). 토크나이저 파일은 아래에서 다시 저장한 것을 쓴다
STT_KO_COPY_FILES = ("preprocessor_config.json",)
# 토크나이저를 읽어 다시 저장한 파일 중 가져올 것. tokenizer.json 은 원본에 없어서 이렇게 만든다 (11/10 수동 변환과 같은 파일)
STT_KO_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6


def download_vision(dest: Path) -> list[str]:
    target = dest / "yolox" / "yolox_nano.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"내려받는 중: {YOLOX_NANO_URL}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(YOLOX_NANO_URL, tmp)
        tmp.replace(target)
    print(f"  {target} ({target.stat().st_size / 1e6:.1f} MB, sha256 {_sha256(target)[:16]}…)")
    return [f"VISION_MODEL_PATH={target.as_posix()}"]


def download_stt_base(dest: Path) -> list[str]:
    target = dest / "whisper-small"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub 가 필요합니다 (pip install -r requirements.txt).") from exc
    print(f"내려받는 중: {WHISPER_SMALL_REPO}")
    snapshot_download(repo_id=WHISPER_SMALL_REPO, local_dir=str(target))
    print(f"  {target} ({_size_mb(target):.0f} MB)")
    return [f"STT_MODEL_PATH={target.as_posix()}", "STT_MODEL_NAME=whisper-small"]


def stt_ko_ready(target: Path) -> bool:
    """변환이 끝난 폴더인가 (faster-whisper 가 읽는 파일이 다 있음)"""
    return all((target / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json"))


def download_stt_ko(dest: Path) -> list[str]:
    """SungBeom/whisper-small-ko 를 받아 CTranslate2 int8 로 바꾼다 (11/10 수동 변환과 같은 순서)."""
    target = dest / STT_KO_NAME
    lines = [f"STT_MODEL_PATH={target.as_posix()}", f"STT_MODEL_NAME={STT_KO_NAME}"]
    if stt_ko_ready(target):
        print(f"  이미 있음: {target} ({_size_mb(target):.0f} MB)")
        return lines
    try:
        import torch  # noqa: F401 - 변환할 때 transformers 가 원본 가중치를 torch 로 읽는다
    except ImportError as exc:
        raise SystemExit("음성(한국어) 변환에는 torch 가 필요합니다:\n"
                         "  python -m pip install -r requirements.release.txt\n"
                         "변환 없이 원본 Whisper small 을 쓰려면 --only stt-base") from exc
    try:
        from ctranslate2.converters import TransformersConverter   # faster-whisper 와 같이 설치됨
        from huggingface_hub import snapshot_download
        from transformers import WhisperTokenizerFast
    except ImportError as exc:
        raise SystemExit(f"변환 도구를 불러오지 못했습니다 ({exc}). pip install -r requirements.release.txt") from exc

    with tempfile.TemporaryDirectory(prefix="whisper-ko-") as tmp:
        src = Path(tmp) / "src"
        print(f"내려받는 중: {SUNGBEOM_REPO} ({SUNGBEOM_REVISION})")
        snapshot_download(repo_id=SUNGBEOM_REPO, revision=SUNGBEOM_REVISION, local_dir=str(src))

        print(f"변환 중: CTranslate2 int8 → {target}")
        copy_files = [name for name in STT_KO_COPY_FILES if (src / name).is_file()]
        TransformersConverter(str(src), copy_files=copy_files).convert(str(target), quantization="int8", force=True)

        # 원본 저장소에 tokenizer.json 이 없다 → 토크나이저를 읽어 만들어 넣는다 (없으면 faster-whisper 가 허브에서 받으려 함)
        tok_dir = Path(tmp) / "tok"
        WhisperTokenizerFast.from_pretrained(str(src)).save_pretrained(str(tok_dir))
        for name in STT_KO_TOKENIZER_FILES:
            shutil.copyfile(tok_dir / name, target / name)

    if not stt_ko_ready(target):
        raise SystemExit(f"변환 결과가 모자랍니다: {target} (model.bin·config.json·tokenizer.json 확인)")
    print(f"  {target} ({_size_mb(target):.0f} MB)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C 담당 모델 파일 미리 받기")
    parser.add_argument("--dest", required=True, help="모델을 둘 폴더 (예: C:/models)")
    parser.add_argument("--only", choices=["vision", "stt", "stt-base"],
                        help="하나만 받기 (stt: 한국어 모델 변환, stt-base: 원본 Whisper small)")
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    lines: list[str] = []
    if args.only in (None, "vision"):
        lines += download_vision(dest)
    if args.only in (None, "stt"):
        lines += download_stt_ko(dest)
    if args.only == "stt-base":
        lines += download_stt_base(dest)
    print("\n.env 에 적을 줄:")
    for line in lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
