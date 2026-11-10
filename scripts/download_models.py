"""C 담당 모델 파일을 미리 받아 둔다 (인터넷 되는 PC 에서 한 번). 서버는 실행 중에 모델을 내려받지 않는다 (폐쇄망).

    python -m scripts.download_models --dest D:/models            # 전부
    python -m scripts.download_models --dest D:/models --only vision

받은 뒤 .env 에 경로를 적는다 (출력에 나온 줄을 그대로 복사):
    VISION_MODEL_PATH=D:/models/yolox/yolox_nano.onnx

폐쇄망 서버에는 --dest 폴더를 통째로 복사하고 경로만 서버 기준으로 바꿔 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# YOLOX 공식 배포 (Megvii-BaseDetection/YOLOX, Apache-2.0). COCO 80종, 입력 416×416, 출력 (1, 3549, 85)
YOLOX_NANO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_vision(dest: Path) -> str:
    target = dest / "yolox" / "yolox_nano.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"내려받는 중: {YOLOX_NANO_URL}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(YOLOX_NANO_URL, tmp)
        tmp.replace(target)
    print(f"  {target} ({target.stat().st_size / 1e6:.1f} MB, sha256 {_sha256(target)[:16]}…)")
    return f"VISION_MODEL_PATH={target.as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C 담당 모델 파일 미리 받기")
    parser.add_argument("--dest", required=True, help="모델을 둘 폴더 (예: D:/models)")
    parser.add_argument("--only", choices=["vision"], help="하나만 받기")
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    lines = []
    if args.only in (None, "vision"):
        lines.append(download_vision(dest))
    print("\n.env 에 적을 줄:")
    for line in lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
