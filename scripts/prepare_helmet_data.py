"""안전모 학습 데이터 정리: SHWD + Hard Hat Detection (Pascal VOC) → YOLOX 용 COCO (helmet / head 두 클래스).

    # 1) 개수만 보기 (이미지를 읽지 않음, 몇 초)
    python -m scripts.prepare_helmet_data --shwd <SHWD 폴더> --hardhat <hardhat 폴더> --dry-run
    # 2) 만들기 (이미지 고르기·줄이기·COCO·zip)
    python -m scripts.prepare_helmet_data --shwd <SHWD 폴더> --hardhat <hardhat 폴더> --out <출력 폴더> --zip

원본 폴더는 읽기만 한다. 결과:
  <출력>/train/*.jpg, <출력>/valid/*.jpg, <출력>/annotations/train.json, valid.json, <출력>/meta.json (+ <출력>.zip)

클래스 (인원 = helmet + head, 안전모 미착용 = head):
  helmet ← SHWD "hat", Hard Hat "helmet"     (안전모를 쓴 머리)
  head   ← SHWD "person", Hard Hat "head"    (맨머리. SHWD 의 person 은 머리 박스)
  버림   ← Hard Hat "person"(몸 전체 박스), 그 밖 ("dog" 등)

비율 맞추기: SHWD 의 PartA·PartB 이미지(4,340장)는 공사 현장이 아닌 군중 사진(SCUT-HEAD)이라 맨머리만 이미지당 수십 개다.
  전부 넣으면 맨머리가 안전모의 12배가 되고, 전부 빼면 안전모가 4배 이상 많아진다 → --crowd 장수만 무작위로 남긴다.
라이선스: SHWD (MIT), Hard Hat Detection (CC0 1.0, Make ML). meta.json 에 출처를 남긴다.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CLASSES = ("helmet", "head")
CLASS_MAP = {
    "shwd": {"hat": "helmet", "person": "head"},
    "hardhat": {"helmet": "helmet", "head": "head"},
}
CROWD_PREFIXES = ("PartA", "PartB")            # SHWD 안의 SCUT-HEAD 군중 사진
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_BOX = 2.0                                  # 이보다 작은 박스는 라벨 오류로 보고 버림
SOURCES = {
    "shwd": "Safety Helmet Wearing Dataset (njvisionpower/Safety-Helmet-Wearing-Dataset), MIT",
    "hardhat": "Hard Hat Detection (andrewmvd / Make ML, Kaggle·Voxel51), CC0 1.0",
}


@dataclass
class Sample:
    source: str
    stem: str
    xml: Path
    image: Path | None
    width: int                                  # xml 에 적힌 크기 (0 일 수 있음 → 이미지에서 다시 읽음)
    height: int
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)   # (클래스, x1, y1, x2, y2)
    dropped: Counter = field(default_factory=Counter)

    @property
    def crowd(self) -> bool:
        return self.source == "shwd" and self.stem.startswith(CROWD_PREFIXES)


def _annotation_folder(root: Path) -> Path:
    for name in ("Annotations", "annotations"):
        if (root / name).is_dir():
            return root / name
    found = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "annotations" and any(p.glob("*.xml"))]
    if not found:
        sys.exit(f"Annotations 폴더(xml)를 찾지 못했습니다: {root}")
    return found[0]


def _image_index(root: Path) -> dict[str, Path]:
    """root 아래 이미지 파일: 파일 이름(소문자) → 경로, 확장자 뺀 이름(소문자) → 경로"""
    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name.lower(), p)
            index.setdefault(p.stem.lower(), p)
    return index


def read_source(source: str, root: Path) -> list[Sample]:
    mapping = CLASS_MAP[source]
    images = _image_index(root)
    samples = []
    for xml in sorted(_annotation_folder(root).glob("*.xml")):
        tree = ET.parse(xml).getroot()
        filename = (tree.findtext("filename") or "").strip()
        image = images.get(filename.lower()) or images.get(xml.stem.lower())
        sample = Sample(source, xml.stem, xml, image,
                        int(float(tree.findtext("size/width") or 0)), int(float(tree.findtext("size/height") or 0)))
        for obj in tree.iter("object"):
            name = (obj.findtext("name") or "").strip().lower()
            box = obj.find("bndbox")
            if name not in mapping or box is None:
                sample.dropped[name or "(이름 없음)"] += 1
                continue
            x1, y1, x2, y2 = (float(box.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            sample.boxes.append((mapping[name], x1, y1, x2, y2))
        samples.append(sample)
    return samples


def select(samples: list[Sample], crowd: int, seed: int) -> list[Sample]:
    """군중 사진은 crowd 장만 무작위로, 나머지는 박스가 있는 것 모두"""
    rng = random.Random(seed)
    crowd_pool = [s for s in samples if s.crowd and s.boxes]
    keep_crowd = set(id(s) for s in rng.sample(crowd_pool, min(crowd, len(crowd_pool))))
    return [s for s in samples if s.boxes and (not s.crowd or id(s) in keep_crowd)]


def count(samples: list[Sample]) -> Counter:
    return Counter(c for s in samples for c, *_ in s.boxes)


def clip_scale(boxes, width: int, height: int, scale: float) -> list[tuple[str, float, float, float, float]]:
    """이미지 밖으로 나간 박스를 자르고 줄인 크기에 맞춘다. 너무 작은 박스는 버린다"""
    out = []
    for name, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
        if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
            out.append((name, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)))
    return out


def report(title: str, samples: list[Sample]) -> None:
    c = count(samples)
    helmet, head = c["helmet"], c["head"]
    small = sum(1 for s in samples for _, x1, y1, x2, y2 in s.boxes if max(x2 - x1, y2 - y1) <= 32)
    total = helmet + head
    ratio = f"{helmet / head:.2f}" if head else "-"
    print(f"{title}: 이미지 {len(samples)}장 · helmet {helmet} · head {head} · helmet/head {ratio}"
          f" · 32px 이하 {small / max(total, 1):.0%}")


def build(samples: list[Sample], out: Path, valid_ratio: float, max_side: int, seed: int, args: dict) -> None:
    from PIL import Image

    if out.exists() and any(out.iterdir()):
        sys.exit(f"출력 폴더가 비어 있지 않습니다: {out} (지우거나 다른 폴더를 주세요)")
    rng = random.Random(seed)
    order = samples[:]
    rng.shuffle(order)
    n_valid = max(1, round(len(order) * valid_ratio))
    splits = {"valid": order[:n_valid], "train": order[n_valid:]}
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    cat_id = {c["name"]: c["id"] for c in categories}
    missing, final = [], {}
    for split, items in splits.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for sample in items:
            if sample.image is None:
                missing.append(sample.xml.name)
                continue
            with Image.open(sample.image) as img:
                img = img.convert("RGB")        # EXIF 회전은 적용하지 않는다 (라벨이 원본 화소 기준)
                width, height = img.size
                scale = min(1.0, max_side / max(width, height))
                boxes = clip_scale(sample.boxes, width, height, scale)
                if not boxes:
                    continue
                if scale < 1.0:
                    img = img.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
                name = f"{sample.source}_{sample.stem}.jpg"
                img.save(out / split / name, "JPEG", quality=90)
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": name, "width": img.width, "height": img.height})
            for cls, x1, y1, x2, y2 in boxes:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat_id[cls],
                                    "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)],
                                    "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
        (out / "annotations").mkdir(exist_ok=True)
        (out / "annotations" / f"{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False),
            encoding="utf-8")
        final[split] = {"images": len(images), **Counter(c["name"] for a in annotations
                                                          for c in categories if c["id"] == a["category_id"])}
    meta = {"classes": list(CLASSES), "sources": SOURCES, "args": args, "splits": final,
            "missing_images": missing[:50], "missing_count": len(missing)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    for split, info in final.items():
        print(f"{split}: {info}")
    if missing:
        print(f"이미지 파일을 찾지 못한 라벨 {len(missing)}개 (meta.json 에 일부 기록)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SHWD + Hard Hat Detection → YOLOX COCO (helmet/head)")
    parser.add_argument("--shwd", type=Path, required=True, help="SHWD 를 푼 폴더 (Annotations, JPEGImages 가 아래에 있음)")
    parser.add_argument("--hardhat", type=Path, required=True, help="Hard Hat Detection 을 푼 폴더 (annotations, images)")
    parser.add_argument("--out", type=Path, default=Path("data/helmet-train"))
    parser.add_argument("--crowd", type=int, default=300, help="남길 SHWD 군중 사진(PartA·PartB) 장수")
    parser.add_argument("--valid", type=float, default=0.1, help="검증으로 뗄 비율")
    parser.add_argument("--max-side", type=int, default=1280, help="긴 변을 이 크기 이하로 줄임 (학습 입력 640의 2배)")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--dry-run", action="store_true", help="개수만 보고 파일은 만들지 않음")
    parser.add_argument("--zip", action="store_true", help="만든 뒤 <출력>.zip 으로 묶음 (Drive 에 올릴 파일)")
    args = parser.parse_args(argv)

    samples = read_source("shwd", args.shwd) + read_source("hardhat", args.hardhat)
    dropped = sum((s.dropped for s in samples), Counter())
    report("원본 전체", [s for s in samples if s.boxes])
    report("  SHWD 군중 사진(PartA·PartB)", [s for s in samples if s.crowd and s.boxes])
    report("  SHWD 나머지", [s for s in samples if s.source == "shwd" and not s.crowd and s.boxes])
    report("  Hard Hat", [s for s in samples if s.source == "hardhat" and s.boxes])
    print("버린 라벨:", dict(dropped.most_common()))
    chosen = select(samples, args.crowd, args.seed)
    report(f"고른 것 (군중 {args.crowd}장)", chosen)
    no_image = sum(1 for s in chosen if s.image is None)
    if no_image:
        print(f"주의: 이미지 파일을 찾지 못한 라벨 {no_image}개")
    if args.dry_run:
        return

    build(chosen, args.out, args.valid, args.max_side, args.seed,
          {k: str(v) for k, v in vars(args).items() if k not in ("dry_run", "zip")})
    if args.zip:
        archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
        size = Path(archive).stat().st_size / 1024 / 1024
        print(f"zip: {archive} ({size:.0f}MB)")


if __name__ == "__main__":
    main()
