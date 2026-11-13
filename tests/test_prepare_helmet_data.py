"""안전모 데이터 정리 스크립트: 클래스 맞추기, 군중 사진 고르기, 크기 줄이기, COCO 출력. 작은 가짜 VOC 로 확인."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from scripts.prepare_helmet_data import clip_scale, count, main, read_source, select  # noqa: E402


def _voc(folder: Path, stem: str, objects: list[tuple[str, int, int, int, int]], size=(800, 600),
         ext: str = ".jpg", xml_size: bool = True) -> None:
    ann = folder / ("Annotations" if ext == ".jpg" else "annotations")
    img = folder / ("JPEGImages" if ext == ".jpg" else "images")
    ann.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(img / f"{stem}{ext}")
    w, h = size if xml_size else (0, 0)
    body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
                   f"</bndbox></object>" for n, a, b, c, d in objects)
    (ann / f"{stem}.xml").write_text(f"<annotation><filename>{stem}{ext}</filename><size><width>{w}</width>"
                                     f"<height>{h}</height></size>{body}</annotation>", encoding="utf-8")


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, Path]:
    shwd, hardhat = tmp_path / "SHWD" / "VOC2028", tmp_path / "hardhat"
    _voc(shwd, "000001", [("hat", 10, 10, 60, 60), ("person", 100, 10, 150, 60), ("dog", 1, 1, 9, 9)])
    _voc(shwd, "part1", [("hat", 10, 10, 60, 60)], xml_size=False)
    for i in range(5):                                         # 군중 사진: 맨머리만 여러 개
        _voc(shwd, f"PartA_{i:05d}", [("person", 10 + 20 * k, 10, 28 + 20 * k, 28) for k in range(6)])
    _voc(hardhat, "hard_hat_workers0", [("helmet", 5, 5, 40, 40), ("head", 50, 5, 90, 40), ("person", 0, 0, 200, 400)],
         size=(416, 416), ext=".png")
    _voc(hardhat, "hard_hat_workers1", [("helmet", 300, 300, 500, 500)], size=(416, 416), ext=".png")   # 밖으로 나감
    return tmp_path / "SHWD", hardhat


def test_classes_are_mapped_and_other_labels_dropped(data):
    shwd, hardhat = data
    s = read_source("shwd", shwd)
    h = read_source("hardhat", hardhat)
    assert count(s) == {"helmet": 2, "head": 31}               # hat → helmet, person(머리) → head
    assert count(h) == {"helmet": 2, "head": 1}
    assert sum((x.dropped for x in s + h), start=__import__("collections").Counter()) == {"dog": 1, "person": 1}
    assert all(x.image is not None for x in s + h)               # filename·png 로 이미지 파일을 찾음


def test_crowd_images_are_limited(data):
    shwd, hardhat = data
    samples = read_source("shwd", shwd) + read_source("hardhat", hardhat)
    chosen = select(samples, crowd=2, seed=1)
    assert sum(1 for x in chosen if x.crowd) == 2
    assert sum(1 for x in chosen if not x.crowd) == 4             # 군중 아닌 것은 모두
    assert select(samples, crowd=2, seed=1) == chosen             # 같은 seed 면 같은 결과


def test_clip_scale():
    assert clip_scale([("helmet", 300, 300, 500, 500)], 416, 416, 1.0) == [("helmet", 300.0, 300.0, 416.0, 416.0)]
    assert clip_scale([("head", 10, 10, 30, 50)], 800, 600, 0.5) == [("head", 5.0, 5.0, 15.0, 25.0)]
    assert clip_scale([("head", 10, 10, 11, 50)], 800, 600, 1.0) == []          # 너무 작음


def test_build_writes_coco_and_resizes(data, tmp_path: Path):
    shwd, hardhat = data
    out = tmp_path / "out"
    main(["--shwd", str(shwd), "--hardhat", str(hardhat), "--out", str(out), "--crowd", "1",
          "--valid", "0.3", "--max-side", "400", "--zip"])
    train = json.loads((out / "annotations" / "train.json").read_text(encoding="utf-8"))
    valid = json.loads((out / "annotations" / "valid.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in train["categories"]] == ["helmet", "head"]
    images = train["images"] + valid["images"]
    assert len(images) == 5                                        # 군중 1 + SHWD 2 + Hard Hat 2
    for split, coco in (("train", train), ("valid", valid)):
        for image in coco["images"]:
            assert (out / split / image["file_name"]).is_file()
            assert max(image["width"], image["height"]) <= 400        # 긴 변 400 이하로 줄임
    shwd_img = next(i for i in images if i["file_name"] == "shwd_000001.jpg")
    assert (shwd_img["width"], shwd_img["height"]) == (400, 300)
    anns = train["annotations"] + valid["annotations"]
    assert all(a["bbox"][2] > 0 and a["bbox"][3] > 0 for a in anns)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert "MIT" in meta["sources"]["shwd"] and "CC0" in meta["sources"]["hardhat"]
    assert (tmp_path / "out.zip").is_file()


def test_dry_run_writes_nothing(data, tmp_path: Path, capsys):
    shwd, hardhat = data
    out = tmp_path / "dry"
    main(["--shwd", str(shwd), "--hardhat", str(hardhat), "--out", str(out), "--dry-run"])
    assert not out.exists()
    assert "고른 것" in capsys.readouterr().out
