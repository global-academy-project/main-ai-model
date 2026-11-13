"""TBM 사진 안전모 평가 스크립트: 정답 csv 읽기, 기준값 조합별 채점, 틀린 사진 그리기. 가짜 세션으로 돈다 (모델 파일 없이)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.vision.yolox import HELMET_LABELS, YoloxDetector
from scripts.eval_helmet_photos import Result, Score, Truth, main, read_truth
from tests.test_vision import FakeSession, encode, jpeg
from tests.test_vision_helmet import HSIZE, encode_heads


def _truth(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8-sig")   # 엑셀처럼 BOM 붙여 저장
    return path


def test_read_truth(tmp_path: Path):
    truths = read_truth(_truth(tmp_path / "t.csv", "file,people,missing\na.jpg,5,1\n b.jpg ,3,0\n"))
    assert truths == [Truth("a.jpg", 5, 1), Truth("b.jpg", 3, 0)]
    with pytest.raises(ValueError, match="file,people,missing"):
        read_truth(_truth(tmp_path / "bad.csv", "name,count\na.jpg,1\n"))
    with pytest.raises(ValueError, match="3번째 줄"):
        read_truth(_truth(tmp_path / "bad2.csv", "file,people,missing\na.jpg,1,0\nb.jpg,두명,0\n"))
    with pytest.raises(ValueError, match="missing 이 people 보다"):
        read_truth(_truth(tmp_path / "bad3.csv", "file,people,missing\na.jpg,1,2\n"))


def test_score_counts_missed_and_false_alarm():
    s = Score()
    s.add(Result(Truth("a", 3, 1), 3, 0, []))   # 놓침
    s.add(Result(Truth("b", 3, 0), 2, 1, []))   # 잘못 잡음
    s.add(Result(Truth("c", 2, 1), 2, 1, []))   # 맞음
    assert (s.photos, s.people_exact, s.missing_exact, s.missed, s.false_alarm, s.people_error) == (3, 2, 1, 1, 1, 1)


def _detectors() -> tuple[YoloxDetector, YoloxDetector]:
    """사람 박스 없음. 머리 3개: 안전모 0.9, 맨머리 0.45 (맨머리 기준 0.4 이하에서만 셈), 맨머리 0.8"""
    person = YoloxDetector(None, session=FakeSession(encode([])))
    helmet = YoloxDetector(None, size=HSIZE, session=FakeSession(encode_heads([
        (100, 100, 40, 40, 0.9, 0), (300, 100, 40, 40, 0.45, 1), (500, 100, 40, 40, 0.8, 1)])),
        labels=HELMET_LABELS, thresholds=(0.4, 0.5))
    return person, helmet


def test_main_compares_thresholds_and_lists_wrong_photos(tmp_path: Path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "extra.jpg"):
        (photos / name).write_bytes(jpeg(416, 416))
    truth = _truth(tmp_path / "truth.csv", "file,people,missing\n"
                   "a.jpg,2,1\n"        # 지금 설정(0.4, 0.5)에서 맞음
                   "b.jpg,3,2\n"        # 맨머리 기준 0.4 이하에서만 맞음
                   "c.jpg,2,0\n"        # 전원 착용인데 미착용이 나옴 (잘못 잡음)
                   "gone.jpg,1,0\n")    # 폴더에 없음
    person, helmet = _detectors()
    out = tmp_path / "result.csv"
    code = main([str(photos), str(truth), "--helmet-thresholds", "0.4", "--head-thresholds", "0.4,0.5",
                 "--draw", str(tmp_path / "drawn"), "--out", str(out)], person=person, helmet=helmet)
    text = capsys.readouterr().out
    assert code == 0
    assert "폴더에 없는 파일 1개" in text and "gone.jpg" in text
    assert "정답이 없는 사진 1개" in text and "extra.jpg" in text
    assert "사진 3장" in text
    lines = {line.split("|")[0].split()[1]: line for line in text.splitlines() if line.strip().startswith("0.4 ")}
    # 맨머리 0.4: 모두 3명/미착용 2 → b 만 맞음. 0.5: 모두 2명/미착용 1 → a 만 맞음. 둘 다 c 는 잘못 잡음
    assert lines["0.4"].split("|")[2].split() == ["1/3", "0", "1"]
    assert lines["0.5"].split("|")[2].split()[:3] == ["1/3", "0", "1"]
    assert "지금 설정" in lines["0.5"] and "가장 좋음" in lines["0.5"]   # 점수가 같으면 지금 설정
    assert "가장 좋음" not in lines["0.4"]
    assert "미착용 있는 사진 2장 중 경고 2장 · 전원 착용 1장 중 잘못 경고 1장" in text
    assert "틀린 사진 2장" in text and "b.jpg: 정답 3명/미착용 2 → 결과 2명/미착용 1" in text
    assert sorted(p.name for p in (tmp_path / "drawn").iterdir()) == ["b.jpg", "c.jpg"]
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert [(r["file"], r["got_people"], r["got_missing"]) for r in rows] == [
        ("a.jpg", "2", "1"), ("b.jpg", "2", "1"), ("c.jpg", "2", "1")]


def test_main_without_photos(tmp_path: Path, capsys):
    person, helmet = _detectors()
    (tmp_path / "photos").mkdir()
    truth = _truth(tmp_path / "truth.csv", "file,people,missing\na.jpg,1,0\n")
    assert main([str(tmp_path / "photos"), str(truth)], person=person, helmet=helmet) == 2
    assert "채점할 사진이 없습니다" in capsys.readouterr().err


def test_people_only_counts_person_boxes_without_helmet_model(tmp_path: Path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (photos / name).write_bytes(jpeg(416, 416))
    truth = _truth(tmp_path / "truth.csv", "file,people,missing\na.jpg,2,1\nb.jpg,3,0\n")
    person = YoloxDetector(None, session=FakeSession(encode([(60, 200, 60, 300, 0.9), (300, 200, 60, 300, 0.9)])))
    code = main([str(photos), str(truth), "--people-only", "--out", str(tmp_path / "r.csv")], person=person, helmet=None)
    text = capsys.readouterr().out
    assert code == 0
    assert "사람 모델만" in text and "인원 맞음 1/2" in text and "평균 차이 0.50명" in text
    assert "미착용 있는 사진 1장 중 경고 0장" in text
    assert "b.jpg: 정답 3명 → 결과 2명" in text and "a.jpg: 정답" not in text
    rows = list(csv.DictReader((tmp_path / "r.csv").open(encoding="utf-8-sig")))
    assert [(r["got_people"], r["got_missing"]) for r in rows] == [("2", "0"), ("2", "0")]
