"""우리 TBM 사진으로 인원·안전모 미착용 수를 채점하고 기준값을 비교한다 (C, 11/13). 사진·정답은 repo 에 넣지 않는다.

    python -m scripts.eval_helmet_photos <사진 폴더> <정답 csv> [--draw <폴더>] [--out <결과 csv>]

정답 csv (utf-8, 엑셀 저장도 됨):
    file,people,missing
    IMG_0001.jpg,5,0
    IMG_0002.jpg,4,1

- 모델·입력 크기·NMS 는 .env 설정(VISION_MODEL_PATH, VISION_HELMET_MODEL_PATH ...)을 그대로 쓴다. --model, --helmet-model 로 바꿀 수 있다.
- 사진마다 모델은 한 번만 돌리고, 기준값 조합(안전모 × 맨머리)마다 박스를 다시 골라 센다.
  인원·미착용 계산은 서버와 같은 함수(detect_tbm_photo)를 쓴다.
- 조합마다 보여 주는 것
    인원 맞음      사람 수가 정답과 같은 사진 수
    미착용 맞음    미착용 수가 정답과 같은 사진 수
    놓침          미착용자가 있는데 0명으로 나온 사진 (가장 위험)
    잘못 잡음     전원 착용인데 미착용이 나온 사진 (화면에 쓸데없는 UNSURE)
- 사진 단위 경고: 미착용 있는 사진 중 경고(미착용 1명 이상)가 뜬 수, 전원 착용인데 경고가 뜬 수
- --people-only: 안전모 모델 없이 사람 모델만 (학습 전, VISION_HELMET_MODEL_PATH 가 비었을 때와 같은 계산). 인원만 채점하고 경고는 0
- 지금 설정(기본) 조합에서 틀린 사진을 나열하고, --draw 를 주면 박스를 그린 사진을 저장한다 (초록 안전모, 빨강 미착용, 파랑 사람).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.vision_service import detect_tbm_photo
from app.vision.yolox import HELMET_LABELS, Box, YoloxDetector, class_boxes, letterbox, load_image

HELMET_THRESHOLDS = (0.3, 0.4, 0.5)
HEAD_THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
COLORS = {"helmet": (40, 180, 70), "no_helmet": (220, 40, 40), "person": (40, 110, 230)}

Combo = tuple[float, float]   # (안전모 기준, 맨머리 기준)


@dataclass(frozen=True)
class Truth:
    file: str
    people: int
    missing: int


@dataclass
class Detected:
    truth: Truth
    data: bytes
    persons: list[Box]
    heads_at: Callable[[Combo], list[Box]] | None   # 사람 모델만이면 None
    seconds: float


@dataclass
class Result:
    truth: Truth
    people: int
    missing: int
    boxes: list[dict]


@dataclass
class Score:
    photos: int = 0
    people_exact: int = 0
    missing_exact: int = 0
    missed: int = 0          # 미착용자가 있는데 0명
    false_alarm: int = 0     # 전원 착용인데 미착용
    people_error: int = 0    # |정답 - 결과| 합

    def add(self, r: Result) -> None:
        self.photos += 1
        self.people_exact += r.people == r.truth.people
        self.missing_exact += r.missing == r.truth.missing
        self.missed += r.truth.missing > 0 and r.missing == 0
        self.false_alarm += r.truth.missing == 0 and r.missing > 0
        self.people_error += abs(r.people - r.truth.people)

    def rank(self) -> tuple:
        """좋은 순: 놓침 적게 → 잘못 잡음 적게 → 미착용 맞음 많이 → 인원 맞음 많이"""
        return (self.missed, self.false_alarm, -self.missing_exact, -self.people_exact)


class _Fixed:
    """이미 찾은 박스를 돌려주는 검출기 (서버와 같은 계산 함수에 넣기 위해)"""

    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def detect(self, _data: bytes) -> list[Box]:
        return self.boxes


def read_truth(path: Path) -> list[Truth]:
    with path.open(encoding="utf-8-sig", newline="") as f:   # 엑셀이 붙이는 BOM 도 읽음
        rows = list(csv.DictReader(f))
    if not rows or not {"file", "people", "missing"} <= set(rows[0]):
        raise ValueError(f"{path}: 첫 줄이 file,people,missing 이어야 합니다")
    truths = []
    for n, row in enumerate(rows, start=2):
        try:
            truth = Truth(row["file"].strip(), int(row["people"]), int(row["missing"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} {n}번째 줄: people·missing 은 숫자여야 합니다 ({row})") from exc
        if truth.missing > truth.people:
            raise ValueError(f"{path} {n}번째 줄: missing 이 people 보다 큽니다 ({row})")
        truths.append(truth)
    return truths


def detect(person: YoloxDetector, helmet: YoloxDetector | None, truth: Truth, data: bytes) -> Detected:
    """사람 모델·안전모 모델을 한 번씩 돌린다. 안전모는 격자 출력을 두고 기준값마다 다시 고른다"""
    started = time.perf_counter()
    persons = person.detect(data)
    if helmet is None:
        return Detected(truth, data, persons, None, time.perf_counter() - started)
    session = helmet._get_session()
    image = load_image(data)
    tensor, ratio = letterbox(image, helmet.size)
    raw = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    seconds = time.perf_counter() - started
    width, height = image.size

    def heads_at(combo: Combo) -> list[Box]:
        return class_boxes(raw, size=helmet.size, ratio=ratio, width=width, height=height, labels=HELMET_LABELS,
                           thresholds=combo, nms_threshold=helmet.nms_threshold)

    return Detected(truth, data, persons, heads_at, seconds)


def count(d: Detected, combo: Combo | None) -> Result:
    heads = _Fixed(d.heads_at(combo)) if d.heads_at is not None else None
    data = detect_tbm_photo(_Fixed(d.persons), d.data, heads).model_dump(mode="json")["data"]
    return Result(d.truth, data["people"], data["helmet_missing"] or 0, data["boxes"])   # 안전모 모델 없으면 경고 없음


def evaluate(detected: list[Detected], combos: list[Combo | None]) -> dict[Combo | None, tuple[Score, list[Result]]]:
    out = {}
    for combo in combos:
        score, results = Score(), []
        for d in detected:
            r = count(d, combo)
            score.add(r)
            results.append(r)
        out[combo] = (score, results)
    return out


def draw(path: Path, data: bytes, r: Result) -> None:
    from PIL import ImageDraw

    image = load_image(data)
    w, h = image.size
    pen = ImageDraw.Draw(image)
    line = max(2, round(max(w, h) / 400))
    for b in r.boxes:
        x1, y1 = b["x"] * w, b["y"] * h
        color = COLORS.get(b["label"], (0, 0, 0))
        pen.rectangle([x1, y1, x1 + b["w"] * w, y1 + b["h"] * h], outline=color, width=line)
        pen.text((x1 + 2, max(0, y1 - 12)), f"{b['label']} {b['score']:.2f}", fill=color)
    pen.text((8, 8), f"truth {r.truth.people} people / {r.truth.missing} missing   "   # 기본 글꼴은 한글이 안 나옴
                     f"got {r.people} / {r.missing}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path.with_suffix(".jpg"), "JPEG", quality=90)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def main(argv: list[str] | None = None, *, person: YoloxDetector | None = None,
         helmet: YoloxDetector | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", type=Path, help="사진 폴더")
    parser.add_argument("truth", type=Path, help="정답 csv (file,people,missing)")
    parser.add_argument("--model", help="사람 모델 (기본 VISION_MODEL_PATH)")
    parser.add_argument("--helmet-model", help="안전모 모델 (기본 VISION_HELMET_MODEL_PATH)")
    parser.add_argument("--helmet-thresholds", type=_floats, default=HELMET_THRESHOLDS, help="예: 0.3,0.4,0.5")
    parser.add_argument("--head-thresholds", type=_floats, default=HEAD_THRESHOLDS, help="예: 0.3,0.4,0.5,0.6")
    parser.add_argument("--draw", type=Path, help="틀린 사진에 박스를 그려 저장할 폴더 (지금 설정 기준)")
    parser.add_argument("--draw-all", action="store_true", help="--draw 때 맞은 사진도 저장")
    parser.add_argument("--out", type=Path, help="사진별 결과 csv (지금 설정 기준)")
    parser.add_argument("--people-only", action="store_true", help="안전모 모델 없이 사람 모델만 (학습 전 비교용)")
    args = parser.parse_args(argv)

    if person is None or (helmet is None and not args.people_only):   # 테스트는 가짜 검출기를 넘긴다
        from app.api.v1.agent.vision import get_detector, get_helmet_detector
        from app.core import get_settings

        settings = get_settings()
        if args.model:
            settings = settings.model_copy(update={"vision_model_path": args.model})
        if args.helmet_model:
            settings = settings.model_copy(update={"vision_helmet_model_path": args.helmet_model})
        person = person or get_detector(settings)
        helmet = helmet or get_helmet_detector(settings)
    if args.people_only:
        helmet = None
    elif helmet is None:
        print("안전모 모델이 없습니다: .env 의 VISION_HELMET_MODEL_PATH 또는 --helmet-model", file=sys.stderr)
        return 2
    current: Combo | None = None if helmet is None else (float(helmet.thresholds[0]), float(helmet.thresholds[1]))

    truths = read_truth(args.truth)
    files = {p.name for p in args.photos.iterdir() if p.is_file()} if args.photos.is_dir() else set()
    absent = [t.file for t in truths if t.file not in files]
    if absent:
        print(f"사진 폴더에 없는 파일 {len(absent)}개 (빼고 채점): {', '.join(absent[:10])}")
    unlabeled = sorted(f for f in files - {t.file for t in truths} if Path(f).suffix.lower() in (".jpg", ".jpeg", ".png"))
    if unlabeled:
        print(f"정답이 없는 사진 {len(unlabeled)}개 (채점 안 함): {', '.join(unlabeled[:10])}")
    truths = [t for t in truths if t.file in files]
    if not truths:
        print("채점할 사진이 없습니다", file=sys.stderr)
        return 2

    detected = []
    for i, t in enumerate(truths, start=1):
        detected.append(detect(person, helmet, t, (args.photos / t.file).read_bytes()))
        print(f"\r인식 {i}/{len(truths)}", end="", flush=True)
    print()
    seconds = [d.seconds for d in detected]
    print(f"사진 {len(detected)}장 · 정답 인원 {sum(t.people for t in truths)}명 · 정답 미착용 "
          f"{sum(t.missing for t in truths)}명 (미착용 있는 사진 {sum(t.missing > 0 for t in truths)}장)")
    models = "사람 모델만" if helmet is None else "사람+안전모"
    print(f"장당 {np.mean(seconds):.2f}초 (최대 {max(seconds):.2f}초, {models}, 이 PC CPU)")
    n = len(detected)
    with_missing = sum(t.missing > 0 for t in truths)

    if helmet is None:
        score, people_results = evaluate(detected, [None])[None]
        print(f"\n[사람 모델만] 인원 맞음 {score.people_exact}/{n} · 평균 차이 {score.people_error / n:.2f}명 · 안전모 판단 안 함")
        print(f"사진 단위 경고: 미착용 있는 사진 {with_missing}장 중 경고 0장 · 전원 착용 {n - with_missing}장 중 잘못 경고 0장")
        wrong = [r for r in people_results if r.people != r.truth.people]
        print(f"\n인원이 틀린 사진 {len(wrong)}장")
        for r in wrong:
            print(f"  {r.truth.file}: 정답 {r.truth.people}명 → 결과 {r.people}명")
        _save(args, detected, people_results, wrong)
        return 0

    combos = sorted(set(itertools.product(args.helmet_thresholds, args.head_thresholds)) | {current})
    results = evaluate(detected, combos)
    best = min(combos, key=lambda c: (results[c][0].rank(), c != current))   # 같으면 지금 설정
    print(f"\n{'안전모':>6} {'맨머리':>6} | {'인원 맞음':>8} {'평균 차이':>8} | {'미착용 맞음':>10} {'놓침':>5} {'잘못 잡음':>8}")
    for combo in combos:
        s = results[combo][0]
        mark = " ← 지금 설정" if combo == current else ""
        mark += " ← 가장 좋음" if combo == best else ""
        print(f"{combo[0]:>6} {combo[1]:>6} | {s.people_exact:>4}/{n:<3} {s.people_error / n:>8.2f} | "
              f"{s.missing_exact:>6}/{n:<3} {s.missed:>5} {s.false_alarm:>8}{mark}")
    print("좋음 기준: 놓침 적게 → 잘못 잡음 적게 → 미착용 맞음 많이 → 인원 맞음 많이 (같으면 지금 설정)")

    s = results[current][0]
    print(f"\n사진 단위 경고 (지금 설정 {current}): 미착용 있는 사진 {with_missing}장 중 경고 {with_missing - s.missed}장 · "
          f"전원 착용 {n - with_missing}장 중 잘못 경고 {s.false_alarm}장")
    wrong = [r for r in results[current][1] if (r.people, r.missing) != (r.truth.people, r.truth.missing)]
    print(f"\n지금 설정 {current} 에서 틀린 사진 {len(wrong)}장")
    for r in wrong:
        print(f"  {r.truth.file}: 정답 {r.truth.people}명/미착용 {r.truth.missing} → 결과 {r.people}명/미착용 {r.missing}")
    _save(args, detected, results[current][1], wrong)
    return 0


def _save(args: argparse.Namespace, detected: list[Detected], results: list[Result], wrong: list[Result]) -> None:
    """--draw 박스 사진, --out 사진별 결과"""
    if args.draw:
        targets = results if args.draw_all else wrong
        by_file = {d.truth.file: d.data for d in detected}
        for r in targets:
            draw(args.draw / r.truth.file, by_file[r.truth.file], r)
        print(f"박스 그린 사진 {len(targets)}장 → {args.draw}")
    if args.out:
        with args.out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "people", "missing", "got_people", "got_missing", "seconds"])
            for r, d in zip(results, detected):
                writer.writerow([r.truth.file, r.truth.people, r.truth.missing, r.people, r.missing, f"{d.seconds:.3f}"])
        print(f"사진별 결과 → {args.out}")


if __name__ == "__main__":
    sys.exit(main())
