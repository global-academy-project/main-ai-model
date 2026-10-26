"""OCR 엔진별 CER 측정 스크립트.

폴더 구조:
    samples/
        a.pdf   # 입력 PDF
        a.txt   # 정답 텍스트 (UTF-8, 사람이 검수한 전체 텍스트)

사용 예:
    pip install requests rapidfuzz
    python eval_cer.py --dir samples --engines TESSERACT
    python eval_cer.py --dir samples --engines TESSERACT --strip-spaces

주의:
    - 엔진을 명시해서 호출하므로 모든 페이지가 OCR 경로를 탄다(AUTO는 텍스트 레이어가
      섞여 CER이 왜곡될 수 있음).
    - 서버에서 요약이 동작해야 응답이 오므로 개발 시 ENGINE=FAKE(요약)를 권장.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz.distance import Levenshtein

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


def normalize(
    text: str,
    *,
    form: str = "NFC",
    casefold: bool = True,
    strip_punct: bool = False,
    strip_spaces: bool = False,
) -> str:
    """정답/OCR 결과에 동일하게 적용하는 정규화.

    Args:
        form: 유니코드 정규화 형식. 한글 자모 분리(NFD)를 합치려면 NFC.
        casefold: 영문 대소문자 차이를 무시한다.
        strip_punct: 구두점(유니코드 범주 P*)을 제거한다.
        strip_spaces: 모든 공백을 제거한다. 한국어 띄어쓰기 차이를 배제할 때 사용.
    """
    text = unicodedata.normalize(form, text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\u00a0", " ")
    if casefold:
        text = text.casefold()
    if strip_punct:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"\s+", "" if strip_spaces else " ", text)
    return text.strip()


@dataclass
class CerResult:
    substitutions: int
    deletions: int
    insertions: int
    ref_len: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        return self.errors / self.ref_len

    @property
    def accuracy(self) -> float:
        """1 - CER. 삽입이 많아 음수가 되면 0으로 처리한다."""
        return max(0.0, 1.0 - self.cer)


def compute_cer(reference: str, hypothesis: str) -> CerResult:
    """CER = (S + D + I) / N. N은 정답 문자 수."""
    if not reference:
        raise ValueError("정답 텍스트가 비어 있어 CER을 계산할 수 없습니다.")
    ops = Levenshtein.editops(reference, hypothesis)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for op in ops:
        counts[op.tag] += 1
    return CerResult(counts["replace"], counts["delete"], counts["insert"], len(reference))


def extract_text(url: str, pdf: Path, engine: str, lang: str) -> str:
    """/ocr/v1/extract를 호출하고 페이지 텍스트를 순서대로 합쳐 반환한다."""
    with pdf.open("rb") as f:
        resp = requests.post(
            url,
            files={"file": (pdf.name, f, "application/pdf")},
            data={"engine": engine, "lang": lang, "extract_images": "false"},
            timeout=600,
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"{pdf.name} [{engine}] 실패: {resp.status_code} {resp.url}\n{resp.text[:500]}"
        )
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(f"{pdf.name} [{engine}] 실패: {body.get('error')}")
    pages = sorted(body["data"]["pages"], key=lambda p: p["pageNo"])
    return "\n\n".join(p["text"] for p in pages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000/api/v1/ocr/extract")
    ap.add_argument("--engines", nargs="+", default=["TESSERACT"])
    ap.add_argument("--lang", default="ko,en")
    ap.add_argument("--strip-punct", action="store_true")
    ap.add_argument("--strip-spaces", action="store_true")
    args = ap.parse_args()

    norm = lambda s: normalize(  # noqa: E731
        s, strip_punct=args.strip_punct, strip_spaces=args.strip_spaces
    )
    pdfs = sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("PDF 샘플이 없습니다.")

    for engine in args.engines:
        results: list[tuple[str, CerResult]] = []
        for pdf in pdfs:
            gt_path = pdf.with_suffix(".txt")
            if not gt_path.exists():
                print(f"[skip] 정답 없음: {gt_path.name}")
                continue
            ref = norm(gt_path.read_text(encoding="utf-8"))
            hyp = norm(extract_text(args.url, pdf, engine, args.lang))
            results.append((pdf.name, compute_cer(ref, hyp)))

        print(f"\n== {engine} ==")
        print(f"{'file':30} {'CER':>8} {'acc':>8} {'S':>6} {'D':>6} {'I':>6} {'N':>7}")
        for name, r in results:
            print(
                f"{name:30} {r.cer:8.4f} {r.accuracy:8.2%} "
                f"{r.substitutions:6d} {r.deletions:6d} {r.insertions:6d} {r.ref_len:7d}"
            )
        if results:
            macro = sum(r.cer for _, r in results) / len(results)
            micro = sum(r.errors for _, r in results) / sum(r.ref_len for _, r in results)
            print(f"macro CER(문서 평균) = {macro:.4f} / micro CER(문자 합산) = {micro:.4f}")


if __name__ == "__main__":
    main()
