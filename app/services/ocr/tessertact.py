import pytesseract
from pytesseract import Output


class TesseractEngine:
    name = "TESSERACT"
    _LANG_MAP = {"ko": "kor", "en": "eng"}

    def __init__(self) -> None:
        # 바이너리가 없으면 요청 처리 중이 아니라 엔진 생성 시점에 바로 실패
        pytesseract.get_tesseract_version()

    def recognize(self, image, *, lang: list[str]) -> tuple[str, float | None]:
        tess_lang = "+".join(self._LANG_MAP.get(code, code) for code in lang) or "eng"
        data = pytesseract.image_to_data(image, lang=tess_lang, output_type=Output.DICT)

        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:  # conf -1은 단어가 아닌 구조 행
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf)

        text = "\n".join(" ".join(words) for words in lines.values())
        confidence = (sum(confs) / len(confs)) / 100 if confs else None  # 0~100 → 0~1
        return text, confidence