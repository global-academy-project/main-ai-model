from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


class PaddleEngine:
    name = "PADDLE"

    def __init__(self) -> None:
        self._lock = threading.Lock()            # Paddle은 직렬화
        self._models: dict[str, object] = {}     # PaddleOCR는 lang에 묶이므로 lang별 캐시

    @staticmethod
    def _to_paddle_lang(lang: list[str]) -> str:
        # korean 모델은 영문도 인식한다. ko가 있으면 korean 하나로 충분.
        if "ko" in lang:
            return "korean"
        if "en" in lang:
            return "en"
        return "korean"                          # 모르는 값은 무시하고 기본값

    def _get_model(self, paddle_lang: str):
        model = self._models.get(paddle_lang)
        if model is None:
            from paddleocr import PaddleOCR      # 지연 import (여기서 처음 로딩)
            model = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",   # 한국어 인식
                text_det_limit_side_len=960,
                text_det_limit_type="max",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            self._models[paddle_lang] = model
        return model

    def recognize(self, image: Image, *, lang: list[str]) -> tuple[str, float | None]:
        paddle_lang = self._to_paddle_lang(lang)
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1]   # RGB → BGR (OpenCV 규약)

        with self._lock:
            model = self._get_model(paddle_lang)
            results = model.predict(arr)

        texts: list[str] = []
        scores: list[float] = []
        for res in results:
            texts.extend(res["rec_texts"])
            scores.extend(float(s) for s in res["rec_scores"])

        if not texts:
            return "", None
        # Paddle 점수는 이미 0~1이므로 정규화 불필요
        return "\n".join(texts), sum(scores) / len(scores)