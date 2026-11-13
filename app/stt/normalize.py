"""받아 적은 글의 표기 맞추기 (규칙, 모델 없음). 측정값 문장이 추출(1.7B·사전)에 잘 들어가게 한다.

- 한글로 적힌 숫자 → 아라비아 숫자: 측정 항목 이름 바로 뒤이거나 단위(퍼센트·프로·피피엠·%·ppm) 바로 앞일 때만 바꾼다.
  ("이십 점 구 퍼센트" → "20.9%", "산소 이십점구" → "산소 20.9", "가연성가스 영" → "가연성가스 0")
  그 밖의 한글은 건드리지 않는다 ("이"·"오" 같은 글자가 조사·낱말에 많아서).
- 단위: 퍼센트·프로 → %, 피피엠(피피엘·피피엔으로 잘못 들은 것도) → ppm, 엘이엘 → LEL
- 사람 목소리 시험(11/13, 12개)에서 본 표기 고치기 (단위 바로 앞 숫자만):
  "05%"·"03 ppm" (영 점 오 의 점이 빠짐) → "0.5%"·"0.3ppm", "20% 9%" (이십 점 구 사이에 % 가 끼어듦) → "20.9%"
값을 지어내지 않는다. 바꿀 수 없으면 그대로 둔다 (사람이 입력칸에서 확인·수정한다).
"""

from __future__ import annotations

import re

_DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_UNITS = {"십": 10, "백": 100, "천": 1000}
_NUM = r"[영공일이삼사오육칠팔구십백천]+(?:\s*점\s*[영공일이삼사오육칠팔구]+)?"
_ITEMS = r"(?:산소|탄산가스|이산화탄소|일산화탄소|황화수소|가연성\s*가스|엘\s*이\s*엘|LEL|암모니아|염소)"
_PPM = r"피\s*피\s*[엠엘엔]"   # 피피엠을 피피엘·피피엔으로 듣는 경우가 있다
_UNIT_WORDS = rf"(?:퍼센트|프로|{_PPM}|%|ppm)"


def korean_number(text: str) -> str | None:
    """한자어 숫자 → 숫자 글자 ("이십점구" → "20.9", "십칠" → "17", "영" → "0"). 숫자가 아니면 None."""
    raw = re.sub(r"\s+", "", text)
    whole, _, frac = raw.partition("점")
    if not whole or any(c not in _DIGITS and c not in _UNITS for c in whole):
        return None
    total, digit = 0, None
    for c in whole:
        if c in _DIGITS:
            if digit is not None:      # "이삼" 처럼 자리 없이 이어지면 숫자로 보지 않는다
                return None
            digit = _DIGITS[c]
        else:
            total += (digit if digit is not None else 1) * _UNITS[c]
            digit = None
    if digit is not None:
        total += digit
    if frac:
        if any(c not in _DIGITS for c in frac):
            return None
        return f"{total}." + "".join(str(_DIGITS[c]) for c in frac)
    return str(total)


def normalize(text: str) -> str:
    """받아 적은 글 → 측정 문장 표기."""
    def by_unit(m: re.Match) -> str:
        n = korean_number(m.group(1))
        return f"{n}{m.group(2)}" if n is not None else m.group(0)

    def by_item(m: re.Match) -> str:
        n = korean_number(m.group(2))
        return f"{m.group(1)}{n}" if n is not None else m.group(0)

    out = re.sub(rf"({_ITEMS}\s*)({_NUM})(?![가-힣])", by_item, text)
    out = re.sub(rf"({_NUM})\s*({_UNIT_WORDS})", by_unit, out)
    out = re.sub(r"(\d)\s*(?:퍼센트|프로)", r"\1%", out)
    out = re.sub(rf"(\d)\s*{_PPM}", r"\1ppm", out)
    out = re.sub(r"(\d)\s+(%|ppm)", r"\1\2", out)                     # "5 ppm" → "5ppm"
    out = re.sub(r"(?<![\d.])0(\d)(%|ppm)", r"0.\1\2", out)             # "05%" → "0.5%" (점이 빠짐)
    out = re.sub(r"(?<![\d.])(\d{1,2})%\s*(\d)%", r"\1.\2%", out)         # "20% 9%" → "20.9%"
    out = re.sub(r"엘\s*이\s*엘", "LEL", out)
    return re.sub(r"\s+", " ", out).strip()
