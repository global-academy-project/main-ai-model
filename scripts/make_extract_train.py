"""문장 추출 어댑터(extract LoRA) 학습 데이터 생성기 (C 담당). 표준 라이브러리만 쓴다.

    python -m scripts.make_extract_train              # data/extract-train/ 에 train.jsonl 3000 + valid.jsonl 200
    python -m scripts.make_extract_train --sample 3   # 유형별 3건씩 화면에만 (저장 안 함)

- 저장 위치 기본은 프로젝트 안 data/extract-train/ (.gitignore·.dockerignore 의 /data/ 로 git·이미지에 안 들어감).
  생성기로 언제든 같은 데이터를 다시 만들 수 있어서 데이터는 올리지 않는다. 학습한 어댑터는 adapters/extract/ 에 둔다.
- 한 줄 = {"messages": [system, user, assistant], "tag": 유형}. system·user 는 실행과 같은 문구
  (app/agent/rule_input_prompt.py 의 ADAPTER_SYSTEM_PROMPT·build_adapter_messages).
  assistant 는 줄 형식 (11/12, app/agent/rule_input_parse.py 의 format_lines. 예: "O2|20.3|PCT|산소 20.3%").
- 11/12 보강 (gold-v2 오답 유형, gold 문장은 베끼지 않음): 밀폐 "진입·들어가기 전", 한 덩어리 두 종류
  (탱크 내부 용접·비계 위 용접), 밀폐 아닌 탱크 주변 작업, 화재감시자 + 용접, 천장 고소, "정상" 같은 숫자 아닌 값과
  실제 값이 함께, "19.9%(산소)" 어순, "다시 20.5%" 재측정, 일반 낱말과 겹치는 성(한·이·정 …)
- 정답 기준 (11/11, eval/rule_input_extract_v2.json 과 같음): 적힌 대로 뽑는다. 판단은 Spring 이 한다.
    단위 없음 → unit null / 다른 단위 → 적힌 단위 그대로 / 같은 항목 두 번 → 둘 다 / 측정 아닌 숫자·값 없는 언급 → 뽑지 않음
    감시자: 이름이 앞이든 뒤든 이름만. "미정·없음·없이·배치 필요" 는 null
- 문장과 정답을 같이 만든다 → 근거(evidence)는 항상 원문에서 잘라 넣은 그대로다. 한 건씩 검사하고 어긋나면 멈춘다.
- 평가 정답(eval/rule_input_extract_v2.json, --gold 로 더 줄 수 있음)과 같거나 거의 같은 문장은 버린다 (평가가 외운 것이 되지 않게).
- 같은 --seed 면 언제 돌려도 같은 데이터가 나온다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.rule_input_parse import format_lines
from app.agent.rule_input_prompt import build_adapter_messages

ROOT = Path(__file__).resolve().parents[1]
GOLD_V2 = ROOT / "eval" / "rule_input_extract_v2.json"
DEFAULT_OUT = ROOT / "data" / "extract-train"

# ---------------------------------------------------------------- 재료

# 측정 항목: 부르는 말(사전 dictionary.json 의 names 와 같은 표기), 영문, 값 범위, 소수 자리, 기준 단위
ITEMS: dict[str, dict] = {
    "O2": {"names": ["산소", "산소 농도", "산소농도"], "abbr": "O2", "lo": 15.0, "hi": 23.5, "dec": (0, 1, 1), "unit": "PCT"},
    "CO2": {"names": ["이산화탄소", "탄산가스"], "abbr": "CO2", "lo": 0.1, "hi": 3.0, "dec": (1, 1, 0), "unit": "PCT"},
    "CO": {"names": ["일산화탄소"], "abbr": "CO", "lo": 0, "hi": 60, "dec": (0, 0, 0), "unit": "PPM"},
    "H2S": {"names": ["황화수소"], "abbr": "H2S", "lo": 0, "hi": 20, "dec": (0, 0, 1), "unit": "PPM"},
    "LEL": {"names": ["가연성가스", "가연성 가스", "인화성 가스", "인화성가스"], "abbr": "LEL", "lo": 0, "hi": 15,
            "dec": (0, 0, 0), "unit": "PCT_LEL"},
    "NH3": {"names": ["암모니아"], "abbr": "NH3", "lo": 0, "hi": 50, "dec": (0, 0, 0), "unit": "PPM"},
    "CL2": {"names": ["염소", "염소가스"], "abbr": "CL2", "lo": 0.1, "hi": 2.0, "dec": (1, 1, 0), "unit": "PPM"},
}
# 단위 표기 → 단위 코드
UNIT_WORDS: dict[str, list[str]] = {
    "PCT": ["%", "%", "%", " %", "퍼센트", " 퍼센트", "프로", " 프로"],
    "PPM": ["ppm", "ppm", "ppm", " ppm", "피피엠", " 피피엠"],
    "PCT_LEL": ["%LEL", "%LEL", "% LEL"],
}
WRONG_UNIT = {"O2": "PPM", "CO2": "PPM", "CO": "PCT", "H2S": "PCT", "NH3": "PCT", "CL2": "PCT"}

SURNAMES = list("김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노하곽성차주우구민류")
# 일반 낱말과 겹치는 성 ("감시자 한다은" 을 "다은" 으로 뽑는 실수, 11/12 검증). 이름의 30% 는 여기서 뽑는다
AMBIGUOUS_SURNAMES = list("한이정고하오우주")
GIVEN = ["민준", "서연", "도현", "하은", "지호", "예준", "수빈", "시우", "지민", "현우", "유진", "건우", "다은", "승민",
         "채원", "준서", "지아", "태윤", "소율", "우진", "가은", "민재", "서준", "윤아", "정우", "하준", "나연", "동현",
         "은서", "재원", "수아", "성민", "예린", "상훈", "보람", "영수", "미경", "광수", "진영", "혜진", "경민", "종현"]

HOT_OBJ = ["배관", "철골", "앵글", "플랜지", "브래킷", "지지대", "철판", "덕트 브래킷", "배관 서포트", "난간"]
HOT_ACT = ["{o} 용접", "{o} 아크용접", "{o} 용단", "{o} 가스절단", "그라인더로 {o} 절단", "{o} 그라인더 작업",
           "토치로 {o} 가열", "{o} 산소절단", "화기작업", "용접 작업"]
# 대상 없이 짧게 ("용접 시작"). 한 덩어리 두 종류 문장에도 쓴다
HOT_BARE = ["용접", "용접 작업", "용접 시작", "용단", "용단 작업", "가스절단", "그라인더 작업", "아크용접", "배관 용접",
            "보강판 용접"]
CONFINED_PLACE = ["{n}번 탱크 내부", "탱크 안", "맨홀", "반응기 내부", "피트 안", "정화조 내부", "저장조 내부", "밀폐공간",
                  "{n}호기 탱크 안", "폐수 피트 내부"]
CONFINED_PLACE += ["집수정 안", "{n}번 저수조 안", "맨홀 안쪽", "반응조 내부", "침전조 안", "{n}호 정화조", "오수 피트",
                   "저류조 내부", "{n}번 반응기", "사일로 내부"]
CONFINED_ACT = ["{p} 청소", "{p} 점검", "{p} 들어가서 도장", "{p} 진입", "{p} 슬러지 제거", "{p} 들어가서 점검", "{p} 보수",
                "{p} 들어가기 전", "{p} 진입 전", "{p} 들어가서 작업", "{p} 진입해서 청소", "{p} 내려가서 점검"]
# 밀폐 장소 이름이 있지만 밖에서 하는 작업 → 밀폐공간 아님
NEAR_CONFINED = ["{n}번 탱크 외벽 도장", "탱크 바깥 배관 점검", "맨홀 뚜껑 교체", "정화조 주변 청소", "저수조 밖에서 대기",
                 "반응기 외부 보온재 교체", "피트 옆 자재 정리", "탱크 상부 계단 점검", "집수정 앞 배수로 청소"]
HEIGHT_ACT = ["비계 위에서 {t}", "사다리 타고 {t}", "고소작업대 타고 {t}", "지붕 위 {t}", "달비계에서 {t}", "높이 {h}m {t}",
              "{h}m 높이에서 {t}"]
HEIGHT_ACT += ["{h}m 천장 {t}", "천장 {h}m 높이에서 {t}"]
HEIGHT_TASK = ["도장", "배선", "조명 교체", "덕트 설치", "판넬 보수", "밸브 교체", "보온재 교체", "점검",
               "외벽 도장", "케이블 트레이 설치"]
# 한 덩어리 두 종류 ("비계 위에서 용접")에 쓰는 고소 자리
HEIGHT_PLACE = ["비계 위", "사다리 위", "고소작업대 위", "지붕 위", "달비계", "작업발판 위", "높이 {h}m", "{h}m 높이"]
PLAIN_ACT = ["배관 볼트 교체", "펌프 정비", "자재 정리", "실내 도장", "바닥 청소", "모터 점검", "케이블 정리", "밸브 점검"]

FILLER_HEAD = ["오늘 오전", "작업 전 TBM 끝났고", "{n}팀", "점심 이후", "아침에", "어제에 이어서", "B동", "2공장",
               "R-{z}", "T-{z}"]
FILLER_TAIL = ["확인했습니다", "이상 없음", "진행할게요", "환기 중", "작업 시작합니다", "기록 남깁니다", "참고 바랍니다"]

NOT_MEASURE = ["작업자 {n}명", "{n}시간 예정", "{n}분 안에 끝냄", "{n}층", "{n}번 펌프", "자재 {n}kg", "볼트 {n}개",
               "자재 {n}톤", "배관 길이 {n}m", "{n}시 {m}분 시작", "{n}시 반 시작", "측정기 {n}대", "지게차 {n}대"]
# 숫자 아닌 측정 결과 ("산소는 정상"): 값으로 뽑지 않는다. 실제 값과 함께 쓰기도 한다
NOT_NUMBER_STATE = ["정상", "이상 없음", "괜찮음", "기준 안", "문제없음"]
NO_VALUE = ["산소 측정 아직 안 했어요", "황화수소는 들어가기 전에 잴 예정", "일산화탄소 수치 높게 나옴", "가스 측정 전",
            "측정기 고장이라 측정 못 함", "산소랑 황화수소는 정상", "가스 이상 없음", "측정값은 나중에 올릴게요",
            "가연성가스는 안 쟀어요", "암모니아 냄새 조금 남"]
WATCH_NONE = ["감시인 미정", "감시자 없음", "화재감시자 없이", "감시인 배치 필요", "감시자 아직 안 정함", "감시인 지정 안 함"]
# 측정 숫자로 착각하기 쉬운 산소 낱말 (측정값 아님)
OXY_TRAP = ["산소 마스크 쓰고 작업", "산소통 {n}개 운반", "산소 호흡기 착용"]


def _batchim(word: str) -> bool:
    ch = word[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def _josa(word: str, pair: str) -> str:
    """은/는, 이/가 같은 조사"""
    a, b = pair.split("/")
    return word + (a if _batchim(word) else b)


# ---------------------------------------------------------------- 조각

@dataclass
class Piece:
    text: str
    measures: list[tuple[str, str, str | None, str]] = field(default_factory=list)  # (항목, 숫자 글자, 단위 코드, 근거)
    work: list[tuple[str, str]] = field(default_factory=list)                        # (작업 코드, 근거)
    watcher: tuple[str, str] | None = None                                            # (이름, 근거)


class Maker:
    def __init__(self, seed: int, avoid_names: set[str]) -> None:
        self.r = random.Random(seed)
        self.avoid_names = avoid_names

    # ---- 값
    def number(self, item: str) -> str:
        spec = ITEMS[item]
        dec = self.r.choice(spec["dec"])
        v = self.r.uniform(spec["lo"], spec["hi"])
        text = f"{v:.{dec}f}" if dec else str(int(round(v)))
        return text

    def unit_word(self, code: str) -> str:
        return self.r.choice(UNIT_WORDS[code])

    def name(self) -> str:
        while True:
            surnames = AMBIGUOUS_SURNAMES if self.r.random() < 0.3 else SURNAMES
            n = self.r.choice(surnames) + self.r.choice(GIVEN)
            if n not in self.avoid_names:
                return n

    def item_name(self, item: str, abbr: bool) -> str:
        return ITEMS[item]["abbr"] if abbr else self.r.choice(ITEMS[item]["names"])

    # ---- 측정값 조각
    def measure(self, item: str, *, abbr: bool = False, unit: str | None = "ok", stt: bool = False,
                num: str | None = None) -> Piece:
        """unit: "ok"(기준 단위) | None(단위 없음) | "wrong"(다른 단위)"""
        nm = self.item_name(item, abbr)
        num = num or self.number(item)
        code = {"ok": ITEMS[item]["unit"], "wrong": WRONG_UNIT.get(item), None: None}[unit]
        if item == "LEL" and unit == "ok" and self.r.random() < 0.25:
            code = "PCT"                                    # "가연성가스 3%" 도 흔하다 (Spring 은 LEL 에 % 를 받음)
        uw = self.unit_word(code) if code else ""
        val = num + uw
        if stt:
            forms = ["{n} {v}"]
        elif abbr:
            forms = ["{n} {v}", "{n} {v}", "{n}={v}", "{n}: {v}"]
        else:
            forms = ["{n} {v}", "{n} {v}", "{n} {v}", _josa(nm, "은/는") + " {v}", _josa(nm, "이/가") + " {v} 나왔고",
                     "{n} 측정값 {v}", "{n} 재 보니 {v}", "{n}: {v}", "{n} {v} 나옴", "{n} 측정 결과 {v}", "{n}만 {v}"]
        form = self.r.choice(forms)
        text = form.format(n=nm, v=val)
        evidence = text.split(" 나왔고")[0].removesuffix(" 나옴")
        return Piece(text, [(item, num, code, evidence)])

    def measure_after(self, item: str) -> Piece:
        """값이 이름보다 앞 ("20.8% 산소")"""
        nm = self.item_name(item, False)
        num = self.number(item)
        code = ITEMS[item]["unit"]
        val = num + self.unit_word(code)
        ev = self.r.choice([f"{val} {nm}", f"{val} {nm}", f"{val}({nm})", f"{val} ({nm})"])
        text = self.r.choice(["{e}", "{e}", "측정값 {e}", "{e} 나옴"]).format(e=ev)
        return Piece(text, [(item, num, code, ev)])

    def duplicate(self, item: str) -> Piece:
        nm = self.item_name(item, self.r.random() < 0.2)
        a = self.number(item)
        b = a
        while float(b) == float(a):   # 22 와 22.0 은 같은 값 → 다시 뽑는다
            b = self.number(item)
        code = ITEMS[item]["unit"]
        ua, ub = self.unit_word(code), self.unit_word(code)
        # (첫 값, 잇는 말, 두 번째 값). 두 번째 값의 근거는 세 번째 칸 그대로
        form = self.r.choice([
            ("{n} {a}", " 나와서 환기 후 ", "다시 재니 {b}"), ("{n} {a}", ", ", "재측정 {b}"),
            ("처음 {n} {a}", " 나왔다가 ", "다시 재니 {b}"), ("{n} {a}", ", 환기하고 ", "다시 잰 {n} {b}"),
            ("오전 {n} {a}", " ", "오후 {n} {b}"),
            ("{n} {a}", " ", "다시 {b}"), ("{n} {a}", ", ", "다시 재서 {b}"),
        ])
        first = form[0].format(n=nm, a=a + ua)
        second = form[2].format(n=nm, b=b + ub)
        text = first + form[1] + second
        return Piece(text, [(item, a, code, first), (item, b, code, second)])

    # ---- 작업
    def hot(self) -> Piece:
        t = self.r.choice(HOT_ACT).format(o=self.r.choice(HOT_OBJ))
        return Piece(t, work=[("HOT", t)])

    def confined(self, act: bool = True) -> Piece:
        p = self.r.choice(CONFINED_PLACE).format(n=self.r.randint(1, 5))
        t = self.r.choice(CONFINED_ACT).format(p=p) if act else p
        return Piece(t, work=[("CONFINED", t)])

    def hot_bare(self) -> Piece:
        t = self.r.choice(HOT_BARE)
        return Piece(t, work=[("HOT", t)])

    def near_confined(self) -> Piece:
        """밀폐 장소 이름이 들어 있지만 밖에서 하는 작업 (작업 종류 없음)"""
        return Piece(self.r.choice(NEAR_CONFINED).format(n=self.r.randint(1, 5)))

    def confined_hot_phrase(self) -> Piece:
        """한 덩어리 "탱크 내부 용접", "반응조 내부 들어가서 용접" (밀폐 + 화기)"""
        place = self.r.choice(CONFINED_PLACE).format(n=self.r.randint(1, 5))
        hot = self.r.choice(HOT_BARE)
        mid = self.r.choice([" ", " ", " 들어가서 ", "에서 "])
        return Piece(f"{place}{mid}{hot}", work=[("CONFINED", place), ("HOT", hot)])

    def height_hot_phrase(self) -> Piece:
        """한 덩어리 "비계 위에서 용접" (고소 + 화기)"""
        place = self.r.choice(HEIGHT_PLACE).format(h=self.r.choice(["2", "2.5", "3", "4", "5", "6"]))
        hot = self.r.choice(HOT_BARE)
        mid = self.r.choice(["에서 ", "에서 ", " "])
        return Piece(f"{place}{mid}{hot}", work=[("HEIGHT", place), ("HOT", hot)])

    def not_number(self, items: list[str]) -> Piece:
        """"산소랑 황화수소는 정상" (값 아님)"""
        names = [self.item_name(i, False) for i in items]
        joined = "랑 ".join(names) if len(names) > 1 else names[0]
        return Piece(f"{_josa(joined, '은/는')} {self.r.choice(NOT_NUMBER_STATE)}")

    def height(self, ok: bool = True) -> Piece:
        task = self.r.choice(HEIGHT_TASK)
        if not ok:   # 2m 미만·층수 → 고소 아님
            t = self.r.choice(["높이 {h}m 선반 {t}", "{h}m 높이 작업대에서 {t}", "{n}층 사무실 {t}"]).format(
                h=self.r.choice(["1", "1.2", "1.5", "1.8"]), n=self.r.randint(2, 9), t=task)
            return Piece(t)
        act = self.r.choice(HEIGHT_ACT)
        t = act.format(t=task, h=self.r.choice(["2", "2.5", "3", "4", "5", "6", "8", "10"]))
        return Piece(t, work=[("HEIGHT", t)])

    # ---- 감시자
    def watcher(self, mode: str, word: str | None = None) -> Piece:
        if mode == "none":
            return Piece(self.r.choice(WATCH_NONE))
        n = self.name()
        word = word or self.r.choice(["감시인", "감시자", "화재감시자"])
        if mode == "front":
            topic, by = _josa(word, "은/는"), _josa(word, "으로/로")   # "감시인은", "감시자는" (11/12: "감시인는" 고침)
            t = self.r.choice([f"{word} {n}", f"{topic} {n}", f"{topic} {n} 씨", f"{by} {n} 지정", f"{word} {n} 배치",
                               f"{word} {n} 대기"])
            ev = t
        else:
            t = self.r.choice([f"{n} 씨가 {word}", f"{_josa(n, '이/가')} {word} 맡음", f"{n} {word} 대기",
                               f"{n} 반장이 {word}", f"{n} {word} 배치 완료"])
            ev = t
        for tail in (" 맡음", " 대기", " 배치 완료", " 배치", " 지정"):
            if ev.endswith(tail) and tail != " 지정":
                ev = ev[: -len(tail)]
        return Piece(t, watcher=(n, ev))

    # ---- 그 밖
    def not_measure(self) -> Piece:
        return Piece(self.r.choice(NOT_MEASURE).format(n=self.r.randint(1, 12), m=self.r.choice(["10", "20", "40", "50"])))

    def filler_head(self) -> str:
        return self.r.choice(FILLER_HEAD).format(n=self.r.randint(1, 5), z=self.r.randint(100, 399))

    def plain(self) -> Piece:
        return Piece(self.r.choice(PLAIN_ACT))

    def oxy_trap(self) -> Piece:
        return Piece(self.r.choice(OXY_TRAP).format(n=self.r.randint(2, 6)))


# ---------------------------------------------------------------- 문장 만들기

TAG_WEIGHTS = {
    "기본": 16, "여러 값": 12, "영문 약어": 6, "CO·CO2 구분": 5, "측정 아닌 숫자": 7, "값 없음": 8, "단위 없음": 9,
    "단위 틀림": 4, "값 두 번": 8, "값 하나씩": 3, "감시인": 9, "작업 종류": 18, "STT 출력형": 8, "어순": 5,
}


def _items(mk: Maker, k: int, exclude: tuple[str, ...] = ()) -> list[str]:
    pool = [i for i in ITEMS if i not in exclude]
    weights = [6 if i in ("O2", "H2S", "CO") else 3 if i in ("LEL", "CO2") else 1 for i in pool]
    out: list[str] = []
    while len(out) < k:
        i = mk.r.choices(pool, weights)[0]
        if i not in out:
            out.append(i)
    return out


def build(mk: Maker, tag: str) -> list[Piece]:
    r = mk.r
    pieces: list[Piece] = []
    if tag in ("기본", "여러 값", "영문 약어", "단위 표기"):
        n = r.randint(1, 2) if tag == "기본" else r.randint(3, 5) if tag == "여러 값" else r.randint(1, 4)
        pieces += [mk.measure(i, abbr=tag == "영문 약어") for i in _items(mk, n)]
    elif tag == "CO·CO2 구분":
        both = [mk.measure("CO", abbr=r.random() < 0.4), mk.measure("CO2", abbr=r.random() < 0.4)]
        r.shuffle(both)
        pieces += both if r.random() < 0.8 else [both[0]]
    elif tag == "측정 아닌 숫자":
        pieces += [mk.not_measure() for _ in range(r.randint(1, 2))]
        if r.random() < 0.5:
            pieces.append(mk.measure(r.choice(["O2", "H2S", "CO"])))
        if r.random() < 0.3:
            pieces.append(mk.oxy_trap())
    elif tag == "값 없음":
        if r.random() < 0.5:
            pieces.append(Piece(r.choice(NO_VALUE)))
        else:   # "산소랑 황화수소는 정상, 일산화탄소만 5ppm": 숫자 있는 것만 뽑는다
            states = _items(mk, r.randint(1, 2))
            pieces.append(mk.not_number(states))
            if r.random() < 0.8:
                pieces.append(mk.measure(_items(mk, 1, tuple(states))[0]))
    elif tag == "단위 없음":
        for i in _items(mk, r.randint(1, 3)):
            pieces.append(mk.measure(i, unit=None if r.random() < 0.7 else "ok", abbr=r.random() < 0.2))
        if all(m[2] is not None for p in pieces for m in p.measures):
            pieces[0] = mk.measure(pieces[0].measures[0][0], unit=None)
    elif tag == "단위 틀림":
        wrong = r.choice(list(WRONG_UNIT))
        pieces.append(mk.measure(wrong, unit="wrong"))
        if r.random() < 0.5:   # 다른 항목 하나 (같은 항목을 또 넣으면 "값 두 번" 이 된다)
            pieces.append(mk.measure(_items(mk, 1, (wrong,))[0]))
    elif tag == "값 두 번":
        pieces.append(mk.duplicate(r.choice(["O2", "O2", "H2S", "CO", "LEL", "CO2"])))
        if r.random() < 0.5:
            used = {pieces[0].measures[0][0]}
            pieces.append(mk.measure(_items(mk, 1, tuple(used))[0]))
    elif tag == "값 하나씩":
        num = mk.number("H2S") if r.random() < 0.5 else None
        for i in _items(mk, r.randint(2, 3), ("O2", "CO2", "LEL")):
            pieces.append(mk.measure(i, num=num))
    elif tag == "감시인":
        pieces.append(mk.watcher(r.choice(["front", "front", "back", "back", "none"])))
        if r.random() < 0.5:
            pieces.append(mk.measure(r.choice(["O2", "H2S"])))
    elif tag == "작업 종류":
        kind = r.choice(["hot", "hot_bare", "confined", "confined", "height", "height_no", "hot_confined",
                         "hot_height", "plain", "oxy", "near_confined", "near_confined", "confined_hot_phrase",
                         "confined_hot_phrase", "height_hot_phrase", "height_hot_phrase", "fire_watch"])
        pieces += {
            "hot": lambda: [mk.hot()], "hot_bare": lambda: [mk.hot_bare()], "confined": lambda: [mk.confined()],
            "height": lambda: [mk.height()],
            "height_no": lambda: [mk.height(ok=False)], "plain": lambda: [mk.plain()], "oxy": lambda: [mk.oxy_trap()],
            "hot_confined": lambda: [Piece(f"{mk.confined(act=False).text}에서 {mk.hot().text}")],
            "hot_height": lambda: [mk.height(), mk.hot()],
            "near_confined": lambda: [mk.near_confined()],
            "confined_hot_phrase": lambda: [mk.confined_hot_phrase()],
            "height_hot_phrase": lambda: [mk.height_hot_phrase()],
            # "화재감시자 ○○ 대기, 용접 시작" (화재감시자 + 짧은 화기)
            "fire_watch": lambda: [mk.watcher(r.choice(["front", "back"]), word="화재감시자"), mk.hot_bare()],
        }[kind]()
        if kind == "hot_confined":   # 한 덩어리 문장: 근거를 다시 단다
            t = pieces[0].text
            place, act = t.split("에서 ", 1)
            pieces[0] = Piece(t, work=[("CONFINED", place), ("HOT", act)])
    elif tag == "STT 출력형":
        if r.random() < 0.5:
            pieces.append(r.choice([mk.confined, mk.hot])())
        pieces += [mk.measure(i, stt=True) for i in _items(mk, r.randint(1, 3))]
        if r.random() < 0.6:
            pieces.append(mk.watcher(r.choice(["front", "front", "back"])))
    elif tag == "어순":
        pieces += [mk.measure_after(i) for i in _items(mk, r.randint(1, 2))]
    else:
        raise ValueError(tag)

    # 작업·감시자·잡담을 섞어 현장 문장처럼
    if tag not in ("작업 종류", "STT 출력형") and r.random() < 0.45:
        pieces.insert(0, r.choice([mk.confined, mk.hot, mk.height, mk.plain])())
    if tag not in ("감시인", "STT 출력형") and r.random() < 0.2 and not any(p.watcher for p in pieces):
        pieces.append(mk.watcher(r.choice(["front", "back", "none"])))
    return pieces


def sentence(mk: Maker, tag: str) -> tuple[str, dict]:
    pieces = build(mk, tag)
    stt = tag == "STT 출력형"
    head = "" if stt or mk.r.random() < 0.7 else mk.filler_head() + " "
    tail = "" if stt or mk.r.random() < 0.75 else mk.r.choice([", ", " "]) + mk.r.choice(FILLER_TAIL)
    sep = " " if stt else mk.r.choice([", ", ", ", " ", " / ", ". "])
    text = head + sep.join(p.text for p in pieces) + tail

    measures, work, watcher = [], [], None
    for p in pieces:
        for item, num, code, ev in p.measures:
            measures.append({"item": item, "value": json.loads(num), "unit": code, "evidence": ev})
        for code, ev in p.work:
            if code not in {w["type"] for w in work}:
                work.append({"type": code, "evidence": ev})
        if p.watcher and watcher is None:
            watcher = p.watcher
    out = {"measurements": measures, "workTypes": work,
           "watcherName": watcher[0] if watcher else None, "watcherEvidence": watcher[1] if watcher else None}
    return text, out


def check(text: str, out: dict) -> None:
    """정답이 원문과 맞는지 (근거는 원문 그대로, 숫자는 근거 안에, 이름은 원문 안에)"""
    for m in out["measurements"]:
        num = json.dumps(m["value"])
        assert m["evidence"] in text, (text, m)
        assert num in m["evidence"], (text, m)
        assert m["item"] in ITEMS and m["unit"] in (None, "PCT", "PPM", "PCT_LEL"), m
    for w in out["workTypes"]:
        assert w["evidence"] in text and w["type"] in ("HOT", "CONFINED", "HEIGHT"), (text, w)
    if out["watcherName"]:
        assert out["watcherName"] in text and out["watcherEvidence"] in text, (text, out)
        assert out["watcherName"] in out["watcherEvidence"], out


def _grams(t: str) -> set[str]:
    t = t.replace(" ", "")
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}


def _near(a: set[str], b: set[str]) -> bool:
    return len(a & b) / max(1, len(a | b)) >= 0.9


def load_gold(paths: list[Path]) -> tuple[list[str], set[str]]:
    texts, names = [], set()
    for p in paths:
        for c in json.loads(p.read_text(encoding="utf-8")):
            texts.append(c["text"])
            w = (c.get("form") or {}).get("watcher_name")
            if w:
                names.add(w)
    return texts, names


def generate(n: int, seed: int, gold_texts: list[str], gold_names: set[str]) -> list[dict]:
    mk = Maker(seed, gold_names)
    tags = list(TAG_WEIGHTS)
    weights = [TAG_WEIGHTS[t] for t in tags]
    gold_exact = set(gold_texts)
    gold_grams = [_grams(t) for t in gold_texts]
    seen: set[str] = set()
    rows: list[dict] = []
    while len(rows) < n:
        tag = mk.r.choices(tags, weights)[0]
        text, out = sentence(mk, tag)
        if text in seen or text in gold_exact:
            continue
        g = _grams(text)
        if any(_near(g, x) for x in gold_grams):
            continue
        check(text, out)
        seen.add(text)
        messages = build_adapter_messages(text)
        messages.append({"role": "assistant", "content": format_lines(out)})
        rows.append({"messages": messages, "tag": tag, "answer": out})   # answer: 검사·평가용 정답 (학습은 messages 만 씀)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="문장 추출 어댑터 학습 데이터 생성")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="저장 폴더 (기본 data/extract-train, git 에 안 올라감)")
    parser.add_argument("--train", type=int, default=3000)   # 11/12: 유형을 늘려 2000 → 3000
    parser.add_argument("--valid", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gold", type=Path, action="append", default=[],
                        help="겹치면 안 되는 평가 정답 파일 (기본 eval/rule_input_extract_v2.json 은 늘 포함)")
    parser.add_argument("--sample", type=int, default=0, help="유형별 N건씩 화면에만 보여 주고 끝 (저장 안 함)")
    args = parser.parse_args(argv)

    gold_texts, gold_names = load_gold([GOLD_V2, *args.gold])
    rows = generate(args.train + args.valid, args.seed, gold_texts, gold_names)

    if args.sample:
        shown: Counter = Counter()
        for row in rows:
            if shown[row["tag"]] < args.sample:
                shown[row["tag"]] += 1
                print(f"[{row['tag']}] {row['messages'][1]['content'].removeprefix('문장: ')}")
                print(f"   {row['messages'][2]['content']}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", rows[:args.train]), ("valid.jsonl", rows[args.train:])):
        with (args.out / name).open("w", encoding="utf-8", newline="\n") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = Counter(r["tag"] for r in rows)
    print(f"저장: {args.out / 'train.jsonl'} {args.train}건, {args.out / 'valid.jsonl'} {args.valid}건 (seed {args.seed})")
    for tag, count in stats.most_common():
        print(f"   {tag}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
