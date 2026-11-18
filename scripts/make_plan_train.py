"""계획 단계 어댑터(plan_brief LoRA) 학습 데이터 생성기 (B). app 코드와 표준 라이브러리만 쓴다.

    python -m scripts.make_plan_train              # data/plan-train/ 에 train.jsonl 2400 + valid.jsonl 240
    python -m scripts.make_plan_train --sample 3   # 유형별 3건씩 화면에만 (저장 안 함)

- 한 줄 = {"messages": [system, user, assistant], "kind": 유형, "utterances": [...], "answer": {...}}.
  **system·user 는 서버와 같은 코드로 만든다** (app/agent/plan_adapter.py 의 ADAPTER_SYSTEM_PROMPT·build_adapter_input).
  user 의 '주변 물질'·'투입 장비' 줄은 서버와 같은 경로(slots.merge_slots 를 발화마다 거친 slots)로 채운다.
  assistant = {"summary": "...", "equipment": [...], "materials": [...]} (서버가 기대하는 JSON 3키).
  학습은 messages 만 쓰고 answer 는 검사·평가용이다.
- 정답 기준 (scripts/plan_eval/eval_set.jsonl 과 같음)
    equipment  쓰는 공구·장비. 조건부('고착되면 토치로')는 넣고, 쓰지 않는다고 한 것('크레인은 안 씀', 'A 말고 B', 'A 대신 B')은 뺀다.
               사전에 없는 것도 넣는다 (고압세척기 같은 것).
    materials  이름이 나온 화학물질·가스 (사전 밖 포함). 설비·부품·장소·설비 번호(T-105, 플랜지, 베어링)는 물질이 아니다.
    summary    작업 한 줄 (60자 이내). 원문에 있는 말로만 쓴다.
  항목은 **문장에 적힌 그대로의 표기**로 쓴다 (서버 검증이 원문에 그 글자가 있는지 본다. 사전 표기 '신나'는 '신나'로).
- **코드 검증 필터**: 한 건마다 서버의 검증 함수(plan_brief.clean_equipment·clean_materials·clean_summary)를 그대로 돌려
  버려지는 항목·요약이 하나라도 있으면 만들기를 멈춘다 (만든 쪽 실수이므로). 요약은 원문 근거 지표(harness.summary_support)도 1.0이어야 한다.
  슬롯 줄이 정답과 어긋나는지도 본다 (사전 단어가 정답에 없는데 줄에 들어가는 경우는 '사전 단어를 품은 긴 이름' 같은 T25 빈틈이라 만들지 않는다).
- **평가 오염 방지**: 평가 40문장과 같거나 비슷한 문장은 버리고, 사전 밖 물질·장비 이름은 평가 정답에 나오는 것과 겹치지 않게 따로 둔다.
  검증(valid)은 학습에 없는 사전 밖 이름만 쓴다 (처음 보는 이름에 얼마나 일반화되는지 본다).
- 같은 --seed 면 언제 돌려도 같은 데이터가 나온다. 데이터는 git·이미지에 올리지 않는다 (data/ 는 .gitignore).

이 생성기의 문구·조립은 임시값(내 휴리스틱)이다. 현장 문장이 쌓이면 실무자 문장을 섞는다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from app.agent import plan_brief as pb
from app.agent import text_rules
from app.agent.evidence_collector.models import WorkContext
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, build_adapter_input
from app.agent.slots import full_text, merge_slots
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms
from app.agent.work_register import MAX_CONTEXT_CHARS
from app.schemas.agent_work import WorkSlots
from scripts.plan_eval.harness import DEFAULT_EVAL_SET, load_cases, summary_support

MULTILINE_EVAL_SET = DEFAULT_EVAL_SET.with_name("eval_set_multiline.jsonl")  # 여러 줄 평가 12문장 (이것도 학습에 새면 안 된다)
USER_HARD_EVAL_SET = DEFAULT_EVAL_SET.with_name("eval_set_user_hard.jsonl")  # 실제 현장 문장 5 + 어려운 문장 22 (11/17, 이것도 학습에 새면 안 된다)
DEFAULT_EVAL_SETS = (DEFAULT_EVAL_SET, MULTILINE_EVAL_SET, USER_HARD_EVAL_SET)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "plan-train"
TODAY = date(2026, 1, 15)  # '내일' 같은 날짜 해석 기준 (슬롯 줄에는 영향이 없지만 결과가 날짜에 흔들리지 않게 고정)

# ---------------------------------------------------------------- 재료

# 사전 표기: 사전의 모든 표기를 쓰되, 영문 이름(toluene 등)과 뜻이 다른 '렌탈'은 현장 글에 잘 안 나와 뺀다.
# 'H2S'도 뺀다: 서버의 clean_materials 가 설비 번호 모양(영문 1~4자+숫자+영문, plan_brief._TAG_RE)으로 보고 버리므로
# 모델 정답에 넣을 수 없다 (슬롯 사전이 대신 잡는다).
_SKIP_ALIASES = frozenset({"렌탈", "H2S", "toluene", "methanol", "ethanol", "acetone", "benzene", "xylene", "hexane", "thinner"})
DICT_EQUIPMENT: dict[str, list[str]] = {c: [a for a in al if a not in _SKIP_ALIASES] for c, al in EQUIPMENT.items()}
DICT_MATERIALS: dict[str, list[str]] = {c: [a for a in al if a not in _SKIP_ALIASES] for c, al in MATERIALS.items()}

# 사전 밖 장비. 평가 정답(핸드브레이커·고압세척기·와이어브러시·열풍기·샌딩기·스프레이건·윈치 …)과 겹치지 않고, 사전 단어(드릴·렌치 …)를 품지 않는다.
# 11/17: 윈치는 어려운 평가 문장(h15)의 정답이라 검증용에서 뺐다.
OOD_EQUIPMENT_TRAIN = ["전동드라이버", "토크렌치", "전동톱", "연마기", "덤프트럭", "굴착기", "진공흡입차", "스팀세척기", "리프트", "착암기"]
OOD_EQUIPMENT_VALID = ["분쇄기", "에어건", "컷팅기"]
# 사전 밖 물질. 평가 정답(가성소다·아르곤·부탄·포름알데히드·에틸렌·이산화탄소·과산화수소·헬륨 …)과 겹치지 않고
# 사전 단어(수소·메탄·헥산·에틸렌 …)를 품지 않는다 (슬롯 줄이 정답과 어긋나지 않게).
# 11/17: 질산·초산(·아세트산)은 사전에 들어가 사전 밖이 아니다 (text_rules.EXTENDED_VOCAB). 헬륨은 어려운 평가 문장(h19)의 정답이라 검증용에서 뺐다.
OOD_MATERIALS_TRAIN = [
    "인산", "불산", "수산화칼륨", "차아염소산나트륨", "염화메틸렌", "메틸에틸케톤", "스티렌", "이소시아네이트",
    "프로필렌", "일산화탄소", "이산화황", "클로로포름", "아크릴로니트릴", "이산화질소", "과망간산칼륨", "염화칼슘",
    "페놀", "글리세린",
]
OOD_MATERIALS_VALID = ["개미산", "네온", "락스", "요소수", "사염화탄소"]

# 장비마다 하는 일 (명사). '그라인더로 표면 연마'처럼 쓴다.
EQUIP_ACTS: dict[str, list[str]] = {
    "그라인더": ["표면 연마", "녹 제거", "볼트 절단", "용접부 다듬기"],
    "용접기": ["배관 용접", "브래킷 용접", "균열 보수 용접", "연결부 용접"],
    "토치": ["가열", "예열", "배관 절단", "볼트 가열"],
    "가스절단기": ["기존 배관 절단", "철판 절단", "플랜지 절단"],
    "크레인": ["자재 인양", "배관 인양", "모터 인양", "철골 인양"],
    "호이스트": ["펌프 인양", "덮개 인양", "모터 인양"],
    "체인블록": ["임펠러 인양", "기어박스 인양", "밸브 인양"],
    "지게차": ["자재 운반", "팔레트 운반", "드럼 운반", "배관 운반"],
    "고소작업대": ["상부 점검", "조명 교체", "지지대 확인", "상부 보수"],
    "사다리": ["상부 점검", "볼트 조임", "점검"],
    "사다리차": ["상부 점검", "외벽 보수", "지붕 자재 이동"],
    "비계": ["외벽 도장", "상부 보수", "배관 점검"],
    "송풍기": ["환기", "내부 환기", "공기 순환"],
    "가스측정기": ["농도 확인", "내부 공기 측정", "가스 측정"],
    "발전기": ["전원 공급", "임시 전원 확보"],
    "임팩트렌치": ["볼트 체결", "너트 해체", "플랜지 체결"],
    "드릴": ["천공", "앵커 자리 천공", "구멍 뚫기"],
    "전동드라이버": ["나사 체결", "패널 체결"],
    "토크렌치": ["규정 토크 체결", "볼트 최종 체결"],
    "전동톱": ["합판 절단", "각재 절단"],
    "연마기": ["표면 연마", "용접부 연마"],
    "덤프트럭": ["토사 운반", "잔토 반출"],
    "굴착기": ["바닥 굴착", "배관 매설부 굴착"],
    "진공흡입차": ["슬러지 흡입", "잔류물 흡입"],
    "스팀세척기": ["표면 세척", "배관 세척"],
    "리프트": ["자재 상차", "자재 이동"],
    "착암기": ["콘크리트 파쇄", "바닥 파쇄"],
    "윈치": ["케이블 인입", "자재 견인"],
    "에어건": ["먼지 제거", "표면 청소"],
    "컷팅기": ["콘크리트 절단", "바닥 절단"],
    "분쇄기": ["폐자재 분쇄", "잔재 분쇄"],
}

LOCATIONS = [
    "보일러 연소실", "사무동 옥상", "펌프실", "지하 피트", "저장 탱크 상부", "도장 부스", "냉각탑", "열교환기 구역", "컨베이어 라인",
    "폐수처리장", "탱크 하부", "창고 외벽", "배관 랙", "변전실 앞", "기계실", "원료 투입구", "집진기 하우징", "출하장", "세척실",
    "시험동", "{n}호 보일러", "{n}호기 펌프", "{n}공장 지붕", "B동 {n}층", "{n}번 라인", "A동 계단실", "야적장", "정수 설비", "압축기실",
]
PARTS = [
    "플랜지 볼트", "베어링", "가스켓", "패킹", "커플링", "기어박스", "모터", "임펠러", "필터", "센서", "체크밸브", "오링", "벨트",
    "체인", "와이어로프", "배관", "드레인 밸브", "덮개", "레벨계", "압력계", "온도계", "스트레이너", "지지대", "난간", "밸브",
]
TASK_FORMS = ["{p} 교체", "{p} 점검", "{p} 보수", "{p} 정비", "{p} 교환", "{p} 조임", "{p} 철거", "{p} 도색", "{p} 누설 점검", "{p} 분해 정비"]
TASK_PLAIN = [
    "배관 이설", "배관 해체", "상부 난간 보수", "바닥 보수", "표면 처리", "방청 도장", "보온재 교체", "슬러지 제거", "정기 세정",
    "덕트 청소", "필터 교체", "레벨 교정", "지지대 보강", "내부 청소", "외벽 도장", "시운전 점검",
]
TAG_FORMS = ["T-{n}", "P-{n}", "R-{n}", "V-{n}", "E-{n}", "H-{n}"]
NONE_TASKS = [
    "소화기 점검표 작성", "게시판 공지 교체", "비상구 표지 부착 확인", "출입 명부 정리", "서랍 정리", "회의 자료 인쇄와 배포",
    "월간 점검표 작성", "물품 대장 정리", "청소 도구 보충", "근무 일지 작성", "방문자 안내", "출퇴근 기록 확인", "신입 교육 오리엔테이션",
    "비상 연락망 갱신", "휴게실 정리", "교육 자료 배포", "사무용품 재고 확인", "복도 물청소 전 안내문 부착", "안전 퀴즈 진행", "작업복 지급",
    "보호구 보관함 정리", "화분 정리", "주차 구획 안내판 교체", "서류 결재 올리기",
]
HEADS = ["오늘", "내일", "오전에", "오후에", "내일 오전", "다음 주 월요일", "점심 이후", "작업 전 TBM 끝났고", "아침에", "어제 이어서"]
EXTRAS = ["{n}명 투입", "{n}명이 진행", "작업 인원 {n}명", "{n}명 작업", "오후 3시까지", "오전 중에 끝낼 예정", "점심 전에 끝남", "2시간 정도 걸림", "안전모 착용", "작업 허가서 받음"]
# 작업 이름 뒤에 붙는 끝맺음 (붙여 쓰는 것, 띄어 쓰는 것)
ENDINGS = [("함", ""), ("합니다", ""), ("할 예정", ""), ("한다", ""), ("해요", ""), ("할 거야", ""), ("진행", " "), ("예정입니다", " "), ("실시", " ")]

# verbose: 말이 많은 현장 문장 (과거형 서술, 이유·잡담이 섞임). 학습이 '장소+부품+작업명' 한 틀에만 맞춰져 틀 밖 문장에서 요약이 잘리거나 어색해지던 것을 줄인다 (T30)
_KINDS = ("plain", "negation", "conditional", "out_of_dict", "multi", "none", "distractor", "verbose")
KIND_WEIGHTS = {"plain": 20, "negation": 16, "conditional": 12, "out_of_dict": 18, "multi": 9, "none": 8, "distractor": 9, "verbose": 14}
# 말이 많은 문장 재료 (verbose). 사전 단어가 들어가지 않고, 사전 단어 바로 뒤에 부정처럼 읽히는 말이 오지 않게 쓴다.
VERBOSE_ENDINGS = ["했음", "했습니다", "했어요", "하였음"]  # 작업 이름에 바로 붙인다
VERBOSE_OBJECT_ENDINGS = ["마침", "끝냄", "완료함"]  # 작업 이름 + 을/를 + 이 말 ('교체를 마침')
VERBOSE_PLACE_SUFFIX = ["", "", " 앞", " 쪽", " 내부"]
VERBOSE_REASONS = ["볼트가 안 풀려서", "손으로는 안 돼서", "시간이 모자라서", "작업량이 많아서", "바닥이 단단해서", "생각보다 오래 걸려서"]
VERBOSE_FILLERS = [
    "ㅎㅎ", "특이사항 없었음", "날씨는 맑았음", "점심 먹고 시작했음", "내일 이어서 할 예정", "수고하셨습니다", "시간이 좀 걸렸음", "마무리 정리까지 끝냈음",
    "중간에 잠깐 쉬었음", "다들 고생 많았음", "ㅋㅋ", "오전부터 바빴음",
]
MULTI_LINE_RATE = 0.35  # 여러 발화(되묻기 답)로 나누는 비율

# ---------------------------------------------------------------- 말 만들기


def ckey(name: str) -> str:
    """사전 표기를 표준 이름으로 (사전에 없으면 그대로). 같은 사전 항목의 다른 표기는 한 장면에 한 번만 쓴다."""
    key = pb._compact(name).lower()  # noqa: SLF001
    return pb._EQUIPMENT_KEYS.get(key) or pb._MATERIAL_KEYS.get(key) or name  # noqa: SLF001


def used(pieces: list, name: str) -> bool:
    """name(또는 같은 사전 항목의 다른 표기)이 이미 장면에 쓰였나 (쓰는 것·쓰지 않는 것 모두)."""
    return any(ckey(name) == ckey(x) for p in pieces for x in (*p.equipment, *p.materials, *p.excluded))


def _batchim(word: str) -> bool:
    ch = word[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str, pair: str) -> str:
    """'은/는', '이/가', '을/를', '으로/로' 같은 조사 (받침이 있으면 앞쪽). '으로/로'는 ㄹ 받침이면 '로'."""
    a, b = pair.split("/")
    if pair == "으로/로":
        return word + (b if not _batchim(word) or (ord(word[-1]) - 0xAC00) % 28 == 8 else a)
    return word + (a if _batchim(word) else b)


@dataclass
class Piece:
    """문장 한 조각과 그 정답. 정답 항목은 text 안에 적힌 그대로의 표기다."""

    role: str  # task | equip | mat | misc
    text: str
    answer: str | None = None  # 나눠 말할 때 쓰는 짧은 답 형태 (없으면 첫 발화에 그대로 둔다)
    equipment: list[str] = field(default_factory=list)  # 쓰는 것
    materials: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)  # 쓰지 않는다고 한 것 (정답에 없어야 한다)
    soft: bool = False  # excluded 가 사전의 부정 규칙(text_rules.NEGATION_AFTER)에 안 걸리는 표현 (슬롯 줄에 들어간다)
    later: bool = False  # 앞 발화에서 한 말을 뒤 발화에서 거두는 조각. 항상 뒤 발화로 따로 말한다 (서버 검증은 이런 거둠을 못 걸러낸다, T28)
    summary: str = ""  # 요약에 보탤 말 (원문에 있는 말로만)


@dataclass
class Sample:
    kind: str
    pieces: list[Piece]
    summary: str


class Maker:
    def __init__(self, seed: int, *, holdout: bool = False) -> None:
        self.r = random.Random(seed)
        self.ood_equipment = OOD_EQUIPMENT_VALID if holdout else OOD_EQUIPMENT_TRAIN
        self.ood_materials = OOD_MATERIALS_VALID if holdout else OOD_MATERIALS_TRAIN

    # ---- 낱말
    def n(self, lo: int = 1, hi: int = 9) -> int:
        return self.r.randint(lo, hi)

    def equipment(self, *, ood: bool | None = None, exclude: tuple[str, ...] = ()) -> tuple[str, str]:
        """(문장에 쓸 표기, 일 목록의 열쇠). 사전 밖이면 표기=열쇠."""
        use_ood = self.r.random() < 0.25 if ood is None else ood
        while True:
            if use_ood:
                name = self.r.choice(self.ood_equipment)
                key, surface = name, name
            else:
                key = self.r.choice(list(DICT_EQUIPMENT))
                aliases = DICT_EQUIPMENT[key]
                surface = aliases[-1] if len(aliases) > 1 and self.r.random() < 0.25 else aliases[0]
            if ckey(surface) not in {ckey(x) for x in exclude} and key not in exclude:
                return surface, key

    def material(self, *, ood: bool | None = None, exclude: tuple[str, ...] = ()) -> str:
        use_ood = self.r.random() < 0.3 if ood is None else ood
        while True:
            if use_ood:
                surface = self.r.choice(self.ood_materials)
            else:
                key = self.r.choice(list(DICT_MATERIALS))
                aliases = DICT_MATERIALS[key]
                surface = aliases[-1] if len(aliases) > 1 and self.r.random() < 0.25 else aliases[0]
            if ckey(surface) not in {ckey(x) for x in exclude}:
                return surface

    def location(self) -> str:
        return self.r.choice(LOCATIONS).format(n=self.n())

    def part(self) -> str:
        return self.r.choice(PARTS)

    def tag(self) -> str:
        return self.r.choice(TAG_FORMS).format(n=self.r.choice([101, 105, 201, 301, 12, 7, 33]))

    def act(self, key: str) -> str:
        return self.r.choice(EQUIP_ACTS[key])

    def ending(self) -> str:
        """작업 이름에 바로 붙일 끝맺음 (앞에 띄어쓰기 포함)."""
        word, space = self.r.choice(ENDINGS)
        return space + word

    # ---- 조각
    def task(self, core: str | None = None) -> tuple[Piece, str]:
        """작업 조각. (조각, 요약에 쓸 '장소 작업'). 요약은 문장 안에 있는 말로만 만든다."""
        loc = self.location()
        core = core or (self.r.choice(TASK_FORMS).format(p=self.part()) if self.r.random() < 0.6 else self.r.choice(TASK_PLAIN))
        form = self.r.choice(["{loc} {core}.", "{loc} {core} 작업.", "{loc} {core}{end}.", "{loc} {core}", "{loc}에서 {core}{end}."])
        text = form.format(loc=loc, core=core, end=self.ending())
        head = self.r.choice(HEADS) + " " if self.r.random() < 0.3 else ""
        return Piece("task", head + text), f"{loc} {core}"

    def use_equipment(self, surface: str, key: str) -> Piece:
        act = self.act(key)
        eo = josa(surface, "으로/로")
        forms = [
            (f"{eo} {act}함.", f"{surface} 사용"),
            (f"{surface} 사용해서 {act}.", f"{surface} 쓸게요"),
            (f"{eo} {act} 진행.", f"{surface}로 합니다"),
            (f"{eo} {act} 작업.", f"장비는 {surface}"),
            (f"{surface} 가져와서 {act}.", f"{surface} 가져가요"),
            (f"{surface} 투입.", f"{surface} 투입해요"),
            (f"{surface} 동원.", f"{surface} 동원합니다"),
            (f"{surface} 가동 후 {act}.", f"{surface} 씁니다"),
        ]
        text, answer = self.r.choice(forms)
        return Piece("equip", text, answer, equipment=[surface])

    def mention_material(self, m: str) -> Piece:
        n = self.n(2, 6)
        forms = [
            (f"{m} 말통이 근처에 놓여 있음.", f"근처에 {m} 말통이 있어요"),
            (f"{m} 저장 구역과 가까움.", f"{m} 저장 구역이 가까워요"),
            (f"{josa(m, '으로/로')} 부품을 헹궈냄.", f"세척에 {m} 씁니다"),
            (f"{josa(m, '으로/로')} 얼룩 제거 예정.", f"{m} 얼룩 제거에 써요"),
            (f"{m} 새는 곳 없는지 점검.", f"{m} 누설 점검도 합니다"),
            (f"{m} 용기가 {n}개 놓여 있음.", f"{m} 용기 {n}개 있어요"),
            (f"{m} 이송 배관과 인접.", f"{m} 이송 배관이 인접해요"),
            (f"{m} 소분실 앞에서 진행.", f"{m} 소분실이 앞에 있어요"),
            (f"{m} 쓰는 구역임.", f"{m} 쓰는 구역이에요"),
            (f"{m} 취급 라인 옆.", f"{m} 취급 라인이 옆에 있어요"),
            (f"{m} 투입 설비 근처.", f"{m} 투입 설비가 근처에 있어요"),
        ]
        text, answer = self.r.choice(forms)
        return Piece("mat", text, answer, materials=[m])

    def negate_equipment(self, surface: str, *, soft: bool = False) -> Piece:
        if soft:  # 사전의 부정 규칙(text_rules.NEGATION_AFTER)이 지금도 못 잡는 표현 (슬롯 줄에 들어간다). 11/17 정중형·'쓸 일 없'·'사용 금지'를 규칙에 더한 뒤 남은 말들 (T28, T30)
            text, answer = self.r.choice([
                (f"{josa(surface, '은/는')} 빼고 진행함.", f"{surface} 빼고 갑니다"), (f"{josa(surface, '은/는')} 가져오지 않음.", f"{surface} 가져오지 않아요"),
                (f"{josa(surface, '은/는')} 안 가져옴.", f"{surface} 안 가져왔어요"), (f"{surface} 반입하지 않음.", f"{surface} 반입하지 않습니다"),
                (f"{surface} 반입 안 함.", f"{surface} 반입 안 해요"),
            ])
            return Piece("equip", text, answer, excluded=[surface], soft=True)
        forms = [
            (f"{josa(surface, '은/는')} 안 씀.", f"{surface} 안 써요"), (f"{surface} 사용 안 함.", f"{surface} 사용 안 합니다"),
            (f"{surface} 사용하지 않음.", f"{surface} 사용하지 않습니다"), (f"{josa(surface, '은/는')} 필요 없음.", f"{surface} 필요 없어요"),
            (f"{surface} 없이 진행.", f"{surface} 없이 합니다"), (f"{josa(surface, '은/는')} 제외.", f"{surface} 제외입니다"),
            (f"{surface} 미사용.", f"{surface} 미사용입니다"), (f"{surface} 안 쓰고 진행.", f"{surface} 안 쓰고 진행합니다"),
            # 정중형·완료형 (11/17 규칙에 더함, T28 3번)
            (f"{josa(surface, '은/는')} 안 씁니다.", f"{surface} 안 씁니다"), (f"{josa(surface, '은/는')} 안 썼음.", f"{surface} 안 썼어요"),
            (f"{josa(surface, '은/는')} 쓰지 않음.", f"{surface} 쓰지 않아요"), (f"{josa(surface, '은/는')} 필요하지 않음.", f"{surface} 필요하지 않아요"),
            (f"{josa(surface, '은/는')} 투입하지 않음.", f"{surface} 투입하지 않습니다"), (f"{josa(surface, '은/는')} 쓰지 않기로 함.", f"{surface} 쓰지 않기로 했어요"),
            # '쓸 일 없'·'사용 금지' (11/17 규칙에 더함, T30)
            (f"{surface} 쓸 일 없음.", f"{surface} 쓸 일 없어요"), (f"{surface} 쓸 계획 없음.", f"{surface} 쓸 계획 없습니다"),
            (f"{josa(surface, '은/는')} 쓸 필요가 없음.", f"{surface} 쓸 필요가 없어요"), (f"{surface} 쓸 일 없어서 다른 방법으로 진행.", f"{surface} 쓸 일 없어서 다른 방법으로 해요"),
            (f"{surface} 사용 금지.", f"{surface} 사용 금지예요"), (f"{surface} 사용 금지 상태라 수공구로 진행.", f"{surface} 사용 금지라서 수공구로 해요"),
        ]
        text, answer = self.r.choice(forms)
        return Piece("equip", text, answer, excluded=[surface])

    def plan_equipment(self, surface: str, key: str) -> Piece:
        """쓰겠다고 했다가 뒤 발화에서 거두는 것의 앞 조각 (부정 없이 한 말)."""
        act, eo = self.act(key), josa(surface, "으로/로")
        text = self.r.choice([
            f"처음엔 {eo} {act}하려고 했음.", f"{eo} {act}할 계획이었는데.", f"{eo} {act}하기로 정해 뒀었음.", f"{surface} 가져오려 했는데.", f"원래는 {eo} {act} 예정이었음.",
        ])
        return Piece("equip", text)

    def plan_material(self, surface: str) -> Piece:
        eo = josa(surface, "으로/로")
        text = self.r.choice([f"처음엔 {eo} 닦으려고 했음.", f"{eo} 세척할 계획이었는데.", f"원래 {surface} 쓰기로 했었음.", f"{eo} 청소하려 했는데."])
        return Piece("mat", text)

    def replace_equipment(self, old: str, new: str, key: str) -> Piece:
        act = self.act(key)
        forms = [
            f"{old} 말고 {josa(new, '으로/로')} {act}함.", f"{old} 대신 {new} 사용해서 {act}.", f"{josa(old, '이/가')} 아니라 {josa(new, '으로/로')} {act} 진행.",
            f"{old} 대신 {josa(new, '으로/로')} {act}.", f"{old} 말고 {new} 쓸 예정.",
        ]
        return Piece("equip", self.r.choice(forms), equipment=[new], excluded=[old])

    def replace_material(self, old: str, new: str) -> Piece:
        forms = [f"{old} 말고 {josa(new, '으로/로')} 닦아냄.", f"{old} 대신 {josa(new, '으로/로')} 세척.", f"{josa(old, '은/는')} 안 쓰고 {new} 사용."]
        return Piece("mat", self.r.choice(forms), materials=[new], excluded=[old])

    def negate_material_generic(self) -> Piece:
        return Piece("mat", self.r.choice(["화학물질 취급 없음.", "위험물은 반입하지 않음.", "취급 약품 없음.", "물질은 없음."]), "물질 없음")

    def conditional(self, surface: str, key: str, material: str | None = None) -> Piece:
        trigger, stem = self.r.choice([
            ("볼트가 고착되면", "고착 시"), ("플랜지가 고착되면", "고착 시"), ("필요하면", "필요 시"), ("누설이 확인되면", "누설 시"),
            ("파손되어 있으면", "파손 시"), ("안 빠지면", ""), ("높이가 안 나오면", ""), ("막혀 있으면", ""), ("체결이 안 되면", ""),
        ])
        act = self.act(key)
        eo = josa(surface, "으로/로")
        text = self.r.choice([f"{trigger} {eo} {act}할 예정.", f"{trigger} {eo} {act}함.", f"{trigger} {surface} 사용해서 {act}.", f"{trigger} {eo} {act} 진행."])
        summary = f"{stem} {surface} {act}".strip()
        pieces_text = text
        if material:
            pieces_text += " " + self.mention_material(material).text
        return Piece("equip", pieces_text, equipment=[surface], materials=[material] if material else [], summary=summary)

    # ---- 말이 많은 문장 (verbose)
    def verbose_task(self) -> tuple[Piece, str]:
        """과거형 서술 작업 조각. 요약은 '장소 작업'만 (끝맺음·장소 뒷말은 뺀다)."""
        loc, suffix = self.location(), self.r.choice(VERBOSE_PLACE_SUFFIX)
        forms = [t for t in TASK_FORMS if not t.endswith("조임")]  # '조임했음'처럼 어색한 것은 뺀다
        core = self.r.choice(forms).format(p=self.part()) if self.r.random() < 0.6 else self.r.choice(TASK_PLAIN)
        tail = self.r.choice(VERBOSE_ENDINGS) if self.r.random() < 0.6 else ("을" if _batchim(core) else "를") + " " + self.r.choice(VERBOSE_OBJECT_ENDINGS)
        text = f"{loc}{suffix}에서 {core}{tail}."
        head = self.r.choice(HEADS) + " " if self.r.random() < 0.5 else ""
        return Piece("task", head + text), f"{loc} {core}"

    def verbose_equipment(self, surface: str, key: str) -> Piece:
        act, eo, why = self.act(key), josa(surface, "으로/로"), self.r.choice(VERBOSE_REASONS)
        text = self.r.choice([
            f"{why} {surface} 가져와서 {act} 진행했고.", f"{why} {eo} {act}했음.", f"{surface} 가져와서 {act}했고.", f"{surface} 사용해서 {act} 마침.",
            f"{eo} {act}하고 정리함.", f"{why} {surface} 올려서 {act}했음.",
        ])
        return Piece("equip", text, equipment=[surface])

    def verbose_material(self, m: str) -> Piece:
        eo = josa(m, "으로/로")
        text = self.r.choice([
            f"옆에서 {eo} 닦음.", f"{m} 조금 흘려서 바로 닦았음.", f"{eo} 세척했고.", f"{m} 희석해서 사용했음.", f"{m} 말통 옮기다가 잠깐 쉬었음.",
            f"{m} 냄새가 약간 났음.",
        ])
        return Piece("mat", text, materials=[m])

    def misc(self) -> Piece:
        return Piece("misc", self.r.choice(EXTRAS).format(n=self.n(2, 5)))


# ---------------------------------------------------------------- 유형별 장면


def _task_summary(piece_summary: str, extras: list[Piece]) -> str:
    parts = [piece_summary] + [p.summary for p in extras if p.summary]
    return ", ".join(parts)


def build_sample(mk: Maker, kind: str) -> Sample:
    r = mk.r
    task, base = mk.task()
    pieces = [task]
    extras: list[Piece] = []
    if kind == "plain":
        for _ in range(r.randint(0, 2)):
            e, k = mk.equipment(ood=False if r.random() < 0.6 else None)
            if not used(pieces, e):
                pieces.append(mk.use_equipment(e, k))
        for _ in range(r.randint(0, 2)):
            m = mk.material(ood=False if r.random() < 0.7 else None)
            if not used(pieces, m):
                pieces.append(mk.mention_material(m))
    elif kind == "out_of_dict":
        for _ in range(r.randint(0, 1)):
            e, k = mk.equipment(ood=True)
            pieces.append(mk.use_equipment(e, k))
        for _ in range(r.randint(0 if pieces[1:] else 1, 2)):
            m = mk.material(ood=True)
            if not used(pieces, m):
                pieces.append(mk.mention_material(m))
        if r.random() < 0.3:  # 사전 안의 것도 하나 섞는다
            e, k = mk.equipment(ood=False)
            if not used(pieces, e):
                pieces.append(mk.use_equipment(e, k))
    elif kind == "negation":
        mode = r.choice(["equip_neg", "equip_neg", "replace_eq", "replace_eq", "soft_neg", "mat_replace", "mat_generic", "retract_eq", "retract_pair", "retract_mat"])
        if mode == "equip_neg":
            e, _ = mk.equipment(ood=False if r.random() < 0.75 else None)
            pieces.append(mk.negate_equipment(e))
            e2, k2 = mk.equipment(exclude=(e,))
            pieces.append(mk.use_equipment(e2, k2))
        elif mode == "soft_neg":
            e, _ = mk.equipment(ood=False)
            pieces.append(mk.negate_equipment(e, soft=True))
            e2, k2 = mk.equipment(exclude=(e,))
            pieces.append(mk.use_equipment(e2, k2))
        elif mode == "replace_eq":
            old, _ = mk.equipment(ood=False)
            new, key = mk.equipment(exclude=(old,))
            pieces.append(mk.replace_equipment(old, new, key))
        elif mode == "mat_replace":
            old = mk.material(ood=False)
            new = mk.material(exclude=(old,))
            pieces.append(mk.replace_material(old, new))
        elif mode == "retract_eq":  # 앞 발화에서 쓰겠다고 했다가 뒤 발화에서 바꾼다 (서버 검증은 이 거둠을 못 본다)
            old, key = mk.equipment(ood=False if r.random() < 0.7 else None)
            new, nkey = mk.equipment(exclude=(old,))
            pieces.append(mk.plan_equipment(old, key))
            retract = mk.replace_equipment(old, new, nkey) if r.random() < 0.6 else mk.negate_equipment(old)
            retract.later = True
            pieces.append(retract)
            if not retract.equipment and r.random() < 0.7:  # 거두기만 했으면 다른 장비를 쓴다고 이어 말하기도 한다
                pieces.append(mk.use_equipment(new, nkey))
        elif mode == "retract_pair":  # 둘을 준비한다고 했다가 하나만 쓴다
            keep, kkey = mk.equipment(ood=False)
            drop, _ = mk.equipment(ood=False if r.random() < 0.7 else None, exclude=(keep,))
            pieces.append(Piece("equip", r.choice([f"{keep}{r.choice(['랑', '하고'])} {drop} 준비해 둠.", f"{drop} 그리고 {keep} 같이 준비함."])))
            tail = r.choice([f"{drop} 말고 {keep}만 쓸 거야.", f"{drop} 대신 {keep}만 사용함.", f"{josa(drop, '은/는')} 안 쓰고 {keep}만 씀."])
            pieces.append(Piece("equip", tail, equipment=[keep], excluded=[drop], later=True))
        elif mode == "retract_mat":  # 주변 물질 슬롯은 뒤 발화에서 거둬도 빠지지 않는다(merge_slots 는 장비만 뺀다) → 슬롯과 어긋나지 않게 사전 밖 물질만
            old = mk.material(ood=True)
            new = mk.material(exclude=(old,)) if r.random() < 0.7 else None
            pieces.append(mk.plan_material(old))
            if new is not None:
                retract = mk.replace_material(old, new)
            else:
                retract = Piece("mat", r.choice([f"{josa(old, '은/는')} 안 쓰고 진행함.", f"{old} 사용 안 하기로 함.", f"{josa(old, '은/는')} 쓰지 않는 걸로 바뀜"]), excluded=[old], soft=False)
            retract.later = True
            pieces.append(retract)
        else:
            pieces.append(mk.negate_material_generic())
            if r.random() < 0.6:
                e, k = mk.equipment()
                pieces.append(mk.use_equipment(e, k))
        if r.random() < 0.5 and mode != "mat_generic":
            m = mk.material(ood=False if r.random() < 0.7 else None)
            if not used(pieces, m):
                pieces.append(mk.mention_material(m))
    elif kind == "conditional":
        e, k = mk.equipment(ood=False if r.random() < 0.7 else None)
        m = mk.material(ood=False if r.random() < 0.7 else None) if r.random() < 0.55 else None
        cond = mk.conditional(e, k, m)
        pieces.append(cond)
        extras.append(cond)
        if r.random() < 0.4:
            e2, _ = mk.equipment(ood=False, exclude=(e,))
            pieces.append(mk.negate_equipment(e2))
            if m is not None and ckey(m) == ckey(e2):
                pieces.pop()
    elif kind == "multi":
        pick = r.choice(["materials", "equipment", "both"])
        if pick in ("materials", "both"):
            names: list[str] = []
            for _ in range(r.randint(2, 4)):
                names.append(mk.material(ood=False if r.random() < 0.7 else None, exclude=tuple(names)))
            joiner = ", ".join(names[:-1]) + (", " if len(names) > 2 else "") + names[-1]
            text = r.choice([f"{joiner}를 함께 쓰는 라인.", f"{joiner} 취급 설비 옆에서 작업.", f"{joiner} 남아 있을 수 있음.", f"{joiner} 보관 중."])
            if len(names) == 2 and r.random() < 0.5:
                text = f"{names[0]}{'과' if _batchim(names[0]) else '와'} {names[1]} 남아 있을 수 있음."
            pieces.append(Piece("mat", text, f"{', '.join(names)} 있어요", materials=list(names)))
        if pick in ("equipment", "both"):
            names_e: list[str] = []
            keys: list[str] = []
            for _ in range(r.randint(2, 4)):
                s, k = mk.equipment(ood=False if r.random() < 0.7 else None, exclude=tuple(names_e) + tuple(keys))
                names_e.append(s)
                keys.append(k)
            text = r.choice([f"{', '.join(names_e)} 동원.", f"{', '.join(names_e)} 투입.", f"{', '.join(names_e)} 사용."])
            if r.random() < 0.4:  # '용접기 2대, 그라인더 3대'처럼 수량
                text = ", ".join(f"{s} {mk.n(2, 4)}대" for s in names_e[:2]) + " 투입."
                names_e = names_e[:2]
            pieces.append(Piece("equip", text, f"장비는 {', '.join(names_e)}", equipment=list(names_e)))
    elif kind == "verbose":
        task, base = mk.verbose_task()
        pieces = [task]
        for _ in range(r.randint(0, 2)):
            e, k = mk.equipment(ood=False if r.random() < 0.6 else None)
            if not used(pieces, e):
                pieces.append(mk.verbose_equipment(e, k))
        for _ in range(r.randint(0, 2)):
            m = mk.material(ood=False if r.random() < 0.6 else None)
            if not used(pieces, m):
                pieces.append(mk.verbose_material(m))
        if r.random() < 0.4:  # 인원 이야기. 요약에 숫자를 넣는 틀도 한 번씩 보인다 (숫자는 원문에 있는 것만)
            count = mk.n(2, 6)
            pieces.append(Piece("misc", f"작업자 {count}명 투입했고 무사히 끝남."))
            if r.random() < 0.4:
                base = f"{base}, {count}명"
        for _ in range(r.randint(0, 2)):
            pieces.append(Piece("misc", r.choice(VERBOSE_FILLERS)))
        rest = pieces[1:]
        r.shuffle(rest)  # 잡담·장비·물질 문장의 순서는 섞는다 (작업 문장이 늘 맨 앞)
        pieces = [pieces[0], *rest]
    elif kind == "none":
        text = r.choice(NONE_TASKS)
        place = mk.location() + " " if r.random() < 0.5 else ""
        head = r.choice(HEADS) + " " if r.random() < 0.4 else ""
        pieces = [Piece("task", f"{head}{place}{text} {r.choice(['진행', '함', '예정', '실시', '완료'])}.")]
        base = f"{place}{text}"
    elif kind == "distractor":
        tag = mk.tag()
        part1, part2 = mk.part(), mk.part()
        kind_pick = r.choice(["tag", "tag", "drum", "parts"])
        if kind_pick == "drum":
            m = mk.material(ood=False if r.random() < 0.6 else None)
            text, base = r.choice([
                (f"{m} 드럼 {mk.n(2, 6)}개를 보관 구역으로 이동. 빈 드럼은 따로 모음.", f"{m} 드럼 이동"),
                (f"{m} 드럼 {mk.n(2, 6)}개를 야적장으로 옮김. 빈 드럼 정리.", f"{m} 드럼 옮김"),
                (f"빈 드럼 회수. {m} 드럼은 별도 구역에 보관.", "빈 드럼 회수"),
            ])
            pieces = [Piece("task", text, materials=[m])]
        else:
            body = f"{tag} {r.choice(['탱크', '펌프', '반응기', '배관', '설비'])} {part1} {r.choice(['교체', '점검', '보수'])}"
            text = f"{body}. {part2} 점검도 같이 함." if kind_pick == "tag" else f"{body}. {part2} 확인하고 {mk.part()} 교환."
            pieces = [Piece("task", text + (f" {mk.location()}에서 작업." if r.random() < 0.5 else ""))]
            base = body
    else:
        raise ValueError(kind)
    if kind not in ("distractor", "verbose") and r.random() < 0.3:
        pieces.append(mk.misc())
    return Sample(kind, pieces, _task_summary(base, extras) if kind != "none" else base)


# ---------------------------------------------------------------- 발화로 나누기·슬롯·정답


def _join(texts: list[str]) -> str:
    """한 발화로 잇는다. 앞 문장에 마침표가 없으면 붙인다 ('점검 지게차 3대'처럼 붙어 버리지 않게)."""
    out = ""
    for text in texts:
        out = text if not out else f"{out} {text}" if out.endswith((".", "!", "?")) else f"{out}. {text}"
    return out


def to_utterances(mk: Maker, sample: Sample) -> list[str]:
    """한 번에 다 말하거나, 작업만 먼저 말하고 나머지는 되묻기에 답하듯 나눠 말한다."""
    pieces = [p for p in sample.pieces if not p.later]
    withdrawals = [p for p in sample.pieces if p.later]  # 앞에서 한 말을 거두는 조각은 늘 뒤 발화로 따로 (분할 확률과 상관없이)
    splittable = [p for p in pieces[1:] if p.answer]
    if mk.r.random() >= MULTI_LINE_RATE or not splittable:
        head = [_join([p.text for p in pieces])]
    else:
        first = [pieces[0]] + [p for p in pieces[1:] if not p.answer]
        later = [p for p in pieces[1:] if p.answer]
        head = [_join([p.text for p in first])] + [p.answer or p.text for p in later]
    return head + [_join([p.text]) for p in withdrawals]


def slots_for(utterances: list[str]) -> WorkSlots:
    slots: WorkSlots | None = None
    for message in utterances:
        slots = merge_slots(slots, message, today=TODAY)
    assert slots is not None
    return slots


def gold_of(sample: Sample) -> tuple[list[str], list[str], list[str]]:
    equipment: list[str] = []
    materials: list[str] = []
    excluded: list[str] = []
    for p in sample.pieces:
        equipment += [e for e in p.equipment if e not in equipment]
        materials += [m for m in p.materials if m not in materials]
        excluded += p.excluded
    return equipment, materials, excluded


def _canon_set(items: list[str], keys: dict[str, str]) -> set[str]:
    return set(pb._canonical(items, keys))  # noqa: SLF001 - 서버의 표준 이름 맞추기와 같은 함수


def check_row(sample: Sample, utterances: list[str], slots: WorkSlots, summary: str, equipment: list[str], materials: list[str]) -> None:
    """정답이 서버의 검증을 모두 통과하는지, 슬롯 줄이 정답과 어긋나지 않는지 본다. 어긋나면 AssertionError (만든 쪽 실수)."""
    text = full_text(slots)[-MAX_CONTEXT_CHARS:]
    where = f"[{sample.kind}] {text!r}"
    assert text == "\n".join(utterances), where
    # 서버 검증 (plan_brief): 버려지는 항목이 하나도 없어야 하고, 표준 이름으로 맞춘 결과가 정답과 같아야 한다
    kept_e, dropped_e = pb.clean_equipment(equipment, text)
    kept_m, dropped_m = pb.clean_materials(materials, text)
    assert dropped_e == 0 and dropped_m == 0, (where, equipment, materials, kept_e, kept_m)
    assert set(kept_e) == _canon_set(equipment, pb._EQUIPMENT_KEYS), where  # noqa: SLF001
    assert set(kept_m) == _canon_set(materials, pb._MATERIAL_KEYS), where  # noqa: SLF001
    assert len(equipment) <= pb.MAX_BRIEF_ITEMS and len(materials) <= pb.MAX_BRIEF_ITEMS, where
    # 같은 사전 항목을 두 표기로 중복해 적지 않는다 (서버가 하나로 합치므로 정답 목록과 서버 결과의 길이가 어긋난다)
    assert len(_canon_set(equipment, pb._EQUIPMENT_KEYS)) == len(equipment), (where, equipment)  # noqa: SLF001
    assert len(_canon_set(materials, pb._MATERIAL_KEYS)) == len(materials), (where, materials)  # noqa: SLF001
    assert pb.clean_summary(summary, text) == summary and 4 <= len(summary) <= pb.BRIEF_SUMMARY_CHARS, (where, summary)
    support, unsupported = summary_support(summary, text)
    assert support == 1.0, (where, summary, unsupported)
    # 정답에 있는 항목은 원문에서 부정되지 않았고, 쓰지 않는다고 한 항목은 정답에 없다
    excluded = [x for p in sample.pieces for x in p.excluded]
    assert not set(excluded) & set(equipment + materials), where
    assert all(x in text for x in excluded), where
    # 슬롯 줄 = 사전 단어 중 쓰는 것. 정답과 어긋나면 안 된다 (사전 단어를 품은 긴 이름은 T25 빈틈이라 만들지 않는다)
    soft = {x for p in sample.pieces if p.soft for x in p.excluded}
    want_e = {k for k in _canon_set(equipment, pb._EQUIPMENT_KEYS) if k in EQUIPMENT}  # noqa: SLF001
    want_e |= _canon_set(sorted(soft), pb._EQUIPMENT_KEYS) & set(EQUIPMENT)  # noqa: SLF001
    want_m = {k for k in _canon_set(materials, pb._MATERIAL_KEYS) if k in MATERIALS}  # noqa: SLF001
    assert set(slots.equipment) == want_e, (where, slots.equipment, want_e)
    assert set(slots.nearby_materials) == want_m, (where, slots.nearby_materials, want_m)
    # 한 줄 안의 여러 발화에서 부정이 다음 발화로 넘어가지 않는다는 전제 (슬롯이 이미 지킴). 쓰는 것이 부정 목록에 있으면 안 된다
    negated_e = {c for u in utterances for c in find_terms(u, EQUIPMENT)[1]}
    assert not negated_e & set(want_e) or soft, where


def make_row(mk: Maker, kind: str) -> dict:
    sample = build_sample(mk, kind)
    utterances = to_utterances(mk, sample)
    slots = slots_for(utterances)
    equipment, materials, excluded = gold_of(sample)
    check_row(sample, utterances, slots, sample.summary, equipment, materials)
    context = WorkContext(
        text=full_text(slots)[-MAX_CONTEXT_CHARS:], location=slots.location, hot_work_kind=slots.hot_work_kind,
        nearby_materials=slots.nearby_materials, headcount=slots.headcount, end_time=slots.end_time, equipment=slots.equipment,
    )
    target = {"summary": sample.summary, "equipment": equipment, "materials": materials}
    return {
        "messages": [
            {"role": "system", "content": ADAPTER_SYSTEM_PROMPT},
            {"role": "user", "content": build_adapter_input(context)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "kind": kind,
        "utterances": utterances,
        "answer": {**target, "excluded": excluded},
    }


# ---------------------------------------------------------------- 평가 문장과 겹침 막기


def _grams(t: str) -> set[str]:
    t = "".join(t.split())
    return {t[i : i + 3] for i in range(max(1, len(t) - 2))}


def _sentences(text: str) -> list[str]:
    """문장 단위 (마침표·줄바꿈 기준, 공백 없이). 너무 짧은 조각(예: '3명')은 우연히 겹칠 수 있어 6자 이상만 본다."""
    parts = ["".join(x.split()) for x in re.split(r"[.!?\n]+", text)]
    return [x for x in parts if len(x) >= 6]


def similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


@dataclass
class EvalGuard:
    texts: list[str]
    grams: list[set[str]]
    items: set[str]  # 평가 정답에 나오는 물질·장비 표기
    sentences: frozenset[str] = frozenset()  # 평가 문장을 마침표로 나눈 문장 (공백 없이)
    ood_items: set[str] = field(default_factory=set)  # 그중 사전에 없는 것 (염산, 핸드브레이커 …)

    @classmethod
    def load(cls, path: Path | str | Sequence[Path | str] = DEFAULT_EVAL_SETS) -> EvalGuard:
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        cases = [c for p in paths for c in load_cases(p)]
        items = {x for c in cases for x in (*c.gold_equipment, *c.gold_materials, *c.excluded)}
        texts = [c.text for c in cases]
        ood = {x for x in items if not find_terms(x, EQUIPMENT)[0] and not find_terms(x, MATERIALS)[0]}
        return cls(
            texts=texts, grams=[_grams(t) for t in texts], items=items, ood_items=ood,
            sentences=frozenset(s for t in texts for s in _sentences(t)),
        )

    def too_close(self, text: str, threshold: float) -> bool:
        flat = " ".join(text.split())
        if any(flat == " ".join(t.split()) for t in self.texts):
            return True
        if self.sentences & set(_sentences(text)):  # 문장 하나라도 평가 문장과 똑같으면 버린다
            return True
        g = _grams(text)
        return any(similarity(g, e) >= threshold for e in self.grams)

    def max_similarity(self, text: str) -> float:
        g = _grams(text)
        return max((similarity(g, e) for e in self.grams), default=0.0)


def check_vocab(guard: EvalGuard) -> None:
    """사전 밖 이름이 평가 정답·평가 문장 본문과 겹치지 않고, 사전 단어를 품지 않는지 (품으면 슬롯 줄이 정답과 어긋난다)."""
    for name in (*OOD_EQUIPMENT_TRAIN, *OOD_EQUIPMENT_VALID):
        assert name not in guard.items, f"평가 정답과 겹침: {name}"
        assert not any(name in item or item in name for item in guard.ood_items), f"평가 정답과 부분 겹침: {name}"
        assert not any(name in text for text in guard.texts), f"평가 문장 본문에 나옴 (정답이 아니어도 겹침, 예: 쓰지 않는다고 거둔 말): {name}"
        assert not find_terms(name, EQUIPMENT)[0] and not find_terms(name, MATERIALS)[0], f"사전 단어를 품음: {name}"
        assert name in EQUIP_ACTS, f"일 목록이 없음: {name}"
    for name in (*OOD_MATERIALS_TRAIN, *OOD_MATERIALS_VALID):
        assert name not in guard.items, f"평가 정답과 겹침: {name}"
        assert not any(name in item or item in name for item in guard.ood_items), f"평가 정답과 부분 겹침: {name}"
        assert not any(name in text for text in guard.texts), f"평가 문장 본문에 나옴 (정답이 아니어도 겹침, 예: 쓰지 않는다고 거둔 말): {name}"
        assert not find_terms(name, EQUIPMENT)[0] and not find_terms(name, MATERIALS)[0], f"사전 단어를 품음: {name}"
    assert not set(OOD_EQUIPMENT_TRAIN) & set(OOD_EQUIPMENT_VALID) and not set(OOD_MATERIALS_TRAIN) & set(OOD_MATERIALS_VALID)


# ---------------------------------------------------------------- 생성


def require_polite_negation() -> None:
    """정중형 부정('안 씁니다', '쓰지 않음' 등)을 규칙이 잡아야 이 생성기의 정답이 서버와 맞는다 (11/17 text_rules.POLITE_NEGATION, T28 3번).
    규칙이 꺼졌거나 옛 app 이면 슬롯 줄에 거둔 장비가 남아 만들 때 이유를 알기 어려운 오류가 나므로 먼저 알려 준다."""
    if not getattr(text_rules, "POLITE_NEGATION", False):
        raise RuntimeError("app/agent/text_rules.py 에 POLITE_NEGATION = True 가 없습니다. apply_polite_negation.py 를 먼저 적용하세요 "
                           "(Colab 은 app.zip 을 PC 의 현재 app/ 로 다시 묶어 올립니다).")


def require_rules_v2() -> None:
    """사전 확장(사다리차·황산 …)과 '쓸 일 없'·'사용 금지' 부정이 켜져 있어야 이 생성기의 정답이 서버와 맞는다 (11/17 text_rules, T30)."""
    missing = [name for name in ("EXTENDED_VOCAB", "PROHIBIT_NEGATION") if not getattr(text_rules, name, False)]
    if missing:
        raise RuntimeError(f"app/agent/text_rules.py 에 {', '.join(missing)} = True 가 없습니다. apply_rules_v2.py 를 먼저 적용하세요 "
                           "(Colab 은 app.zip 을 PC 의 현재 app/ 로 다시 묶어 올립니다).")


def generate(n: int, seed: int, guard: EvalGuard, *, holdout: bool = False, seen: set[str] | None = None, threshold: float = 0.6) -> list[dict]:
    require_polite_negation()
    require_rules_v2()
    mk = Maker(seed, holdout=holdout)
    kinds = list(KIND_WEIGHTS)
    weights = [KIND_WEIGHTS[k] for k in kinds]
    seen = seen if seen is not None else set()
    rows: list[dict] = []
    tries = 0
    while len(rows) < n:
        tries += 1
        if tries > n * 60:
            raise RuntimeError(f"서로 다른 문장을 {n}건 만들지 못했습니다 ({len(rows)}건에서 멈춤). 재료가 모자랍니다.")
        row = make_row(mk, mk.r.choices(kinds, weights)[0])
        text = "\n".join(row["utterances"])
        if text in seen or guard.too_close(text, threshold):
            continue
        seen.add(text)
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="계획 단계 어댑터 학습 데이터 생성", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="저장 폴더 (기본 data/plan-train, git 에 안 올라감)")
    parser.add_argument("--train", type=int, default=2400)
    parser.add_argument("--valid", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-set", nargs="+", default=[str(p) for p in DEFAULT_EVAL_SETS],
                        help="겹치면 안 되는 평가 문장 파일들 (기본 eval_set.jsonl · eval_set_multiline.jsonl · eval_set_user_hard.jsonl)")
    parser.add_argument("--threshold", type=float, default=0.6, help="평가 문장과 3글자 조각 겹침이 이 값 이상이면 버린다")
    parser.add_argument("--sample", type=int, default=0, help="유형별 N건씩 화면에만 보여 주고 끝 (저장 안 함)")
    args = parser.parse_args(argv)

    guard = EvalGuard.load(args.eval_set)
    check_vocab(guard)
    seen: set[str] = set()
    train = generate(args.train, args.seed, guard, seen=seen, threshold=args.threshold)
    valid = generate(args.valid, args.seed + 1, guard, holdout=True, seen=seen, threshold=args.threshold)

    if args.sample:
        shown: Counter = Counter()
        for row in [*train, *valid]:
            if shown[row["kind"]] < args.sample:
                shown[row["kind"]] += 1
                print(f"[{row['kind']}] " + " ⏎ ".join(row["utterances"]))
                print(f"   입력 줄: {row['messages'][1]['content'].splitlines()[1:] or '(없음)'}")
                print(f"   정답: {row['messages'][2]['content']}")
        return 0

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    print(f"저장: {args.out / 'train.jsonl'} {len(train)}건, {args.out / 'valid.jsonl'} {len(valid)}건 (seed {args.seed})")
    for name, rows in (("train", train), ("valid", valid)):
        stats = Counter(r["kind"] for r in rows)
        multi = sum(len(r["utterances"]) > 1 for r in rows)
        print(f"  {name}: " + ", ".join(f"{k} {stats[k]}" for k in _KINDS) + f" / 여러 발화 {multi}건")
    worst = max((guard.max_similarity("\n".join(r["utterances"])) for r in train), default=0.0)
    print(f"  평가 문장과 가장 비슷한 학습 문장의 3글자 겹침: {worst:.2f} (기준 {args.threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
