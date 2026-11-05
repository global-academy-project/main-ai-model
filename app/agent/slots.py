"""되묻기: 발화에서 필수 항목을 읽고(규칙), 빠진 항목을 체크리스트로 고른다.

필수 항목 (지침 2장 5번, 계획서 9장 시나리오 A의 되묻기 항목)
  - 작업 장소                       항상
  - 화기 종류                       화기작업일 때
  - 주변 취급 물질 ('없음'도 답)     화기·밀폐공간 작업일 때
  - 인원 · 시작 시간 · 종료 시간      항상 (Spring은 시작·종료가 모두 있어야 허가서를 만든다, 계획서 7-2장)
  - 투입 장비 ('안 씀'·'없음'도 답)  항상
  - 작업 내용                       작업유형을 하나도 못 읽었을 때

한 번만 묻는 항목 (OPTIONAL_KEYS, 답이 없어도 초안은 만든다)
  - 가연성 가스(LEL) 측정값          화기·밀폐공간 작업일 때. 다른 빠진 항목을 물을 때 함께 묻고(ASK_LEL_ALONE=False),
                                    한 번 물었으면 다시 묻지 않는다 (slots.asked_optional_keys).
    계획서 7-3장: 측정은 작업 직전 현장에서 하므로 측정값이 없어도 초안을 만들고 승인 요청만 막는다.
    그래서 LEL은 WAITING_INPUT의 조건이 아니다. '측정 전'·'없음' 같은 말을 0으로 바꾸지 않는다 (숫자만 읽음).

읽기만 하는 항목 (묻지 않음): 작업 날짜(workDate, work_date.py), 다른 측정값, 화재감시자·감시인 이름.

새 발화에서 읽은 값만 덮어쓰고, 나머지는 이전 slots 값을 유지한다. 목록은 합친다.
직전에 되물은 항목(slots.asked_keys)이 있으면 "3", "5시", "없음"처럼 짧은 답도 그 항목의 답으로 읽는다.
어느 항목의 답인지 정할 수 없으면 읽지 않는다 (예: 인원과 LEL을 함께 물었는데 "3", 시작·종료를 함께 물었는데 "5시").
LLM 판단은 2주차(계획서 8장). 지금은 text_rules의 키워드·정규식만 쓴다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from app.schemas.agent_work import MAX_UTTERANCES, MissingField, WorkSlots

from .clarify import MARKER_PATTERN, TIME_KEYS, ampm_fields, is_unmarked, resolve_answer, to_24h
from .text_rules import EQUIPMENT, MATERIALS, extract_locations, find_terms, is_negated, sentences, unique
from .work_date import extract_work_date

# 한 번만 묻는 항목. 답이 없어도 WAITING_INPUT으로 막지 않는다.
OPTIONAL_KEYS = frozenset({"lel"})
# True면 다른 빠진 항목이 없어도 LEL 하나만 묻는 턴을 만든다. False면 다른 항목을 물을 때 함께 묻는다
# (첫 발화에 필수 항목이 다 있으면 LEL 때문에 대화가 한 턴 늘지 않는다).
ASK_LEL_ALONE = False
# 가연성 가스(LEL)를 묻는 작업유형 (C 규칙: HOT_LEL_MAX, CONFINED_LEL_MAX)
LEL_WORK_TYPES = frozenset({"화기", "밀폐공간"})

_HOUR = r"(" + MARKER_PATTERN + r")?\s*(\d{1,2})\s*시\s*(?:(\d{1,2})\s*분|(반))?"
_CLOCK = r"(\d{1,2}):(\d{2})"
# '17시 30분에 끝나요'의 '에'를 시작 시각으로 읽지 않게, 뒤에 끝·종료·마감이 오면 시작이 아니다.
_START_RE = re.compile(_HOUR + r"\s*(?:에|부터)(?!\s*(?:끝|종료|마감))")
# '10~12시', '오후 1시~3시': 물결 앞이 시작이다. 앞쪽은 '시'를 생략해도 된다 ('10~12시')
_START_TILDE_RE = re.compile(
    r"(" + MARKER_PATTERN + r")?\s*(\d{1,2})\s*(?:시\s*(?:(\d{1,2})\s*분|(반))?)?\s*[~∼]\s*(?:" + MARKER_PATTERN + r")?\s*\d{1,2}\s*시"
)
_END_RE = re.compile(_HOUR + r"\s*(?:에\s*)?(?:까지|종료|마감|끝)")
_END_TILDE_RE = re.compile(r"[~∼]\s*" + _HOUR)
_START_CLOCK_RE = re.compile(_CLOCK + r"\s*(?:에|부터)(?!\s*(?:끝|종료|마감))")
_START_CLOCK_TILDE_RE = re.compile(_CLOCK + r"\s*[~∼]\s*\d{1,2}:\d{2}")
_END_CLOCK_RE = re.compile(_CLOCK + r"\s*(?:에\s*)?(?:까지|종료|마감|끝)")
_END_CLOCK_TILDE_RE = re.compile(r"[~∼]\s*" + _CLOCK)
_HEADCOUNT_RE = re.compile(r"(\d{1,3})\s*(?:명|인)(?!\s*분|치)")
_KO_NUMBERS = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_HEADCOUNT_KO_RE = re.compile(r"(?<![가-힣])(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:명|인)(?!\s*분|치)")
# 직전 질문에 대한 짧은 답 (문장 전체가 이것뿐일 때만)
_ANSWER_TAIL = r"\s*(?:이요|요|입니다|이에요|예요)?\s*[.!]?\s*$"
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,3})\s*(명|인)?" + _ANSWER_TAIL)
_PCT_UNIT = r"(?:%\s*LEL|%|퍼센트|프로)"
_BARE_LEL_RE = re.compile(
    r"^\s*(LEL|가연성\s*가스)?\s*(?:농도|측정값)?\s*(?:는|은|이|가|:)?\s*(\d+(?:\.\d+)?)\s*(" + _PCT_UNIT + r")?" + _ANSWER_TAIL,
    re.IGNORECASE,
)
_BARE_TIME_RE = re.compile(r"^\s*" + _HOUR + r"\s*(?:쯤|정도)?" + _ANSWER_TAIL)
_BARE_CLOCK_RE = re.compile(r"^\s*" + _CLOCK + r"\s*(?:쯤|정도)?" + _ANSWER_TAIL)
_BARE_NONE_RE = re.compile(r"^\s*(?:없(?:음|어요|습니다|네요|어|다)|해당\s*없(?:음|어요|습니다))\s*[.!]?\s*$")
_BARE_NOT_USED_RE = re.compile(
    r"^\s*(?:안\s*(?:씀|써요?|쓰요|씁니다)|사용\s*(?:안\s*(?:함|해요?|합니다)|하지\s*않(?:음|아요|습니다)))\s*[.!]?\s*$"
)
_MATERIALS_NONE_RE = re.compile(r"(?:물질|가연물|위험물)\s*(?:은|는|이|도)?\s*(?:없|무)")
_EQUIPMENT_NONE_RE = re.compile(r"장비\s*(?:은|는|도)?\s*(?:없|안\s*씀|안\s*써|필요\s*없)")
# 숫자 뒤에 기준 표현이 오면 측정값이 아니라 기준을 말한 것이다 ('산소 18% 이상이어야', 'LEL 10% 미만 유지').
# 기준을 측정값으로 읽으면 규칙이 PASS로 판정할 수 있어서 읽지 않는다.
_NOT_CRITERION = r"(?!\s*(?:이하|미만|이상|초과|넘|기준|까지))"
_MEASUREMENT_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("O2", re.compile(r"(?:산소|(?<!C)O2)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*%" + _NOT_CRITERION)),
    ("LEL", re.compile(
        r"(?:LEL|가연성\s*가스)\s*(?:농도|측정값|수치)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*" + _PCT_UNIT + _NOT_CRITERION,
        re.IGNORECASE,
    )),
    # 'LEL 0', 'LEL은 3이에요': LEL이라고 직접 말했으면 단위가 없어도 %LEL로 읽는다 (단위 고정, 계획서 7-3장)
    ("LEL", re.compile(
        r"LEL\s*(?:농도|측정값|수치)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)(?![\d.])"
        r"(?!\s*(?:곳|개|대|명|회|번|시|분|ppm|%|퍼센트|프로|이하|미만|이상|초과|넘|기준|까지))",
        re.IGNORECASE,
    )),
    ("LEL", re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%\s*LEL" + _NOT_CRITERION, re.IGNORECASE)),  # '3%LEL', '2 %LEL'
    ("H2S", re.compile(r"(?:황화수소|H2S)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*ppm" + _NOT_CRITERION, re.IGNORECASE)),
    ("CO", re.compile(r"(?:일산화탄소|CO(?!2))\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*ppm" + _NOT_CRITERION, re.IGNORECASE)),
    ("CO2", re.compile(r"(?:이산화탄소|탄산가스|CO2)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*%" + _NOT_CRITERION)),
)
_NOT_A_NAME = frozenset({"배치", "지정", "필요", "없음", "미정", "없이", "추가", "확인"})
_FIRE_WATCHER_RE = re.compile(r"화재\s*감시자\s*(?:는|은|:)?\s*([가-힣]{2,4})(?:씨|님)?(?=\s|$|[,.])")
_ATTENDANT_RE = re.compile(r"감시인\s*(?:는|은|:)?\s*([가-힣]{2,4})(?:씨|님)?(?=\s|$|[,.])")

# (화기 종류 표시, 찾는 표현, 같은 문장에 있으면 표시를 바꾸는 표현, 바뀐 표시)
_HOT_WORK_KINDS: tuple[tuple[str, tuple[str, ...], str | None, str | None], ...] = (
    ("가스 절단", ("가스절단", "산소절단", "가스 절단", "용단"), None, None),
    ("플라즈마 절단", ("플라즈마",), None, None),
    ("그라인더 연마", ("그라인더",), "절단", "그라인더 절단"),
    ("용접", ("용접",), None, None),
    ("토치 사용", ("토치",), "가열", "토치 가열"),
    ("납땜", ("납땜",), None, None),
)


def _to_hhmm(m: re.Match[str], *, end: bool = False, daytime: bool = True) -> str | None:
    ampm, hour_s, minute_s, half = m.group(1), m.group(2), m.group(3), m.group(4)
    minute = 30 if half else int(minute_s or 0)
    hour = to_24h(ampm, int(hour_s), end=end, daytime=daytime)  # 오전·오후·저녁 등과 주간 작업 전제 (clarify.to_24h)
    if hour > 24 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _clock_to_hhmm(m: re.Match[str]) -> str | None:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 24 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _note_unmarked(unmarked: set[str] | None, key: str, m: re.Match[str]) -> None:
    """오전·오후 없이 읽은 시각이면 key를 적어 둔다 (clarify.is_unmarked). unmarked가 None이면 아무것도 안 한다."""
    if unmarked is not None and is_unmarked(m.group(1), m.group(2)):
        unmarked.add(key)


def _hot_work_kind(text: str) -> str | None:
    kinds: list[str] = []
    for sentence in sentences(text):
        for label, keywords, modifier, modified in _HOT_WORK_KINDS:
            for kw in keywords:
                pos = sentence.find(kw)
                if pos == -1:
                    continue
                if is_negated(sentence, pos + len(kw)):
                    break
                kinds.append(modified if modifier and modifier in sentence else label)
                break
    # '그라인더 절단'이 있으면 같은 공구의 다른 표시는 뺀다
    result = unique(kinds)
    if "그라인더 절단" in result and "그라인더 연마" in result:
        result.remove("그라인더 연마")
    return ", ".join(result) or None


def extract_slots(
    message: str, expecting: Sequence[str] = (), *, today: date | None = None, unmarked: set[str] | None = None
) -> dict:
    """발화 하나에서 읽은 값만 돌려준다 (못 읽은 항목은 키가 없다).

    `expecting`은 직전에 되물은 항목 key다. 발화가 짧은 답("3", "5시", "없음")이면 그 항목의 답으로 읽는다.
    `today`는 '내일' 같은 날짜의 기준일이다 (없으면 한국 시간 오늘).
    `unmarked`를 주면 오전·오후 없이 읽은 시각의 key('start_time'·'end_time')를 거기에 담는다 (돌려주는 값은 그대로다).
    """
    found: dict = {}
    location, material_locations = extract_locations(message)
    if location:
        found["location"] = location
    if material_locations:
        found["material_locations"] = material_locations
    if kind := _hot_work_kind(message):
        found["hot_work_kind"] = kind

    materials, _ = find_terms(message, MATERIALS)
    if materials:
        found["nearby_materials"] = materials
    if _MATERIALS_NONE_RE.search(message):
        found["nearby_materials_none"] = True

    equipment, excluded = find_terms(message, EQUIPMENT)
    if equipment:
        found["equipment"] = equipment
    if excluded:
        found["excluded_equipment"] = excluded
    if _EQUIPMENT_NONE_RE.search(message):
        found["equipment_none"] = True

    if m := _HEADCOUNT_RE.search(message):
        found["headcount"] = int(m.group(1))
    elif m := _HEADCOUNT_KO_RE.search(message):
        found["headcount"] = _KO_NUMBERS[m.group(1)]
    if work_date := extract_work_date(message, today):
        found["workDate"] = work_date
    start_marker = None
    if (m := _START_RE.search(message) or _START_TILDE_RE.search(message)) and (hhmm := _to_hhmm(m)):
        found["start_time"] = hhmm
        start_marker = m.group(1)
        _note_unmarked(unmarked, "start_time", m)
    elif (m := _START_CLOCK_RE.search(message) or _START_CLOCK_TILDE_RE.search(message)) and (hhmm := _clock_to_hhmm(m)):
        found["start_time"] = hhmm
    end_match = _END_RE.search(message) or _END_TILDE_RE.search(message)
    if end_match and (hhmm := _to_hhmm(end_match, end=True)):
        found["end_time"] = hhmm
        # 오전으로 시작하고 종료에 표시가 없으면 시작 뒤의 같은 오전으로 읽는다 ('오전 3시~5시', 주간 전제는 쓰지 않는다)
        plain = _to_hhmm(end_match, daytime=False)
        if start_marker and not end_match.group(1) and plain and plain > found["start_time"]:
            found["end_time"] = plain
        _note_unmarked(unmarked, "end_time", end_match)
    elif (end_match := _END_CLOCK_RE.search(message) or _END_CLOCK_TILDE_RE.search(message)) and (
        hhmm := _clock_to_hhmm(end_match)
    ):
        found["end_time"] = hhmm

    # 한 발화에 같은 항목이 다른 값으로 두 번 나오면('산소 19%, 환기 후 산소 20.9%') 어느 값인지 정하지 않는다.
    # 값이 비면 규칙이 MISSING으로 승인을 막고, 다음 발화에서 한 값만 말하면 그 값을 쓴다.
    values: dict[str, set[float]] = {}
    for item, pattern in _MEASUREMENT_RES:
        for m in pattern.finditer(message):
            values.setdefault(item, set()).add(float(m.group(1)))
    measurements = {item: next(iter(vs)) for item, vs in values.items() if len(vs) == 1}
    if measurements:
        found["measurements"] = measurements
    if (m := _FIRE_WATCHER_RE.search(message)) and m.group(1) not in _NOT_A_NAME:
        found["fire_watcher"] = m.group(1)
    if (m := _ATTENDANT_RE.search(message)) and m.group(1) not in _NOT_A_NAME:
        found["attendant"] = m.group(1)
    for key, value in _read_short_answer(message, expecting, unmarked).items():
        found.setdefault(key, value)
    return found


def _read_short_answer(message: str, expecting: Sequence[str], unmarked: set[str] | None = None) -> dict:
    """직전에 물은 항목의 짧은 답을 읽는다. 어느 항목의 답인지 정할 수 없으면 읽지 않는다 (추측하지 않음)."""
    if not expecting:
        return {}
    out: dict = {}
    # 숫자만 온 답: 인원과 LEL을 함께 물었으면 단위('명', '%')가 있어야 어느 쪽인지 안다
    if "headcount" in expecting and (m := _BARE_NUMBER_RE.match(message)) and 1 <= int(m.group(1)) <= 500:
        if m.group(2) or "lel" not in expecting:
            out["headcount"] = int(m.group(1))
    if "lel" in expecting and (m := _BARE_LEL_RE.match(message)):
        if m.group(1) or m.group(3) or "headcount" not in expecting:
            out["measurements"] = {"LEL": float(m.group(2))}
    # 시각만 온 답: 시작·종료를 함께 물었으면 어느 쪽인지 모른다 ('10시부터 12시까지'는 extract_slots가 읽는다)
    asked_times = [k for k in ("start_time", "end_time") if k in expecting]
    if len(asked_times) == 1:
        if (m := _BARE_TIME_RE.match(message)) and (hhmm := _to_hhmm(m, end=asked_times[0] == "end_time")):
            out[asked_times[0]] = hhmm
            _note_unmarked(unmarked, asked_times[0], m)
        elif (m := _BARE_CLOCK_RE.match(message)) and (hhmm := _clock_to_hhmm(m)):
            out[asked_times[0]] = hhmm
    # '없음'만 온 답: 물질·장비·LEL 중 하나만 물었을 때만 읽는다. LEL '없음'은 0으로 바꾸지 않는다 (숫자만 읽음)
    asked_none = [k for k in ("nearby_materials", "equipment", "lel") if k in expecting]
    if _BARE_NOT_USED_RE.match(message) and "equipment" in expecting:
        out["equipment_none"] = True
    elif _BARE_NONE_RE.match(message) and len(asked_none) == 1 and asked_none[0] != "lel":
        out["nearby_materials_none" if asked_none[0] == "nearby_materials" else "equipment_none"] = True
    return out


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _afternoon_end(start: str | None, end: str | None) -> str | None:
    """종료가 시작보다 이르고 12시 전이면 오후로 본다 ('오후 1시~3시', '11시부터 1시까지' → 15:00, 13:00).

    같은 날 안에서 종료가 시작보다 이를 수는 없다 (Spring은 이때 종료 시각을 비우고 허가서를 만들지 않는다).
    12시간을 더해도 시작보다 이르면(자정을 넘는 야간작업 '22시~2시') 그대로 둔다.
    """
    if not start or not end:
        return end
    s, e = _minutes(start), _minutes(end)
    if e <= s and e < 12 * 60 and e + 12 * 60 > s:
        return f"{(e + 12 * 60) // 60:02d}:{e % 60:02d}"
    return end


def _held_times(slots: WorkSlots | None, today: date | None = None) -> dict[str, str]:
    """오전·오후 없이 읽어 저장하지 않은 시각 {key: 'HH:MM'} (값이 아직 없는 항목만).

    저장하지 않은 시각은 slots.utterances에 말한 그대로 남아 있으므로 발화를 다시 읽어 구한다 (slots에 새 필드가 없다).
    '3' 같은 짧은 답은 시작·종료 중 값이 없는 항목이 하나일 때만 그 항목의 답으로 본다 (되물을 때와 같은 규칙).
    """
    held: dict[str, str] = {}
    if slots is None:
        return held
    empty = [k for k in TIME_KEYS if getattr(slots, k) is None]
    for utterance in slots.utterances:
        unmarked: set[str] = set()
        found = extract_slots(utterance, empty if len(empty) == 1 else (), today=today, unmarked=unmarked)
        held.update({k: found[k] for k in unmarked if k in empty})
    return held


def merge_slots(previous: WorkSlots | None, message: str, *, today: date | None = None) -> WorkSlots:
    held = _held_times(previous, today)
    base = (previous or WorkSlots()).model_dump()
    base["utterances"] = (base["utterances"] + [message.strip()])[-MAX_UTTERANCES:]
    unmarked: set[str] = set()
    found = extract_slots(message, base.get("asked_keys", []), today=today, unmarked=unmarked)
    found.update(resolve_answer(message, held) or {})  # '오후'만 온 답
    base["asked_keys"] = []  # 이번 발화로 답을 받았다. 다시 되묻게 되면 에이전트가 새로 채운다
    for key, value in found.items():
        if isinstance(value, list):
            base[key] = unique([*base.get(key, []), *value])
        elif isinstance(value, dict):
            base[key] = {**base.get(key, {}), **value}
        else:
            base[key] = value
    # 나중에 '안 씀'이라고 한 장비는 사용 목록에서 뺀다 (반대도 같다)
    latest_excluded = found.get("excluded_equipment", [])
    latest_used = found.get("equipment", [])
    base["equipment"] = [e for e in base["equipment"] if e not in latest_excluded]
    base["excluded_equipment"] = [e for e in base["excluded_equipment"] if e not in latest_used]
    if base["nearby_materials"]:
        base["nearby_materials_none"] = False
    # 오전·오후 없는 시각은 저장하지 않고 되묻는다 (_held_times가 발화에서 다시 읽는다). 시작이 확정이면 종료는 아래 규칙으로 정한다.
    if "start_time" in unmarked:
        end = base.get("end_time")
        # 오후로 읽으면 종료보다 늦어지는 경우('6시부터 오후 3시까지')는 같은 날 안에서 오전뿐이라 되묻지 않는다
        if not (end and "end_time" not in unmarked and _minutes(base["start_time"]) + 12 * 60 >= _minutes(end)):
            base["start_time"] = None
    if "end_time" in unmarked and base.get("start_time") is None:
        base["end_time"] = None
    if base.get("end_time") is None and held.get("end_time") and base.get("start_time") is not None:
        base["end_time"] = held["end_time"]  # 종료를 먼저 말했고 이제 시작이 확정됐다
    base["end_time"] = _afternoon_end(base.get("start_time"), base.get("end_time"))
    return WorkSlots.model_validate(base)


def full_text(slots: WorkSlots) -> str:
    return "\n".join(slots.utterances)


_QUESTIONS: dict[str, tuple[str, str]] = {
    "task": ("작업 내용", "어떤 작업인지 대상 설비와 작업 내용을 알려주세요."),
    "location": ("작업 장소", "작업 장소(설비 번호나 구역)는 어디인가요?"),
    "hot_work_kind": ("화기 종류", "화기 종류는 무엇인가요? (용접 · 절단 · 그라인더 · 토치 등)"),
    "nearby_materials": ("주변 취급 물질", "작업 장소 주변에 취급하거나 보관 중인 물질이 있나요? 없으면 '물질 없음'이라고 알려주세요."),
    "headcount": ("작업 인원", "작업 인원은 몇 명인가요?"),
    "start_time": ("시작 시간", "작업은 몇 시에 시작하나요?"),
    "end_time": ("종료 시간", "작업은 몇 시에 끝나나요?"),
    "equipment": ("투입 장비", "투입하는 장비·공구를 알려주세요. 쓰지 않는 장비는 '크레인 안 씀'처럼 알려주셔도 됩니다."),
    "lel": (
        "가연성 가스(LEL) 측정값",
        "가연성 가스(LEL)를 측정했다면 값을 알려주세요 (예: LEL 0%). 아직 측정 전이면 '측정 전'이라고 하셔도 됩니다. "
        "측정값이 없어도 초안은 만들지만, 승인 요청은 측정값을 넣은 뒤에 할 수 있습니다.",
    ),
}


def missing_fields(slots: WorkSlots, work_types: Sequence[str]) -> list[MissingField]:
    """빠진 항목. 필수 항목이 먼저, 한 번만 묻는 항목(OPTIONAL_KEYS)이 뒤에 온다.

    한 번만 묻는 항목만 남으면(필수는 다 채움) 빈 목록이다 (ASK_LEL_ALONE=False일 때). 그래서 이 목록이 비면 DONE이다.
    """
    keys = _required_keys(slots, work_types)
    if (
        LEL_WORK_TYPES.intersection(work_types)
        and "LEL" not in slots.measurements
        and "lel" not in slots.asked_optional_keys
        and (keys or ASK_LEL_ALONE)
    ):
        keys.append("lel")
    fields = [MissingField(key=k, label=_QUESTIONS[k][0], question=_QUESTIONS[k][1]) for k in keys]
    return ampm_fields(fields, _held_times(slots))


def with_asked(slots: WorkSlots, missing: Sequence[MissingField]) -> WorkSlots:
    """되묻기 응답에 실을 slots: 이번에 물은 항목(asked_keys)과 한 번 물은 선택 항목(asked_optional_keys)을 남긴다."""
    keys = [f.key for f in missing]
    return slots.model_copy(update={
        "asked_keys": keys,
        "asked_optional_keys": unique([*slots.asked_optional_keys, *(k for k in keys if k in OPTIONAL_KEYS)]),
    })


def _required_keys(slots: WorkSlots, work_types: Sequence[str]) -> list[str]:
    keys: list[str] = []
    if not work_types:
        keys.append("task")
    if not slots.location:
        keys.append("location")
    if "화기" in work_types and not slots.hot_work_kind:
        keys.append("hot_work_kind")
    if ("화기" in work_types or "밀폐공간" in work_types) and not (slots.nearby_materials or slots.nearby_materials_none):
        keys.append("nearby_materials")
    if slots.headcount is None:
        keys.append("headcount")
    if slots.start_time is None:
        keys.append("start_time")
    if slots.end_time is None:
        keys.append("end_time")
    if not (slots.equipment or slots.excluded_equipment or slots.equipment_none):
        keys.append("equipment")
    return keys


def ask_message(missing: Sequence[MissingField]) -> str:
    lines = ["초안을 만들려면 몇 가지가 더 필요합니다."]
    lines += [f"{i}. {f.question}" for i, f in enumerate(missing, 1)]
    return "\n".join(lines)
