"""오전·오후가 없는 시각을 읽는다 (되묻기 개선).

"3시부터 5시까지"처럼 오전·오후 없이 말하면 새벽 3시인지 오후 3시인지 알 수 없다. 예전에는 03:00으로 읽어 그대로
초안을 만들었다. 이 프로젝트는 주간 작업이 기본이라는 전제(11/05 사용자)로 읽는다. 되묻지 않는다.

  - 표시 없는 1~5시 → 오후 (13~17시)            DAYTIME_PM_HOURS
  - 표시 없는 7~11시 → 오전                     (그대로)
  - 표시 없는 12시 → 낮 12시                    (그대로)
  - 표시 없는 6시 → 시작이면 오전(06:00), 마감이면 오후(18:00)   END_PM_HOURS
  - 오전·오후·새벽·아침·낮·점심·저녁·밤이 붙으면 그 말대로 읽는다 (to_24h)
  - 오전·새벽·아침으로 시작하고 종료에 표시가 없으면, 시작 뒤로 오는 같은 오전으로 읽는다 ('오전 3시~5시' → 03:00~05:00).
    '오전 9시부터 5시까지'는 5시가 9시보다 앞서므로 오후(17:00)다

되묻기 장치(AMPM_ASK_HOURS)는 남겨 두었다. 야간 작업이 있는 현장이면 이 상수로 오전·오후를 되묻게 할 수 있다.
되묻는 시각은 slots에 저장하지 않고(값 없음 = 빠진 항목) 오전인지 오후인지 묻는다. 저장하지 않은 시각은
slots.utterances(받은 발화)에 남아 있으므로 다음 턴에 다시 읽어 구한다. 그래서 slots에 새 필드가 없다
(Spring·웹이 slots를 그대로 돌려보내기만 하면 된다).

확정으로 보는 경우 (되묻지 않음)
  - 시각 앞에 오전·오후·새벽·아침·낮·점심·저녁·밤이 붙은 경우 ('오후 3시', '저녁 7시', '새벽 3시')
  - 13시 이상, 0시, 12시 ('15시', '12시'는 낮 12시), HH:MM 표기 ('15:00')
  - 종료 시각이 표시 없이 왔어도 시작이 확정이면, 기존 규칙(slots._afternoon_end)이 시작 뒤의 시각으로 정한다

이 모듈은 slots.py를 임포트하지 않는다 (slots.py가 이 모듈을 쓴다). 판단에 필요한 값만 받는다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.schemas.agent_work import MissingField

# 시각 앞에 붙어 오전·오후를 알려 주는 말. slots.py의 정규식이 이 문자열을 쓴다 (정규식 첫 번째 그룹).
MARKERS: tuple[str, ...] = ("오전", "오후", "새벽", "아침", "낮", "점심", "저녁", "밤")
MARKER_PATTERN = "|".join(MARKERS)
_PM_WORDS = frozenset({"오후", "낮", "점심", "저녁", "밤"})  # '오후'만 온 답 대신 이 말만 와도 오후의 답이다

# 오전·오후 표시가 없으면 되묻는 시(時). 기본은 되묻지 않는다 (주간 작업 기본, 아래 두 상수로 읽는다).
# 야간 작업이 있는 현장이면 frozenset(range(1, 12))로 넓히고 DAYTIME_PM_HOURS·END_PM_HOURS를 비운다.
AMPM_ASK_HOURS: frozenset[int] = frozenset()
# 표시가 없으면 오후로 읽는 시(時). 주간 작업이 기본이라는 전제다.
DAYTIME_PM_HOURS: frozenset[int] = frozenset(range(1, 6))
# 표시가 없으면 마감(종료) 시각만 오후로 읽는 시(時). 6시 시작은 06:00, 6시 마감은 18:00.
END_PM_HOURS: frozenset[int] = frozenset({6})
# 세 상수를 모두 비우면 예전 동작과 같다 ('3시'는 03:00).

START = "start_time"
END = "end_time"
TIME_KEYS = (START, END)

_AMPM_ONLY_RE = re.compile(r"^\s*(" + MARKER_PATTERN + r")\s*(?:이요|요|입니다|이에요|예요)?\s*[.!]?\s*$")


def to_24h(marker: str | None, hour: int, *, end: bool = False, daytime: bool = True) -> int:
    """시각 앞의 말(marker)과 시(hour)로 24시간제 시를 구한다. end는 종료(마감) 시각이면 True.

    marker가 없으면 주간 작업 전제(DAYTIME_PM_HOURS, END_PM_HOURS)로 읽는다. daytime=False면 그 전제 없이 쓴 그대로 읽는다.
    13시 이상은 그대로다.
    '12시'는 낮 12시이고 '밤 12시'·'저녁 12시'는 자정이다 (시작이면 0시, 종료면 24시).
    """
    if not 1 <= hour <= 12:
        return hour  # 0시·13시 이상은 말해 준 그대로
    if not marker:
        return hour + 12 if daytime and (hour in DAYTIME_PM_HOURS or (end and hour in END_PM_HOURS)) else hour
    if marker == "오후":
        return hour + 12 if hour < 12 else hour
    if marker == "새벽":
        return 0 if hour == 12 else hour  # 새벽 12시는 자정
    if marker in ("낮", "점심"):
        return hour + 12 if 1 <= hour <= 5 else hour  # 낮 2시 → 14시, 낮 12시 → 12시
    if marker == "저녁":
        return hour + 12 if hour < 12 else (24 if end else 0)
    if marker == "밤":
        if 6 <= hour <= 11:
            return hour + 12
        return (24 if end else 0) if hour == 12 else hour  # 밤 2시 → 02시
    return hour  # 오전·아침: 그대로


def is_unmarked(ampm: str | None, hour_text: str) -> bool:
    """'3시'처럼 말(오전·오후·저녁 등) 없이 읽은 시각이 되묻는 시(AMPM_ASK_HOURS)이면 True. 정규식 첫 그룹과 시를 그대로 받는다."""
    return not ampm and int(hour_text) in AMPM_ASK_HOURS


def resolve_answer(message: str, held: Mapping[str, str]) -> dict[str, str] | None:
    """'오후'·'오전'만 온 답을 보류 중인 시각에 적용한다. held는 {key: 'HH:MM'} (저장하지 않고 되묻는 중인 시각).

    돌려주는 값은 {key: 'HH:MM'}이다 (오전이면 값이 그대로다). 답이 아니거나 보류 중인 시각이 없으면 None.
    """
    if not held or not (m := _AMPM_ONLY_RE.match(message)):
        return None
    afternoon = m.group(1) in _PM_WORDS
    out: dict[str, str] = {}
    for key in TIME_KEYS:
        if not (hhmm := held.get(key)):
            continue
        hour, minute = hhmm.split(":")
        value = int(hour)
        if afternoon and value < 12:
            value += 12
        out[key] = f"{value:02d}:{minute}"
    return out or None


def _hour_label(hhmm: str) -> str:
    hour, minute = int(hhmm[:2]), int(hhmm[3:])
    hour = hour - 12 if hour > 12 else hour  # '11시부터 1시까지'의 보류 종료 13:00은 사용자가 말한 대로 '1시'로 보인다
    return f"{hour}시" if minute == 0 else f"{hour}시 {minute}분"


def _topic(label: str) -> str:
    return f"{label}은" if label.endswith("분") else f"{label}는"


def ampm_fields(fields: Sequence[MissingField], held: Mapping[str, str]) -> list[MissingField]:
    """빠진 항목 중 보류 중인 시각의 물음을 '오전인가요, 오후인가요?'로 바꾼다.

    시작·종료가 모두 보류면 하나로 합쳐 시작 항목(start_time)에서 함께 묻는다 (종료 항목은 뺀다).
    key는 그대로라서 웹과 asked_keys의 짧은 답 읽기는 달라지지 않는다.
    """
    pending = [k for k in TIME_KEYS if held.get(k)]
    both = len(pending) == len(TIME_KEYS)
    out: list[MissingField] = []
    for field in fields:
        if field.key not in pending:
            out.append(field)
        elif both and field.key == END:
            continue
        elif both:
            span = f"{_hour_label(held[START])}~{_hour_label(held[END])}"
            out.append(MissingField(
                key=field.key,
                label="시작·종료 시간",
                question=f"작업 시간 {_topic(span)} 오전인가요, 오후인가요? (예: 오후 3시부터 5시까지)",
            ))
        else:
            out.append(MissingField(
                key=field.key,
                label=field.label,
                question=f"{field.label} {_topic(_hour_label(held[field.key]))} 오전인가요, 오후인가요?",
            ))
    return out
