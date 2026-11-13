"""시작·종료 시각 읽기 (11/13, 시각 읽기 개선). 오독을 줄이고, 오독 위험이면 읽지 않고 되묻는다.

문제: slots.py는 숫자 뒤 조사로만 역할을 정했다 (시작 = N시+에/부터, 종료 = N시+까지/종료/마감/끝).
그래서 "11시에 마쳐"가 시작 11:00으로, "15시까지가 아니라 16시까지로"가 종료 15:00으로 읽혔고,
"23시부터 2시까지"는 종료 14:00, "9시부터 9시까지"는 21:00으로 조용히 보정됐다. 라벨형("종료는 11시")·어미 변형
("9시 시작", "11시쯤 끝나")·한글 수("아홉 시")는 못 읽어 같은 질문이 반복됐다.

읽는 순서 (LLM 없음, 정규식·단서어만)
  1. 시각 후보를 모두 찾는다: "N시", "N시 M분", "N시 반", "HH:MM", 한글 수("열한 시"), 앞의 오전·오후 등.
     "2시간"의 '시간'은 시각이 아니다.
  2. 역할은 같은 절의 단서로 정한다 (우선순위 순).
       라벨   "시작(시간)은 9시", "종료는 11시", "끝나는 건 11시"
       뒤 단서  시작: 부터·에서·시작·착수 / 종료: 까지·전까지·종료·마감·끝·마쳐·마무리·완료 / "N시에"는 종료 단서가 없으면 시작
       범위   "A~B", "A-B", "A부터 B", "A에서 B 사이" → 앞이 시작, 뒤가 종료
       교체   "A가 아니라 B", "A 말고 B" → A는 버리고 B만 읽는다 (B에 단서가 없으면 A의 역할을 받는다)
  3. 단서가 없는 시각은, 직전에 시작·종료 중 하나만 물었고 단서 없는 시각이 하나뿐일 때만 그 항목의 답으로 읽는다.
  4. 읽지 않고 되묻는 경우 (추측하지 않는다)
       - 시각이 있는데 역할을 정할 수 없거나, 같은 역할에 서로 다른 값이 둘이다 → 문구 앞에 "정확히 읽지 못했습니다"
       - 같은 날 안에서 종료가 시작보다 이르거나 같다 (표시 없는 12시간 보정은 하지 않는다) → 시작·종료를 둘 다 비운다
       - 시작이 18시 이후이고 종료가 시작보다 이르다 (자정을 넘는 야간) → 미지원 안내, 시작·종료를 둘 다 비운다
  오전·오후·새벽·아침·낮·점심·저녁·밤이 붙었거나 24시간제·HH:MM이면 확정으로 읽는다 (clarify.to_24h의 기존 규칙).
  주간 작업 기본(표시 없는 1~5시 = 오후, 6시 시작 06:00·마감 18:00)과 '오전 3시~5시'의 같은 오전 읽기는 그대로다.

상태를 slots에 저장하지 않는다 (새 필드 없음). "왜 비었는지"는 slots.utterances를 처음부터 다시 읽어 구한다 (explain),
되묻는 문구는 retext가 고친다. missingFields의 key·개수·slots 스키마는 그대로이고 question·ask.message 문구만 바뀐다.

끄기: TIME_READER_V2 = False → slots.py·followup.py가 변경 전 코드를 그대로 쓴다.
이 모듈은 slots.py를 임포트하지 않는다 (slots.py가 쓴다). 18시·단서어는 임시값이다 (정답 세트 전).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.schemas.agent_work import MissingField

from .clarify import MARKER_PATTERN, START, END, TIME_KEYS, is_unmarked, to_24h

TIME_READER_V2 = True
# 시작이 이 시(時) 이후이고 종료가 시작보다 이르면 자정을 넘는 야간 작업으로 본다 (임시값, 주간 작업 기본 전제).
NIGHT_START_HOUR = 18

Problem = Literal["unread", "mismatch", "overnight"]

Q_BOTH = "작업 시간을 알려주세요. 예: '9시부터 11시까지'"
Q_START = "작업은 몇 시에 시작하나요? (예: '9시부터 11시까지')"
Q_END = "작업은 몇 시에 끝나나요? (예: '9시부터 11시까지'처럼 한 번에 말씀하셔도 됩니다)"
PREFIX_UNREAD = "말씀하신 시간을 정확히 읽지 못했습니다. "
MSG_MISMATCH = (
    "시작과 종료 시간이 맞지 않습니다. '오전 9시부터 오후 2시까지'처럼 오전·오후를 붙이거나, "
    "'9시부터 14시까지'처럼 24시간제로 말씀해 주세요."
)
MSG_OVERNIGHT = "자정을 넘는 작업은 아직 지원하지 않습니다. 당일 안에 끝나는 시각으로 말씀해 주세요. 예: '19시부터 23시까지'"
REPLY_TIME_ROLE = "시작과 종료 중 어느 시간인지 말씀해 주세요. 예: '종료를 16시로', '9시부터 11시까지'"  # DONE 뒤 '16시로 변경'

# ── 시각 후보 ──

_KO_HOURS = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "열한": 11, "열두": 12,
}
# 한글 수 뒤에서 '시'가 시각이려면 뒤에 오는 말 ('한 시설', '두 시스템'을 거른다)
_KO_TAIL = r"(?:\s|$|[,.!?~∼～\-–—]|에|부터|까지|쯤|경|전|이전|이후|정도|종료|시작|마감|끝|마쳐|마치|마무리|완료|착수|예요|이에요|입니다|이요|요|야|반)"
_RANGE_MARK = r"[~∼～〜]"
_TOKEN_RE = re.compile(
    r"(?P<marker>" + MARKER_PATTERN + r")?\s*(?:"
    r"(?P<ch>(?<![\d.:])\d{1,2}):(?P<cm>\d{2})(?![\d:])"
    r"|(?<![\d.:])(?P<h>\d{1,2})\s*시(?!간)(?:\s*(?:(?P<m>\d{1,2})\s*분|(?P<half>반)))?"
    r"|(?<![가-힣])(?P<kh>열한|열두|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*시(?!간)(?=" + _KO_TAIL + r")(?:\s*(?P<khalf>반))?"
    r"|(?<![\d.:])(?P<bare>\d{1,2})(?=\s*" + _RANGE_MARK + r"\s*(?:" + MARKER_PATTERN + r")?\s*\d{1,2}\s*시(?!간))"  # '10~12시'의 10
    r")"
)

_DELIMS = ",.;!?\n·…、"
_WINDOW_CAP = 24

_START_WORDS = frozenset({"시작", "착수", "개시"})
# 라벨: 시각 바로 앞의 '시작(시간)은', '종료는', '끝나는 건' 같은 말
_LABEL_RE = re.compile(
    r"(?P<word>시작|착수|개시|종료|끝나는|끝나|끝|마감|완료|마치는|마무리)"
    r"(?:\s*(?:시간|시각))?\s*(?:은|는|이|가|을|를|도|건|거는|거|것은|:|=)?\s*(?:약|대략|바로|정확히)?\s*$"
)
_FILL = r"(?:경|쯤|정도|께)?"
_END_VERBS = r"(?:종료|마감|끝|마쳐|마치|마무리|완료)"
_START_VERBS = r"(?:시작|착수|개시)"
_P_UNTIL = re.compile(r"^" + _FILL + r"\s*(?:에\s*)?(?:이전|전)?\s*까지")
_P_END = re.compile(r"^" + _FILL + r"\s*(?:에\s*)?(?:(?:이전|전)\s*에?\s*)?" + _END_VERBS)
_P_START = re.compile(r"^" + _FILL + r"\s*(?:에\s*)?" + _START_VERBS)
_P_FROM = re.compile(r"^" + _FILL + r"\s*(?:부터|에서)")
_P_AT = re.compile(r"^" + _FILL + r"\s*에(?!서)")
_END_VERB_RE = re.compile(_END_VERBS)
_START_VERB_RE = re.compile(_START_VERBS)
_RANGE_ONLY_RE = re.compile(r"^\s*[~∼～〜\-–—]\s*$")
_REPLACE_RE = re.compile(r"아니라|아니고|말고(?!도)|대신")

Role = Literal["start", "end"]


@dataclass
class _Token:
    start: int
    end: int
    marker: str | None
    hour: int
    minute: int
    form: Literal["hour", "clock", "korean", "bare"]
    role: Role | None = None
    cue: str = ""  # 역할을 정한 단서의 종류 (label|until|end|start|from|at|range|inherit|expect)
    replaced: bool = False

    @property
    def unmarked_hour(self) -> bool:
        return self.marker is None and self.form != "clock"


@dataclass(frozen=True)
class TimeReading:
    """발화 하나에서 읽은 시작·종료.

    start·end  'HH:MM' (못 읽으면 None)
    blank      True면 시작·종료를 둘 다 비워야 한다 (이 발화 안의 모순·야간). start·end는 None이다.
    problem    None | "unread"(시각이 있는데 읽지 못함·같은 역할 값 충돌) | "mismatch" | "overnight"
    unmarked   오전·오후 없이 읽은 시각의 key (clarify.AMPM_ASK_HOURS에 든 시, 기본은 비어 있다)
    end_unmarked  종료를 오전·오후 말 없이 '시'로 읽었는가 (야간 안내와 오전·오후 안내를 가를 때 쓴다)
    """

    start: str | None = None
    end: str | None = None
    blank: bool = False
    problem: Problem | None = None
    unmarked: frozenset[str] = frozenset()
    end_unmarked: bool = False


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _find_tokens(message: str) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(message):
        marker = m.group("marker")
        if m.group("ch") is not None:
            hour, minute, form = int(m.group("ch")), int(m.group("cm")), "clock"
        elif m.group("h") is not None:
            minute = 30 if m.group("half") else int(m.group("m") or 0)
            hour, form = int(m.group("h")), "hour"
        elif m.group("kh") is not None:
            hour, minute, form = _KO_HOURS[m.group("kh")], 30 if m.group("khalf") else 0, "korean"
        else:
            hour, minute, form = int(m.group("bare")), 0, "bare"
        if hour > 24 or minute > 59:
            continue  # 시각이 아니다 ('25시', '9:75')
        tokens.append(_Token(m.start(), m.end(), marker, hour, minute, "hour" if form in ("korean", "bare") else form))
    return tokens


def _clause(text: str) -> str:
    """text의 마지막 구분자(쉼표·마침표 등) 뒤쪽. 구분자가 없으면 text 전체."""
    cut = max(text.rfind(c) for c in _DELIMS)
    return text[cut + 1:]


def _window(message: str, tok: _Token, limit: int) -> str:
    """tok 뒤에서 다음 시각·구분자 전까지의 말 (최대 _WINDOW_CAP자)."""
    seg = message[tok.end:limit]
    for c in _DELIMS:
        i = seg.find(c)
        if i != -1:
            seg = seg[:i]
    return seg[:_WINDOW_CAP]


def _post_cue(window: str) -> tuple[Role | None, str, int]:
    """시각 뒤 단서 → (역할, 단서 종류, 단서가 끝난 위치)."""
    for pattern, role, cue in ((_P_UNTIL, "end", "until"), (_P_END, "end", "end"), (_P_START, "start", "start"), (_P_FROM, "start", "from")):
        if m := pattern.match(window):
            return role, cue, m.end()  # type: ignore[return-value]
    if m := _P_AT.match(window):
        rest = window[m.end():]
        e, s = _END_VERB_RE.search(rest), _START_VERB_RE.search(rest)
        if e and (not s or e.start() < s.start()):
            return "end", "end", m.end() + e.end()  # 'N시에 용접 마쳐'
        return "start", "at", m.end()  # 'N시에 ...' (종료 단서가 없으면 시작)
    return None, "", 0


def _assign_roles(message: str, tokens: list[_Token], expecting: Sequence[str]) -> None:
    """tokens의 role·cue·replaced를 정한다 (같은 역할에 다른 값이 둘인지는 read_times가 본다)."""
    n = len(tokens)
    # 교체 표현: 'A가 아니라 B'의 A는 버린다
    for i in range(n - 1):
        if _REPLACE_RE.search(message[tokens[i].end:tokens[i + 1].start]):
            tokens[i].replaced = True
    consumed = 0  # 앞 시각의 뒤 단서가 쓴 말은 다음 시각의 라벨로 쓰지 않는다
    for i, tok in enumerate(tokens):
        lo = max(consumed, tokens[i - 1].end if i else 0)
        before = _clause(message[lo:tok.start]).rstrip()
        label = _LABEL_RE.search(before)
        limit = tokens[i + 1].start if i + 1 < n else len(message)
        window = _window(message, tok, limit)
        if label:
            tok.role = "start" if label.group("word") in _START_WORDS else "end"
            tok.cue = "label"
            continue
        # 범위: 'A~B', 'A-B' (두 시각이 바로 이어질 때)
        if i + 1 < n and _RANGE_ONLY_RE.match(message[tok.end:limit]):
            tok.role, tok.cue = "start", "range"
            continue
        role, cue, end = _post_cue(window)
        if role:
            tok.role, tok.cue = role, cue
            consumed = tok.end + end
    # 'A부터 B', 'A에서 B 사이': 시작 단서 바로 뒤에 이어진 단서 없는 시각은 종료
    for i in range(n - 1):
        a, b = tokens[i], tokens[i + 1]
        if a.role == "start" and a.cue in ("from", "range") and b.role is None:
            gap = message[a.end:b.start]
            if a.cue == "range" or _P_FROM.match(gap) and not gap[_P_FROM.match(gap).end():].strip():
                b.role, b.cue = "end", "range"
    # 교체: 'A 말고 B'에서 B에 단서가 없으면 A의 역할을 받는다
    for i in range(n - 1):
        if tokens[i].replaced and tokens[i + 1].role is None and tokens[i].role is not None:
            tokens[i + 1].role, tokens[i + 1].cue = tokens[i].role, "inherit"
    # 단서 없는 시각: 시작·종료 중 하나만 물었고 단서 없는 시각이 하나뿐이면 그 항목의 답
    asked = [k for k in TIME_KEYS if k in expecting]
    loose = [t for t in tokens if t.role is None and not t.replaced]
    if len(asked) == 1 and len(loose) == 1:
        loose[0].role, loose[0].cue = ("start" if asked[0] == START else "end"), "expect"


def _to_hhmm(tok: _Token, *, end: bool, daytime: bool = True) -> str:
    if tok.form == "clock" and tok.marker is None:
        hour = tok.hour  # 'HH:MM'은 쓴 그대로 (24시간제)
    else:
        hour = to_24h(tok.marker, tok.hour, end=end, daytime=daytime)
    return f"{hour:02d}:{tok.minute:02d}"


def conflict_kind(start: str | None, end: str | None, end_unmarked: bool = False) -> Literal["mismatch", "overnight"] | None:
    """같은 날 안에서 시작·종료가 맞지 않으면 'mismatch', 자정을 넘는 야간이면 'overnight', 정상이면 None.

    종료 <= 시작이면 어긋난 것이다 (표시 없는 12시간 보정은 하지 않는다). 시작이 NIGHT_START_HOUR 이후이고 종료가 시작보다 이르면 야간이다.
    다만 종료를 오전·오후 없이 말했고 12시간을 더하면 시작 뒤 같은 날이 되는 경우('저녁 7시부터 11시까지')는 야간이 아니라
    오전·오후를 안 붙인 것으로 본다 (mismatch).
    """
    if not start or not end:
        return None
    s, e = _minutes(start), _minutes(end)
    if e > s:
        return None
    if s >= NIGHT_START_HOUR * 60 and e < s and not (end_unmarked and s < e + 12 * 60 < 24 * 60):
        return "overnight"
    return "mismatch"


def read_times(message: str, expecting: Sequence[str] = ()) -> TimeReading:
    """발화 하나에서 시작·종료를 읽는다. expecting은 직전에 되물은 항목 key (짧은 답을 읽을 때 쓴다)."""
    tokens = _find_tokens(message)
    if not tokens:
        return TimeReading()
    _assign_roles(message, tokens, expecting)
    by_role: dict[str, list[_Token]] = {"start": [], "end": []}
    for tok in tokens:
        if tok.role and not tok.replaced:
            by_role[tok.role].append(tok)

    chosen: dict[str, _Token] = {}
    role_conflict = False
    for role, toks in by_role.items():
        if not toks:
            continue
        values = {(t.hour, t.minute, t.marker) for t in toks}
        if len(values) == 1 or len({_to_hhmm(t, end=role == "end") for t in toks}) == 1:
            chosen[role] = toks[0]
        else:
            role_conflict = True  # 같은 역할에 다른 값이 둘: 어느 쪽인지 모른다

    start_tok, end_tok = chosen.get("start"), chosen.get("end")
    start = _to_hhmm(start_tok, end=False) if start_tok else None
    end = None
    if end_tok:
        end = _to_hhmm(end_tok, end=True)
        # 오전으로 시작하고 종료에 표시가 없으면 시작 뒤의 같은 오전으로 읽는다 ('오전 3시~5시', 주간 전제는 쓰지 않는다)
        if start_tok and start_tok.marker and not end_tok.marker and end_tok.form == "hour":
            plain = _to_hhmm(end_tok, end=True, daytime=False)
            if start and plain > start:
                end = plain
    end_unmarked = bool(end_tok and end_tok.unmarked_hour)

    problem: Problem | None = None
    if start and end:
        same_written = (
            start_tok is not None and end_tok is not None and start_tok.unmarked_hour and end_tok.unmarked_hour
            and (start_tok.hour, start_tok.minute) == (end_tok.hour, end_tok.minute)
        )
        kind = "mismatch" if same_written else conflict_kind(start, end, end_unmarked)
        if kind:
            return TimeReading(blank=True, problem=kind, end_unmarked=end_unmarked)
    if role_conflict or not (start or end):
        problem = "unread"  # 시각은 있는데 역할을 못 정했거나 같은 역할 값이 충돌
    unmarked = frozenset(
        key for key, tok in ((START, start_tok if start else None), (END, end_tok if end else None))
        if tok is not None and tok.form == "hour" and is_unmarked(tok.marker, str(tok.hour))
    )
    return TimeReading(start=start, end=end, problem=problem, unmarked=unmarked, end_unmarked=end_unmarked)


# ── 왜 비었는가 (문구용) ──

def explain(utterances: Sequence[str]) -> Problem | None:
    """slots.utterances를 처음부터 다시 읽어, 마지막 발화에서 시간이 비게 된 이유를 구한다 (merge_slots와 같은 순서).

    마지막 발화에 시각 표현이 없으면 None이다 (이전 발화의 이유를 반복하지 않는다).
    """
    start: str | None = None
    end: str | None = None
    problem: Problem | None = None
    for text in utterances:
        empty = [k for k, v in ((START, start), (END, end)) if v is None]
        reading = read_times(text, empty if len(empty) == 1 else ())
        problem = reading.problem
        if reading.blank:
            start = end = None
            continue
        if reading.start:
            start = reading.start
        if reading.end:
            end = reading.end
        if kind := conflict_kind(start, end, reading.end_unmarked if reading.end else False):
            start = end = None
            problem = kind
    return problem


def retext(fields: Sequence[MissingField], utterances: Sequence[str]) -> list[MissingField]:
    """빠진 항목 중 시작·종료의 question을 '~부터 ~까지' 예시가 있는 문구로 바꾼다 (key·label·개수는 그대로).

    시작·종료가 모두 비면 두 항목의 question이 같다 (ask.message에서는 한 줄로 합친다, slots.ask_message).
    """
    need = [f.key for f in fields if f.key in TIME_KEYS]
    if not need:
        return list(fields)
    problem = explain(utterances)
    both = len(need) == len(TIME_KEYS)
    if problem == "mismatch":
        text = MSG_MISMATCH
    elif problem == "overnight":
        text = MSG_OVERNIGHT
    else:
        text = Q_BOTH if both else (Q_START if need[0] == START else Q_END)
        if problem == "unread":
            text = PREFIX_UNREAD + text
    return [f.model_copy(update={"question": text}) if f.key in TIME_KEYS else f for f in fields]


def has_unread_time(message: str) -> bool:
    """발화에 시각이 있는데 시작·종료를 정하지 못했는가 (DONE 뒤 '16시로 변경' 같은 말)."""
    return read_times(message).problem == "unread"


__all__ = [
    "TIME_READER_V2", "NIGHT_START_HOUR", "TimeReading", "read_times", "conflict_kind", "explain", "retext", "has_unread_time",
    "REPLY_TIME_ROLE", "MSG_MISMATCH", "MSG_OVERNIGHT",
]
