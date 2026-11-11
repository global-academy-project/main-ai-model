"""④ 판단 (고르기 방식): 위험요인·점수·대책은 코드가 정하고, LLM은 근거 번호만 고른다.

judgment.judge()는 LLM이 위험요인 전체를 JSON으로 쓴다. 작은 모델은 이 JSON을 못 맞추거나(실패 시 같은 호출을
한 번 더 한다) 길게 생성해서 오래 걸린다. 이 모듈은 일을 둘로 나눈다.

  코드    위험요인·빈도·강도·대책: rule_hazards()의 작업유형별 템플릿 (judgment.py).
  LLM     위험요인마다 뒷받침하는 근거 번호(E1, M1 …)를 고른다. 출력은 {"H1": ["E1", "E3"], "H2": []} 정도로 짧다.
  코드    번호가 목록에 있는지 확인하고 실제 인용으로 바꾼다. 모르는 번호는 버리고 센다 (UNKNOWN_CITATION).

LLM이 실패하거나 못 읽으면 템플릿의 키워드 인용을 그대로 쓴다 (재시도 없음). 그 경우 judgment 출처는 "fallback"이다.
LLM이 하나라도 고르면 "llm"이다. 이때 위험요인 자체의 출처(Hazard.source)는 그대로 "rule"이다 (코드가 만든 위험요인).
LLM은 위험요인을 새로 만들지도, 점수를 바꾸지도 않는다 ("LLM은 후보만, 최종 값은 코드가").

끄기: JUDGMENT_MODE = "full" 이면 work_register가 이 모듈을 부르지 않고 이전처럼 judgment.judge()를 쓴다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.exceptions import LLMError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.forms import Citation

from .evidence_collector import extract_json_object
from .judgment import (
    MAX_EVIDENCE_IN_PROMPT,
    Hazard,
    JudgeInput,
    kb_citation,
    msds_citation,
    rule_hazards,
)

logger = logging.getLogger(__name__)

JUDGMENT_MODE = "pick"  # "pick": 이 모듈 / "full": 이전 방식(judgment.judge)
PICK_MAX_TOKENS = 256  # 출력은 짧다. 판단 전체를 쓰던 1200보다 훨씬 작게 둔다
PICK_SNIPPET_CHARS = 160  # 근거 한 건당 프롬프트에 넣는 글자 수
PICK_MAX_CITATIONS = 3  # 위험요인 하나에 붙이는 근거 상한
PICK_MEASURE_CHARS = 60  # 프롬프트에 보이는 대책 글자 수

_LOG_RAW_CHARS = 200  # 읽지 못한 응답을 로그에 남기는 길이 (진단용)
_ID_RE = re.compile(r"[EM]\d+", re.IGNORECASE)
_H_KEY_RE = re.compile(r"^\s*H?\s*(\d+)\s*$", re.IGNORECASE)

# 예시는 계획서 9장 데모 문장을 쓰지 않는다.
_EXAMPLE = '{"H1": ["E2", "E5"], "H2": ["E1"], "H3": []}'


def build_pick_system_prompt() -> str:
    return f"""당신은 산업현장 안전관리 에이전트의 근거 선택 담당입니다.
위험요인(H번호)마다, 그 위험요인과 직접 관련된 근거 번호(E 또는 M 번호)를 최대 {PICK_MAX_CITATIONS}개 고르세요.
- 목록에 있는 번호만 쓰세요. 관련된 근거가 없으면 빈 목록 []을 쓰세요.
- 새로운 내용을 쓰지 말고, 설명 없이 JSON 객체 하나만 출력하세요.

[출력 형식 예시]
{_EXAMPLE}
"""


def _one_line(text: str, limit: int) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def build_pick_input(inp: JudgeInput, hazards: Sequence[Hazard]) -> tuple[str, dict[str, Citation]]:
    """프롬프트 본문과 {근거 번호: 인용}. 번호 부여는 judgment.build_judge_input과 같은 규칙이다."""
    refs: dict[str, Citation] = {}
    lines = [f"작업 내용: {_one_line(inp.text, 300)}", "\n[위험요인]"]
    for n, h in enumerate(hazards, 1):
        measures = _one_line(" / ".join(h.measures), PICK_MEASURE_CHARS)
        lines.append(f"[H{n}] {h.hazard} (대책: {measures})")
    lines.append("\n[근거]")
    for n, item in enumerate(inp.pack.items[:MAX_EVIDENCE_IN_PROMPT], 1):
        refs[f"E{n}"] = kb_citation(item)
        lines.append(f"[E{n}] ({item.source}) {_one_line(item.text, PICK_SNIPPET_CHARS)}")
    for n, m in enumerate((m for m in inp.msds if m.found), 1):
        refs[f"M{n}"] = msds_citation(m)
        lines.append(f"[M{n}] MSDS {m.name}: {', '.join(m.ghs_classes)} / {', '.join(m.hazards)}")
    return "\n".join(lines), refs


def _ids(value: object) -> list[str]:
    """값에서 E1·M2 같은 번호를 순서대로 꺼낸다 (목록이든 "E1, E3" 같은 문자열이든)."""
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [v for v in value if isinstance(v, str)]
    else:
        return []
    out: list[str] = []
    for text in raw:
        for token in _ID_RE.findall(text):
            if (key := token.upper()) not in out:
                out.append(key)
    return out


def parse_picks(raw: dict, count: int) -> dict[int, list[str]]:
    """{위험요인 번호(1부터): [근거 번호 …]}. 키는 "H1"·"1" 모두 받는다. 범위 밖 키는 버린다."""
    picks: dict[int, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not (m := _H_KEY_RE.match(key)):
            continue
        n = int(m.group(1))
        if 1 <= n <= count:
            picks[n] = _ids(value)
    return picks


def apply_picks(hazards: list[Hazard], picks: dict[int, list[str]], refs: dict[str, Citation]) -> tuple[int, int]:
    """고른 번호를 인용으로 바꿔 위험요인에 붙인다. (인용을 바꾼 위험요인 수, 버린 번호 수).

    고른 근거가 하나도 없는 위험요인은 템플릿의 키워드 인용을 그대로 둔다.
    템플릿이 붙인 MSDS 인용(물질 정보)은 항상 남긴다.
    """
    changed = 0
    dropped = 0
    for n, h in enumerate(hazards, 1):
        valid: list[Citation] = []
        for ref in picks.get(n, []):
            if ref in refs:
                if refs[ref] not in valid:
                    valid.append(refs[ref])
            else:
                dropped += 1
        if not valid:
            continue
        kept_msds = [c for c in h.citations if c.kind == "MSDS" and c not in valid]
        h.citations = valid[:PICK_MAX_CITATIONS] + kept_msds
        h.needs_review = False
        changed += 1
    return changed, dropped


def judge_pick(llm: LLMClient, inp: JudgeInput, *, max_tokens: int = PICK_MAX_TOKENS) -> tuple[list[Hazard], str, int]:
    """(위험요인, 출처 "llm"|"fallback", 버린 근거 번호 수). judgment.judge()와 같은 모양이다.

    LLM 호출은 한 번뿐이다. LLMError도 여기서 삼킨다: 근거 선택은 없어도 코드가 끝낼 수 있는 단계라서
    실행을 멈추지 않고 키워드 인용으로 계속한다 (경고 로그와 source="fallback"으로 남는다).
    """
    hazards = rule_hazards(inp)
    user, refs = build_pick_input(inp, hazards)
    try:
        resp = llm.chat([Message(role="user", content=user)], system=build_pick_system_prompt(), max_tokens=max_tokens)
    except LLMError as exc:
        logger.warning("근거 선택 LLM 호출에 실패해 키워드 인용으로 계속합니다: %s", type(exc).__name__, exc_info=exc)
        return hazards, "fallback", 0
    if resp.stop_reason == "max_tokens":
        logger.info("근거 선택 응답이 max_tokens에서 잘림 (max_tokens=%s)", max_tokens)
    text = resp.text or ""
    raw = extract_json_object(text)
    if raw is None:
        logger.warning("근거 선택 응답을 읽지 못해 키워드 인용으로 계속합니다. 응답 앞부분: %r", text[:_LOG_RAW_CHARS])
        return hazards, "fallback", 0
    changed, dropped = apply_picks(hazards, parse_picks(raw, len(hazards)), refs)
    if not changed:
        logger.warning("근거 선택 응답에 쓸 수 있는 번호가 없어 키워드 인용으로 계속합니다. 응답 앞부분: %r", text[:_LOG_RAW_CHARS])
        return hazards, "fallback", dropped
    logger.info("근거 선택: 위험요인 %d건 중 %d건의 인용을 LLM이 골랐습니다.", len(hazards), changed)
    return hazards, "llm", dropped


__all__ = ["JUDGMENT_MODE", "PICK_MAX_TOKENS", "judge_pick"]
