"""④ 판단 · ⑤ 자기검증 (계획서 7장 루프의 뒤쪽 단계).

판단: 근거 묶음(EvidencePack)과 MSDS를 보고 위험요인·감소대책 후보를 만든다.
  - LLM에 JSON으로 요청한다 (실패 시 1회 재시도). 그래도 못 읽으면 작업유형별 규칙 템플릿으로 만든다 (source="rule").
  - LLM에는 근거를 짧은 번호([E1], [M1])로 보여주고, 응답의 번호를 코드가 실제 청크 ID로 바꾼다.
  - LLM은 후보만 낸다. 허용 값 정규화, 점수 범위, 위험요인 분류 보정, 잔여 위험 상한은 코드가 정한다.

자기검증 (코드):
  - 모르는 근거 번호는 버린다 (UNKNOWN_CITATION).
  - 근거가 하나도 없는 위험요인은 needs_review=True (UNSUPPORTED_HAZARD).
  - 작업유형별로 꼭 있어야 할 발생형태가 빠지면 규칙 템플릿으로 보충한다 (REQUIRED_HAZARD_ADDED).
  - 끝내 못 찾은 필수 근거(EvidencePack.missing)는 EVIDENCE_MISSING으로 알린다.

빈도 1~5 · 강도 1~4는 임시 척도다 (forms.py 참고).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.agent_work import MsdsDto, VerificationIssue
from app.schemas.forms import ACCIDENT_TYPES, FREQUENCY_MAX, RISK_FACTORS, SEVERITY_MAX, Citation

from .evidence_collector import EvidencePack, extract_json_object

logger = logging.getLogger(__name__)

MAX_HAZARDS = 8
MAX_EVIDENCE_IN_PROMPT = 12
SNIPPET_CHARS = 200

_ACCIDENT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "화재·폭발": ("화재", "폭발·파열"),
    "화재폭발": ("화재", "폭발·파열"),
    "폭발": ("폭발·파열",),
    "파열": ("폭발·파열",),
    "누출": ("누출·접촉",),
    "접촉": ("누출·접촉",),
    "질식": ("산소결핍",),
    "추락": ("떨어짐",),
    "낙하": ("맞음",),
    "전도": ("넘어짐",),
    "충돌": ("부딪힘",),
    "절단": ("절단·베임",),
    "베임": ("절단·베임",),
    "협착": ("끼임",),
}
_RISK_FACTOR_SYNONYMS = {"기계": "기계적", "전기": "전기적", "화학": "화학적", "환경": "작업환경", "인적": "인적·관리", "관리": "인적·관리"}
_FACTOR_BY_ACCIDENT = {
    "화재": "화학적", "폭발·파열": "화학적", "누출·접촉": "화학적", "감전": "전기적",
    "끼임": "기계적", "절단·베임": "기계적", "맞음": "기계적", "깔림": "기계적", "부딪힘": "기계적",
    "떨어짐": "작업환경", "넘어짐": "작업환경", "산소결핍": "작업환경",
}
REQUIRED_ACCIDENTS: dict[str, frozenset[str]] = {
    "화기": frozenset({"화재", "폭발·파열"}),
    "밀폐공간": frozenset({"산소결핍"}),
    "고소": frozenset({"떨어짐"}),
    "전기": frozenset({"감전"}),
    "중량물": frozenset({"맞음", "깔림", "끼임"}),
}


@dataclass
class Hazard:
    hazard: str
    accident_types: list[str]
    risk_factor: str
    frequency: int
    severity: int
    measures: list[str]
    residual_frequency: int
    residual_severity: int
    citations: list[Citation] = field(default_factory=list)
    needs_review: bool = False
    source: Literal["llm", "rule"] = "llm"
    template: str | None = None


@dataclass(frozen=True)
class JudgeInput:
    text: str
    summary: str
    work_types: Sequence[str]
    location: str | None
    hot_work_kind: str | None
    materials: Sequence[str]
    equipment: Sequence[str]
    pack: EvidencePack
    msds: Sequence[MsdsDto]


# ── 근거 → 인용 ─────────────────────────────────────────────


def _snippet(text: str) -> str:
    one = " ".join(text.split())
    return one if len(one) <= SNIPPET_CHARS else one[: SNIPPET_CHARS - 1] + "…"


def kb_citation(item: Any) -> Citation:
    return Citation(kind="KB", ref_id=item.id, source=item.source, snippet=_snippet(item.text))


def msds_citation(m: MsdsDto) -> Citation:
    hazards = ", ".join(m.hazards) or ", ".join(m.ghs_classes)
    return Citation(kind="MSDS", ref_id=m.cas_no or m.query, source=m.source or "MSDS", snippet=_snippet(f"{m.name}: {hazards}"))


# ── 규칙 템플릿 ─────────────────────────────────────────────


@dataclass(frozen=True)
class _Template:
    key: str
    accident_types: tuple[str, ...]
    risk_factor: str
    frequency: int
    severity: int
    residual_frequency: int
    measures: tuple[str, ...]
    keywords: tuple[str, ...]  # 이 단어가 있는 근거를 인용한다


_T = _Template
_TEMPLATES: dict[str, _Template] = {
    "hot_ignition": _T("hot_ignition", ("화재", "폭발·파열"), "화학적", 4, 4, 2,
                       ("작업 전·중 가연성가스 농도 측정", "작업 반경 내 가연물 제거, 불티 비산방지포 설치", "화재감시자 배치, 소화기 비치"),
                       ("불티", "가연성가스", "화재감시자", "비산방지포", "가연물")),
    "hot_fume": _T("hot_fume", ("누출·접촉",), "작업환경", 3, 2, 1, ("국소배기 또는 환기", "방진·방독 마스크 착용"), ("용접", "환기")),
    "grinder": _T("grinder", ("절단·베임",), "기계적", 3, 3, 1,
                  ("방호덮개 부착 확인", "연마석 균열·규격 확인", "보안경·안면보호구 착용"), ("그라인더", "방호덮개", "연마석")),
    "pipe_open": _T("pipe_open", ("누출·접촉",), "화학적", 3, 4, 1,
                    ("내부 물질 배출·퍼지·세정 후 맹판 설치", "관련 밸브 차단·잠금(LOTO)과 표지"), ("퍼지", "맹판", "잔류", "잠금")),
    "confined": _T("confined", ("산소결핍",), "작업환경", 3, 4, 1,
                   ("출입 전·작업 중 산소·유해가스 측정", "작업 전·중 환기", "감시인 상주, 송기마스크·구조 장비 비치"),
                   ("산소", "질식", "감시인", "송기마스크", "적정공기")),
    "height": _T("height", ("떨어짐",), "작업환경", 3, 4, 1,
                 ("작업발판·안전난간 설치", "안전대 부착설비 확보와 안전대 착용", "개구부·채광창 덮개"), ("추락", "떨어", "안전대", "작업발판")),
    "electric": _T("electric", ("감전",), "전기적", 3, 4, 1,
                   ("정전 후 차단기 잠금·표지", "검전기로 무전압 확인", "절연 보호구 착용"), ("감전", "정전", "검전")),
    "lifting": _T("lifting", ("맞음", "깔림"), "기계적", 3, 4, 1,
                  ("줄걸이 용구 점검", "신호수 지정", "인양물 하부 출입 통제"), ("줄걸이", "인양", "낙하", "신호수")),
    "general": _T("general", ("넘어짐",), "작업환경", 2, 2, 1, ("작업 통로 정리정돈", "작업에 맞는 보호구 착용"), ("넘어", "정리")),
}
del _T
_TEMPLATE_BY_WORK_TYPE = {"화기": "hot_ignition", "밀폐공간": "confined", "고소": "height", "전기": "electric", "중량물": "lifting"}
_PIPE_OPEN_RE = re.compile(r"(배관|용기|드럼|밸브|플랜지).{0,20}(교체|절단|개방|분해|해체|보수)")


def _flammables(inp: JudgeInput) -> list[MsdsDto]:
    return [m for m in inp.msds if m.found and any("인화성" in g for g in m.ghs_classes)]


def _hazard_text(key: str, inp: JudgeInput) -> str:
    match key:
        case "hot_ignition":
            tool = inp.hot_work_kind or "화기 작업"
            flam = [m.name for m in _flammables(inp) if m.name]
            target = f"{'·'.join(flam)} 증기" if flam else "주변 가연물"
            return f"{tool} 중 불티·열원이 {target}에 착화"
        case "hot_fume":
            return "용접 흄·유해가스 흡입"
        case "grinder":
            return "그라인더 연마석 파손·반동으로 베임"
        case "pipe_open":
            return "설비 개방 시 배관·용기 내 잔류 물질 누출"
        case "confined":
            return "밀폐공간 산소결핍·유해가스로 질식"
        case "height":
            return f"{inp.location or '작업 위치'}에서 떨어짐"
        case "electric":
            return "충전부 접촉으로 감전"
        case "lifting":
            return "인양물 낙하·깔림"
        case _:
            return "작업 중 넘어짐·부딪힘"


def template_hazard(key: str, inp: JudgeInput, *, needs_review: bool = False) -> Hazard:
    t = _TEMPLATES[key]
    measures = list(t.measures)
    citations = [kb_citation(i) for i in inp.pack.items if any(kw in i.text for kw in t.keywords)][:3]
    if key == "hot_ignition":
        flam = _flammables(inp)
        measures += [f"{m.name} 용기 밀폐·격리" for m in flam if m.name]
        citations += [msds_citation(m) for m in flam]
    return Hazard(
        hazard=_hazard_text(key, inp),
        accident_types=list(t.accident_types),
        risk_factor=t.risk_factor,
        frequency=t.frequency,
        severity=t.severity,
        measures=measures,
        residual_frequency=t.residual_frequency,
        residual_severity=t.severity,
        citations=citations,
        needs_review=needs_review or not citations,
        source="rule",
        template=key,
    )


def rule_hazards(inp: JudgeInput) -> list[Hazard]:
    keys: list[str] = [_TEMPLATE_BY_WORK_TYPE[w] for w in inp.work_types if w in _TEMPLATE_BY_WORK_TYPE]
    kind = inp.hot_work_kind or ""
    if "화기" in inp.work_types and "용접" in kind:
        keys.append("hot_fume")
    if "그라인더" in inp.equipment or "그라인더" in kind:
        keys.append("grinder")
    if _PIPE_OPEN_RE.search(inp.text):
        keys.append("pipe_open")
    if not keys:
        keys.append("general")
    return [template_hazard(k, inp) for k in dict.fromkeys(keys)]


# ── LLM 판단 ────────────────────────────────────────────────

_RETRY_SUFFIX = "\n\n(이전 응답은 JSON으로 읽을 수 없었습니다. 형식에 맞는 JSON 객체 하나만 출력하세요.)"

# 예시는 계획서 9장 데모 문장을 쓰지 않는다.
_EXAMPLE = {
    "hazards": [
        {
            "hazard": "용접 불티가 아세톤 증기에 착화",
            "accident_types": ["화재", "폭발·파열"],
            "risk_factor": "화학적",
            "frequency": 4,
            "severity": 4,
            "measures": ["작업 전 가연성가스 측정", "아세톤 용기 격리", "화재감시자 배치"],
            "residual_frequency": 2,
            "residual_severity": 4,
            "evidence": ["E1", "M1"],
        }
    ]
}


def build_judge_system_prompt() -> str:
    return f"""당신은 산업현장 안전관리 에이전트의 위험요인 판단 담당입니다.
작업 내용과 검색된 근거를 보고 일일 위험성평가에 넣을 위험요인과 감소대책을 JSON 객체 하나로만 출력하세요. 설명은 쓰지 마세요.

[규칙]
- 근거 목록에 있는 내용으로만 판단합니다. 각 위험요인의 evidence에는 뒷받침하는 근거 번호(E1, M1 …)를 넣습니다.
- 근거를 찾지 못했다고 표시된 항목은 단정하지 말고, 쓰더라도 evidence를 비워 둡니다.
- 작업 내용에 없는 설비명·수치를 지어내지 않습니다.
- accident_types 허용 값: {", ".join(ACCIDENT_TYPES)}
- risk_factor 허용 값: {", ".join(RISK_FACTORS)}
- frequency 1~{FREQUENCY_MAX}, severity 1~{SEVERITY_MAX}. residual_*은 감소대책 적용 후 값입니다.
- 위험요인은 {MAX_HAZARDS}개 이하.

[출력 형식]
{{"hazards": [{{"hazard": "", "accident_types": [], "risk_factor": "", "frequency": 0, "severity": 0, "measures": [], "residual_frequency": 0, "residual_severity": 0, "evidence": []}}]}}

[예시 출력]
{json.dumps(_EXAMPLE, ensure_ascii=False)}
"""


def build_judge_input(inp: JudgeInput) -> tuple[str, dict[str, Citation]]:
    refs: dict[str, Citation] = {}
    lines = [
        f"작업 요약: {inp.summary}",
        f"작업 내용: {inp.text}",
        f"작업유형: {', '.join(inp.work_types) or '모름'}",
        f"장소: {inp.location or '모름'}",
    ]
    if inp.hot_work_kind:
        lines.append(f"화기 종류: {inp.hot_work_kind}")
    if inp.equipment:
        lines.append(f"장비: {', '.join(inp.equipment)}")
    lines.append("\n[근거]")
    for n, item in enumerate(inp.pack.items[:MAX_EVIDENCE_IN_PROMPT], 1):
        refs[f"E{n}"] = kb_citation(item)
        lines.append(f"[E{n}] ({item.source}) {_snippet(item.text)}")
    for n, m in enumerate((m for m in inp.msds if m.found), 1):
        refs[f"M{n}"] = msds_citation(m)
        lines.append(f"[M{n}] MSDS {m.name}: {', '.join(m.ghs_classes)} / {', '.join(m.hazards)}")
    if inp.pack.missing:
        lines.append("\n[근거를 찾지 못한 항목 — 단정하지 말 것]")
        lines += [f"- {d}" for d in inp.pack.missing]
    return "\n".join(lines), refs


def _accident_types(value: object) -> list[str]:
    raw = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    out: list[str] = []
    for v in raw:
        if not isinstance(v, str):
            continue
        key = v.strip()
        for norm in (key,) if key in ACCIDENT_TYPES else _ACCIDENT_SYNONYMS.get(key.replace(" ", ""), ()):
            if norm not in out:
                out.append(norm)
    return out or ["기타"]


def _risk_factor(value: object, accident_types: Sequence[str]) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v in RISK_FACTORS:
            return v
        for key, norm in _RISK_FACTOR_SYNONYMS.items():
            if key in v:
                return norm
    return _FACTOR_BY_ACCIDENT.get(accident_types[0], "인적·관리")


def _int_in(value: object, low: int, high: int, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(low, min(high, int(round(value))))


def parse_hazards(raw: dict, refs: dict[str, Citation]) -> tuple[list[Hazard], int]:
    """LLM JSON → Hazard 목록. (위험요인, 버린 근거 번호 수)."""
    hazards: list[Hazard] = []
    dropped_refs = 0
    seen: set[str] = set()
    items = raw.get("hazards") if isinstance(raw.get("hazards"), list) else []
    for h in items:
        if not isinstance(h, dict) or not isinstance(h.get("hazard"), str) or not h["hazard"].strip():
            continue
        text = " ".join(h["hazard"].split())[:200]
        if text in seen:
            continue
        seen.add(text)
        accident_types = _accident_types(h.get("accident_types"))
        frequency = _int_in(h.get("frequency"), 1, FREQUENCY_MAX, 3)
        severity = _int_in(h.get("severity"), 1, SEVERITY_MAX, 3)
        measures = [" ".join(m.split()) for m in h.get("measures", []) if isinstance(m, str) and m.strip()] if isinstance(h.get("measures"), list) else []
        citations: list[Citation] = []
        for ref in h.get("evidence", []) if isinstance(h.get("evidence"), list) else []:
            if isinstance(ref, str) and ref.strip() in refs:
                c = refs[ref.strip()]
                if c not in citations:
                    citations.append(c)
            else:
                dropped_refs += 1
        hazards.append(
            Hazard(
                hazard=text,
                accident_types=accident_types,
                risk_factor=_risk_factor(h.get("risk_factor"), accident_types),
                frequency=frequency,
                severity=severity,
                measures=measures or ["감소대책 검토 필요"],
                # 잔여 위험은 원래 위험보다 클 수 없다
                residual_frequency=min(_int_in(h.get("residual_frequency"), 1, FREQUENCY_MAX, max(1, frequency - 2)), frequency),
                residual_severity=min(_int_in(h.get("residual_severity"), 1, SEVERITY_MAX, severity), severity),
                citations=citations,
                needs_review=not citations or not measures,
                source="llm",
            )
        )
        if len(hazards) == MAX_HAZARDS:
            break
    return hazards, dropped_refs


def judge(llm: LLMClient, inp: JudgeInput, *, max_tokens: int = 1200) -> tuple[list[Hazard], Literal["llm", "fallback"], int]:
    """(위험요인, 출처, 버린 근거 번호 수). LLMError는 그대로 올린다."""
    system = build_judge_system_prompt()
    user, refs = build_judge_input(inp)
    raw = None
    for attempt in (user, user + _RETRY_SUFFIX):
        resp = llm.chat([Message(role="user", content=attempt)], system=system, max_tokens=max_tokens)
        if resp.stop_reason == "max_tokens":
            logger.info("판단 응답이 max_tokens에서 잘림")
        raw = extract_json_object(resp.text or "")
        if raw is not None:
            break
    if raw is not None:
        hazards, dropped = parse_hazards(raw, refs)
        if hazards:
            return hazards, "llm", dropped
    logger.warning("판단 응답을 읽지 못해 규칙 템플릿으로 계속합니다.")
    return rule_hazards(inp), "fallback", 0


# ── 자기검증 ────────────────────────────────────────────────


def verify(hazards: list[Hazard], inp: JudgeInput, *, dropped_refs: int = 0) -> tuple[list[Hazard], list[VerificationIssue]]:
    issues: list[VerificationIssue] = []
    result = list(hazards)
    if dropped_refs:
        issues.append(VerificationIssue(code="UNKNOWN_CITATION", message=f"근거 목록에 없는 인용 {dropped_refs}건을 지웠습니다."))
    present = {a for h in result for a in h.accident_types}
    for work_type in inp.work_types:
        required = REQUIRED_ACCIDENTS.get(work_type)
        if required and not (present & required):
            added = template_hazard(_TEMPLATE_BY_WORK_TYPE[work_type], inp, needs_review=True)
            result.append(added)
            present |= set(added.accident_types)
            issues.append(VerificationIssue(
                code="REQUIRED_HAZARD_ADDED",
                message=f"{work_type}작업의 필수 위험요인({'/'.join(sorted(required))})이 없어 규칙으로 보충했습니다.",
                target=added.hazard,
            ))
    for h in result:
        if not h.citations:
            h.needs_review = True
            issues.append(VerificationIssue(code="UNSUPPORTED_HAZARD", message="근거 인용이 없는 위험요인입니다. 승인 전 확인하세요.", target=h.hazard))
    for description in inp.pack.missing:
        issues.append(VerificationIssue(code="EVIDENCE_MISSING", message=f"필수 근거를 찾지 못했습니다: {description}", target=description))
    for f in inp.pack.search_failures:
        issues.append(VerificationIssue(code="SEARCH_FAILED", message=f"검색 일부 실패 ({f.code}): {f.reason}", target=f.query))
    for m in inp.msds:
        if not m.found:
            issues.append(VerificationIssue(code="MSDS_NOT_FOUND", message=f"'{m.query}'의 MSDS를 찾지 못했습니다. 직접 확인하세요.", target=m.query))
    return result[: MAX_HAZARDS + len(REQUIRED_ACCIDENTS)], issues
