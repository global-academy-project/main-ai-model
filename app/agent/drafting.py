"""⑥ 출력: 판단 결과를 관리 항목 초안(dict)으로 옮긴다. 검증은 fill_form 도구가 한다.

  위험성평가(JSA) · 작업허가서(필요한 것만) · 장비·작업구역 점검표 · TBM 교육 포인트
  계획서 9장 시나리오 A 출력 1~4와 같은 구성이다.

허가서의 측정값·배치 칸은 check_rules 결과를 그대로 옮긴다 (판정은 C의 규칙 엔진이 한다).
점검표는 오늘 투입 장비와 작업유형에 맞는 항목만 고른다 (계획서 3장 4번).
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from app.schemas.agent_work import Classification, GuidanceNote, MsdsDto, RuleCheckDto, VerificationIssue, WorkSlots
from app.schemas.forms import Citation
from . import draft_extras as extras
from .judgment import Hazard

PERMIT_TITLES = {
    "HOT_WORK_PERMIT": "화기작업허가서",
    "CONFINED_SPACE_PERMIT": "밀폐공간 작업허가서",
    "HEIGHT_WORK_PERMIT": "고소작업허가서",
}
_PERMIT_ACCIDENTS = {
    "HOT_WORK_PERMIT": {"화재", "폭발·파열", "누출·접촉"},
    "CONFINED_SPACE_PERMIT": {"산소결핍"},
    "HEIGHT_WORK_PERMIT": {"떨어짐"},
}
_ROLE_LABELS = {"FIRE_WATCHER": "화재감시자", "ATTENDANT": "감시인", "SUPERVISOR": "관리감독자"}
# 주변 취급 물질을 되묻는 허가 (slots.py: 화기·밀폐공간 작업)
_MATERIAL_PERMITS = frozenset({"HOT_WORK_PERMIT", "CONFINED_SPACE_PERMIT"})

_EQUIPMENT_CHECKS: dict[str, tuple[str, ...]] = {
    "그라인더": ("그라인더 방호덮개 부착 상태", "연마석 균열·규격 확인"),
    "용접기": ("용접기 접지·케이블 피복 상태", "전격방지기 작동"),
    "토치": ("가스 호스 손상·역화방지기 부착",),
    "가스절단기": ("가스 호스 손상·역화방지기 부착", "가스 용기 전도 방지"),
    "크레인": ("줄걸이 용구 손상 여부", "과부하방지장치 작동"),
    "호이스트": ("훅 해지장치·와이어 상태",),
    "체인블록": ("훅 해지장치·체인 상태",),
    "지게차": ("전조등·후진경보기 작동", "포크 손상 여부"),
    "고소작업대": ("과상승방지장치 작동", "아웃트리거·바퀴 고정"),
    "사다리": ("미끄럼 방지·전도 방지 조치",),
    "비계": ("작업발판·안전난간 설치 상태", "바퀴 고정"),
    "송풍기": ("송풍기 작동·덕트 연결 상태",),
    "가스측정기": ("가스측정기 교정 상태",),
    "발전기": ("발전기 접지 상태",),
    "임팩트렌치": ("소켓·앤빌 고정 상태", "에어 호스·전원 코드 연결부 상태"),
    "드릴": ("비트 고정·균열 확인", "전원 코드 피복·접지 상태"),
}
_WORK_TYPE_CHECKS: dict[str, tuple[str, ...]] = {
    "화기": ("소화기 비치", "불티 비산방지포 설치", "작업 반경 내 가연물 제거"),
    "밀폐공간": ("환기설비 작동", "송기마스크·구조 장비 비치", "출입 인원 확인판 설치"),
    "고소": ("안전대 부착설비 확인", "개구부 덮개 설치"),
    "전기": ("차단기 잠금장치·표지 부착", "검전기 준비"),
    "중량물": ("인양물 하부 출입 통제 구역 표시",),
}

_GUIDANCE: dict[str, tuple[str, str]] = {
    "화기": ("화기작업허가가 필요합니다",
             "불꽃이나 불티가 주변 가연물·인화성 증기에 옮겨붙으면 화재·폭발로 이어집니다. 작업 전 가스 농도를 재고, "
             "불이 붙을 만한 것을 치우거나 덮고, 불티를 지켜볼 사람(화재감시자)을 둡니다."),
    "밀폐공간": ("밀폐공간 작업허가가 필요합니다",
              "환기가 안 되는 곳은 산소가 부족하거나 유해가스가 차 있어 몇 번 숨 쉬는 것만으로 쓰러질 수 있습니다. "
              "들어가기 전 공기를 재고 환기하며, 밖에서 지켜보는 사람(감시인)을 둡니다."),
    "고소": ("고소작업허가가 필요합니다",
           "높은 곳에서 떨어지면 크게 다칩니다. 작업발판과 난간을 설치하고, 안전대를 걸 곳을 마련해 반드시 착용합니다."),
    "전기": ("전원 차단 절차가 필요합니다", "전원을 끊지 않고 만지면 감전됩니다. 차단기를 내리고 잠근 뒤 전기가 없는지 확인합니다."),
    "중량물": ("인양 작업 계획이 필요합니다", "매단 물건이 떨어지거나 깔리는 사고가 납니다. 줄걸이를 점검하고 아래에 사람이 없게 합니다."),
}


def _period(slots: WorkSlots) -> dict[str, str | None]:
    return {"start": slots.start_time, "end": slots.end_time}


def risk_assessment_data(summary: str, slots: WorkSlots, work_types: Sequence[str], hazards: Sequence[Hazard]) -> dict[str, Any]:
    return {
        "work_summary": summary,
        "location": slots.location,
        "work_types": list(work_types),
        "period": _period(slots),
        "headcount": slots.headcount,
        "items": [
            {
                "seq": n,
                "hazard": h.hazard,
                "accident_types": h.accident_types,
                "risk_factor": h.risk_factor,
                "frequency": h.frequency,
                "severity": h.severity,
                "measures": h.measures,
                "residual_frequency": h.residual_frequency,
                "residual_severity": h.residual_severity,
                "citations": [c.model_dump() for c in h.citations],
                "needs_review": h.needs_review,
                "source": h.source,
            }
            for n, h in enumerate(hazards, 1)
        ],
    }


def permit_data(
    form_type: str,
    slots: WorkSlots,
    hazards: Sequence[Hazard],
    rule_check: RuleCheckDto | None,
    raw_evaluations: Sequence[Mapping[str, Any]],
    msds: Sequence[MsdsDto],
    issues: list[VerificationIssue] | None = None,
) -> dict[str, Any]:
    """issues를 주면, 양식에 맞지 않아 칸으로 옮기지 못한 판정 항목을 거기에 남긴다 (실행을 멈추지 않는다)."""
    wanted = _PERMIT_ACCIDENTS[form_type]
    related = [n for n, h in enumerate(hazards, 1) if wanted & set(h.accident_types)]
    precautions: list[str] = []
    for n in related:
        for m in hazards[n - 1].measures:
            if m not in precautions:
                precautions.append(m)
    measurements, assignments = [], []
    for e in raw_evaluations:
        kind, entry, skipped = extras.classify_evaluation(e, _ROLE_LABELS)
        if skipped is not None:
            _note_skipped(issues, form_type, skipped)
        elif kind == "measurement":
            measurements.append(entry)
        elif kind == "assignment":
            assignments.append(entry)
        elif kind == "check":
            precautions.append(entry["message"])
    blocked_reasons = extras.block_reasons(raw_evaluations)  # 칸을 건너뛴 항목도 원래 판정 값으로 계산한다
    if rule_check is None:
        blocked_reasons = ["규칙 판정을 하지 못했습니다. 판정 전에는 승인 요청을 할 수 없습니다."]
    data: dict[str, Any] = {
        "title": PERMIT_TITLES[form_type],
        "location": slots.location,
        "period": _period(slots),
        "headcount": slots.headcount,
        "measurements": measurements,
        "assignments": assignments,
        "precautions": precautions,
        "risk_item_seqs": related,
        "approval_blocked": bool(blocked_reasons),
        "block_reasons": blocked_reasons,
    }
    if form_type == "HOT_WORK_PERMIT":
        data["hot_work_kind"] = slots.hot_work_kind
    if form_type in _MATERIAL_PERMITS:
        by_name = {m.name or m.query: m for m in msds}
        data["nearby_hazardous_materials"] = [
            {
                "name": name,
                "location": slots.material_locations.get(name),
                "category": (extras.flammable_category(by_name[name].ghs_classes) if name in by_name else None),
                "cas_no": by_name[name].cas_no if name in by_name else None,
            }
            for name in slots.nearby_materials
        ]
    return data


def _note_skipped(issues: list[VerificationIssue] | None, form_type: str, rule_code: str) -> None:
    if issues is None:
        return
    issue = VerificationIssue(
        code="PERMIT_EVALUATION_SKIPPED",
        message=f"규칙 판정 '{rule_code}'을(를) {PERMIT_TITLES[form_type]}의 칸으로 옮기지 못했습니다 (항목·단위·역할 값이 양식과 다릅니다). "
                "승인 차단 여부는 원래 판정 값으로 계산했습니다.",
        target=rule_code,
    )
    if issue not in issues:
        issues.append(issue)


def checklist_data(slots: WorkSlots, work_types: Sequence[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: str, target: str, reason: str) -> None:
        if item not in seen:
            seen.add(item)
            items.append({"seq": len(items) + 1, "item": item, "target": target, "reason": reason})

    for equipment in slots.equipment:
        for check in _EQUIPMENT_CHECKS.get(equipment, ()):
            add(check, equipment, f"투입 장비: {equipment}")
    for work_type in work_types:
        for check in _WORK_TYPE_CHECKS.get(work_type, ()):
            add(check, "작업구역", f"작업유형: {work_type}")
    if not items:
        add("작업구역 정리정돈·통로 확보", "작업구역", "기본 점검")
    return {"location": slots.location, "items": items}


def tbm_data(
    hazards: Sequence[Hazard], block_titles: Sequence[str], max_points: int = 6, msds: Sequence[MsdsDto] = ()
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(point: str, seqs: list[int]) -> None:
        if point not in seen and len(points) < max_points:
            seen.add(point)
            points.append({"seq": len(points) + 1, "point": point, "risk_item_seqs": seqs})

    if block_titles:
        add(f"{', '.join(block_titles)} 전에는 작업을 시작하지 않는다", [])
    ranked = sorted(enumerate(hazards, 1), key=lambda t: -(t[1].frequency * t[1].severity))
    for n, h in ranked:
        add(f"{h.hazard}: {h.measures[0]}", [n])
    for text in extras.first_aid_points(msds):  # MSDS 응급조치: 마지막 공통 요점 자리를 남기고 빈 자리에만 넣는다
        if len(points) < max_points - 1:
            add(text, [])
    add("비상 시 작업중지 신호·대피 경로·소화기 위치 공유", [])
    return {"points": points}


def guidance_notes(classification: Classification, missing_evidence: Sequence[str]) -> list[GuidanceNote]:
    notes = [GuidanceNote(title=_GUIDANCE[w][0], why=_GUIDANCE[w][1]) for w in classification.work_types if w in _GUIDANCE]
    notes.append(GuidanceNote(title="위험성평가를 작성해 승인받습니다",
                              why="오늘 작업에서 무엇이 위험한지 미리 적고 대책을 정해 두는 서류입니다. 법적 의무이므로 기록으로 남깁니다."))
    if missing_evidence:
        notes.append(GuidanceNote(title="확인이 필요한 항목이 있습니다",
                                  why="자료에서 찾지 못한 기준이 있습니다: " + ", ".join(missing_evidence) + ". 담당자에게 확인하세요."))
    return notes


def collect_citations(hazards: Sequence[Hazard]) -> list[Citation]:
    out: list[Citation] = []
    for h in hazards:
        for c in h.citations:
            if c not in out:
                out.append(c)
    return out
