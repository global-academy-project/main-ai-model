from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class _Rule:
    code: str
    label: str
    law: str
    phrases: tuple[str, ...]
    section_phrases: tuple[str, ...] = ()


_RULES = (
    _Rule(
        "WORKPLACE_MEASUREMENT",
        "작업환경측정 대상",
        "산업안전보건법",
        ("작업환경측정대상물질",),
    ),
    _Rule(
        "SPECIAL_HEALTH_EXAM",
        "특수건강진단 대상",
        "산업안전보건법",
        ("특수건강진단대상물질",),
    ),
    _Rule(
        "CONTROLLED_HAZARDOUS_SUBSTANCE",
        "관리대상 유해물질",
        "산업안전보건법",
        ("관리대상유해물질",),
    ),
    _Rule(
        "PROCESS_SAFETY_REPORT",
        "공정안전보고서 제출 대상",
        "산업안전보건법",
        ("공정안전보고서(psm)제출대상물질", "공정안전보고서제출대상물질"),
    ),
    _Rule(
        "EXPOSURE_LIMIT_SET",
        "노출기준 설정 물질",
        "산업안전보건법",
        ("노출기준설정물질",),
    ),
    _Rule(
        "PERMISSIBLE_LIMIT_SET",
        "허용기준 설정 물질",
        "산업안전보건법",
        ("허용기준설정물질",),
    ),
    _Rule(
        "ACCIDENT_PREPARED_SUBSTANCE",
        "사고대비물질",
        "화학물질관리법",
        ("사고대비물질",),
    ),
    _Rule(
        "DANGEROUS_SUBSTANCE_REGULATION",
        "위험물안전관리법 규제 대상",
        "위험물안전관리법",
        (),
        section_phrases=("위험물안전관리법",),
    ),
    _Rule(
        "DESIGNATED_WASTE",
        "지정폐기물",
        "폐기물관리법",
        ("지정폐기물",),
    ),
    _Rule(
        "EXISTING_CHEMICAL_SUBSTANCE",
        "기존화학물질",
        "화학물질의 등록 및 평가 등에 관한 법률",
        ("기존화학물질",),
    ),
)

_NOT_APPLICABLE = {"해당없음", "자료없음", "없음", "-"}


def classify_regulations(regulations: Iterable[str]) -> list[dict[str, object]]:
    """Map recognized MSDS section-15 phrases to stable tags with source evidence."""
    classified: dict[str, dict[str, object]] = {}
    for regulation in regulations:
        if not isinstance(regulation, str) or not regulation.strip():
            continue

        section, _, detail = regulation.partition(":")
        section_text = section.strip()
        detail_text = detail.strip() if detail else regulation.strip()
        normalized_detail = _normalize(detail_text)
        if normalized_detail in _NOT_APPLICABLE:
            continue

        for rule in _RULES:
            section_matches = any(
                _normalize(phrase) in _normalize(section_text)
                for phrase in rule.section_phrases
            )
            detail_matches = any(
                _normalize(phrase) in normalized_detail for phrase in rule.phrases
            )
            if not section_matches and not detail_matches:
                continue

            tag = classified.setdefault(
                rule.code,
                {
                    "code": rule.code,
                    "label": rule.label,
                    "law": rule.law,
                    "evidence": [],
                },
            )
            evidence = tag["evidence"]
            if isinstance(evidence, list) and regulation not in evidence:
                evidence.append(regulation)

    return list(classified.values())


def _normalize(value: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", value).casefold()