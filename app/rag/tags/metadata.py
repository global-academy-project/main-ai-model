from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

TAG_STATUS_VALUES = ("KNOWN", "UNKNOWN", "NOT_APPLICABLE")
SIF_POTENTIAL_VALUES = ("POTENTIAL", "PRECURSOR", "NOT_POTENTIAL")
GHS_CLASS_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
CAS_NUMBER_PATTERN = re.compile(r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")
TAG_STATUS_AXES = (
    "severity",
    "accident_type",
    "hazard",
    "work_type",
    "substance",
    "ghs",
    "sif_potential",
)


def _has_value(value: object) -> bool:
    return bool(value) if isinstance(value, (list, tuple, set, dict)) else value is not None


def is_valid_cas_number(value: str) -> bool:
    if not CAS_NUMBER_PATTERN.fullmatch(value):
        return False
    digits = value.replace("-", "")
    checksum = int(digits[-1])
    weighted_sum = sum(int(digit) * weight for weight, digit in enumerate(reversed(digits[:-1]), start=1))
    return weighted_sum % 10 == checksum


def validate_substance_links(substances: object, cas_values: object, ghs_values: object) -> None:
    if not isinstance(substances, list) or not isinstance(cas_values, list) or not isinstance(ghs_values, list):
        raise ValueError("substances, cas, ghs는 배열이어야 합니다.")
    if any(not isinstance(value, str) or not is_valid_cas_number(value) for value in cas_values):
        raise ValueError("cas 배열에 형식 또는 체크섬이 잘못된 CAS 번호가 있습니다.")
    if any(not isinstance(value, str) or not GHS_CLASS_CODE_PATTERN.fullmatch(value) for value in ghs_values):
        raise ValueError("ghs 배열에 대문자 정규화 코드가 아닌 값이 있습니다.")
    linked_cas: list[str] = []
    linked_ghs: list[str] = []
    for item in substances:
        if not isinstance(item, Mapping):
            raise ValueError("substances 항목은 객체여야 합니다.")
        cas = item.get("cas")
        source = item.get("source")
        ghs = item.get("ghs")
        if not isinstance(cas, str) or not is_valid_cas_number(cas):
            raise ValueError("substances.cas의 CAS 형식 또는 체크섬이 올바르지 않습니다.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("substances.source는 비어 있을 수 없습니다.")
        if not isinstance(ghs, list) or any(
            not isinstance(code, str) or not GHS_CLASS_CODE_PATTERN.fullmatch(code) for code in ghs
        ):
            raise ValueError("substances.ghs는 대문자 정규화 코드 배열이어야 합니다.")
        linked_cas.append(cas)
        linked_ghs.extend(ghs)
    if len(set(linked_cas)) != len(linked_cas):
        raise ValueError("substances에 중복 CAS가 있습니다.")
    if len(set(cas_values)) != len(cas_values) or set(cas_values) != set(linked_cas):
        raise ValueError("cas 배열은 substances의 CAS 목록과 같아야 합니다.")
    if len(set(ghs_values)) != len(ghs_values) or set(ghs_values) != set(linked_ghs):
        raise ValueError("ghs 배열은 substances의 GHS 목록과 같아야 합니다.")


def build_tag_status(
    values: Mapping[str, object],
    *,
    not_applicable: Iterable[str] = (),
) -> dict[str, str]:
    """Describe whether each public-data tag axis has a value, is unknown, or does not apply."""
    not_applicable_axes = set(not_applicable)
    unknown_axes = not_applicable_axes - set(TAG_STATUS_AXES)
    if unknown_axes:
        raise ValueError(f"Unknown tag status axes: {sorted(unknown_axes)}")

    status: dict[str, str] = {}
    for axis in TAG_STATUS_AXES:
        if axis in not_applicable_axes:
            if _has_value(values.get(axis)):
                raise ValueError(f"{axis} 값이 있으므로 NOT_APPLICABLE로 표시할 수 없습니다.")
            status[axis] = "NOT_APPLICABLE"
            continue
        value = values.get(axis)
        status[axis] = "KNOWN" if _has_value(value) else "UNKNOWN"
    return status


def validate_tag_status(
    status: object,
    values: Mapping[str, object],
    *,
    not_applicable: Iterable[str] = (),
) -> dict[str, str]:
    """Validate supplied states against values and fill omitted states conservatively."""
    resolved = build_tag_status(values, not_applicable=not_applicable)
    if status is None or status == {}:
        return resolved
    if not isinstance(status, Mapping):
        raise ValueError("tag_status는 객체여야 합니다.")
    unknown_axes = set(status) - set(TAG_STATUS_AXES)
    if unknown_axes:
        raise ValueError(f"tag_status에 알 수 없는 축이 있습니다: {sorted(unknown_axes)}")

    for axis, state in status.items():
        if state not in TAG_STATUS_VALUES:
            raise ValueError(f"tag_status.{axis} 값은 {', '.join(TAG_STATUS_VALUES)} 중 하나여야 합니다.")
        if resolved[axis] == "NOT_APPLICABLE" and state != "NOT_APPLICABLE":
            raise ValueError(f"tag_status.{axis}는 해당 자료 유형에 적용되지 않습니다.")
        if resolved[axis] == "KNOWN" and state != "KNOWN":
            raise ValueError(f"tag_status.{axis}=KNOWN에는 해당 축의 태그값이 필요합니다.")
        if resolved[axis] != "KNOWN" and state == "KNOWN":
            raise ValueError(f"tag_status.{axis}=KNOWN인데 해당 축의 태그값이 없습니다.")
        resolved[axis] = state
    return resolved
