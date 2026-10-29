"""[연습용 mock] lookup_msds (A 담당. 실제는 공단 MSDS API를 CAS 번호 등으로 조회한다).

입력  {"materials": [str, ...]}         물질 이름 (최대 20개)
출력  {"results": [{"query", "found", "name", "cas_no", "ghs_classes", "hazards", "first_aid", "source"}]}
      사전에 없는 물질은 found=false로 돌려준다 (오류가 아니다).

아래 값은 연습용으로 적은 것이며 공단 MSDS로 확인하지 않았다. 실제 판단에 쓰지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.exceptions import ToolError

MAX_MATERIALS = 20
_SOURCE = "[mock] MSDS 연습 데이터 (공단 MSDS API로 확인 필요)"

_MSDS: dict[str, dict[str, Any]] = {
    "톨루엔": {"cas_no": "108-88-3", "ghs_classes": ["인화성 액체", "흡인 유해성", "특정표적장기 독성"],
              "hazards": ["고인화성 액체 및 증기", "증기가 공기와 폭발성 혼합물을 만들 수 있음"], "first_aid": "흡입 시 신선한 공기로 옮긴다."},
    "메탄올": {"cas_no": "67-56-1", "ghs_classes": ["인화성 액체", "급성 독성"],
              "hazards": ["고인화성 액체 및 증기", "삼키거나 흡입하면 유독함"], "first_aid": "삼켰으면 즉시 의료기관에 연락한다."},
    "에탄올": {"cas_no": "64-17-5", "ghs_classes": ["인화성 액체"], "hazards": ["고인화성 액체 및 증기"], "first_aid": "눈에 들어가면 물로 씻는다."},
    "아세톤": {"cas_no": "67-64-1", "ghs_classes": ["인화성 액체", "눈 자극성"], "hazards": ["고인화성 액체 및 증기"], "first_aid": "흡입 시 신선한 공기로 옮긴다."},
    "벤젠": {"cas_no": "71-43-2", "ghs_classes": ["인화성 액체", "발암성"], "hazards": ["고인화성 액체 및 증기", "암을 일으킬 수 있음"], "first_aid": "노출 시 의료기관 진찰."},
    "자일렌": {"cas_no": "1330-20-7", "ghs_classes": ["인화성 액체", "피부 자극성"], "hazards": ["인화성 액체 및 증기"], "first_aid": "피부에 묻으면 물로 씻는다."},
    "수소": {"cas_no": "1333-74-0", "ghs_classes": ["인화성 가스", "고압가스"], "hazards": ["극인화성 가스", "누출 시 폭발 위험"], "first_aid": "누출 지역에서 대피한다."},
    "프로판": {"cas_no": "74-98-6", "ghs_classes": ["인화성 가스", "고압가스"], "hazards": ["극인화성 가스"], "first_aid": "누출 지역에서 대피한다."},
    "아세틸렌": {"cas_no": "74-86-2", "ghs_classes": ["인화성 가스", "고압가스"], "hazards": ["극인화성 가스", "공기 없이도 폭발할 수 있음"], "first_aid": "누출 지역에서 대피한다."},
    "황화수소": {"cas_no": "7783-06-4", "ghs_classes": ["인화성 가스", "급성 독성"], "hazards": ["흡입하면 치명적임"], "first_aid": "구조자는 송기마스크를 착용한다."},
    "암모니아": {"cas_no": "7664-41-7", "ghs_classes": ["급성 독성", "피부 부식성"], "hazards": ["흡입하면 유독함", "심한 화상"], "first_aid": "다량의 물로 씻는다."},
    "질소": {"cas_no": "7727-37-9", "ghs_classes": ["고압가스"], "hazards": ["밀폐된 곳에서 산소를 밀어내 질식을 일으킬 수 있음"], "first_aid": "신선한 공기로 옮긴다."},
}


def lookup_msds(arguments: Mapping[str, Any]) -> dict[str, Any]:
    materials = arguments.get("materials")
    if not isinstance(materials, list) or not all(isinstance(m, str) for m in materials):
        raise ToolError("TOOL_INPUT_INVALID", "lookup_msds에는 물질 이름 목록(materials)이 필요합니다.", tool="lookup_msds")
    if len(materials) > MAX_MATERIALS:
        raise ToolError("TOOL_INPUT_INVALID", f"물질은 한 번에 {MAX_MATERIALS}개까지 조회할 수 있습니다.", tool="lookup_msds")
    results = []
    for name in materials:
        entry = _MSDS.get(name.strip())
        if entry is None:
            results.append({"query": name, "found": False})
        else:
            results.append({"query": name, "found": True, "name": name.strip(), "source": _SOURCE, **entry})
    return {"results": results}


MSDS_PARAMETERS = {
    "type": "object",
    "properties": {"materials": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_MATERIALS}},
    "required": ["materials"],
}
