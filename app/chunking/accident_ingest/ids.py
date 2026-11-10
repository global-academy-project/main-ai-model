"""external_id(D4): SIF 응답에는 사례 id가 없어 재해개요의 해시로 만든다.

    base   = "SIF-" + sha256(정규화한 개요)의 앞 16자(hex)                       (총 20자)
    충돌    = 정규화 개요가 같은 레코드가 둘 이상이면 그 레코드 전부에 "-" + 6자 접미사를 붙인다.
    접미사  = sha256(정규화 개요 + "\\n" + 정규화 유발요인 + "\\n" + 정규화 대책)의 앞 6자(hex)
    정규화  = 모든 공백 제거 + NFC (대책·유발요인이 None이면 빈 문자열)

충돌 검사는 **읽은 레코드 전체**(적재 범위 밖 티어 포함)를 대상으로 한다. 적재 범위를 바꿔도
id가 바뀌지 않게 하려는 것이다. 이 규칙은 적재 후에 바꾸지 않는다(id_stability golden으로 고정).
"""

from __future__ import annotations

import unicodedata
from typing import Any

from .models import sha256_text


def normalize(text: str | None) -> str:
    return unicodedata.normalize("NFC", "".join((text or "").split()))


def base_id(overview: str | None, prefix: str = "SIF-") -> str:
    return prefix + sha256_text(normalize(overview))[:16]


def suffix(overview: str | None, factor: str | None, measures: str | None) -> str:
    return sha256_text("\n".join([normalize(overview), normalize(factor), normalize(measures)]))[:6]


def assign_ids(records: list[dict[str, Any]], prefix: str = "SIF-") -> dict[str, Any]:
    """records의 각 항목에 external_id, id_base, id_suffix(없으면 None)를 채운다. 보고용 요약을 돌려준다."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        r["id_base"] = base_id(r.get("disasterOverview"), prefix)
        groups.setdefault(r["id_base"], []).append(r)
    collisions: list[dict[str, Any]] = []
    for base, rs in groups.items():
        if len(rs) == 1:
            rs[0]["id_suffix"] = None
            rs[0]["external_id"] = base
            continue
        for r in rs:
            r["id_suffix"] = suffix(r.get("disasterOverview"), r.get("disasterFactor"), r.get("dcrsCntrplnCn"))
            r["external_id"] = f"{base}-{r['id_suffix']}"
        collisions.append({"id_base": base, "external_ids": [r["external_id"] for r in rs]})
    ids = [r["external_id"] for r in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    return {"collision_groups": len(collisions), "collisions": collisions, "unresolved_duplicates": dup}
