"""단계 사이에서 주고받는 데이터 모양.

모든 모델은 JSON으로 저장하고 다시 읽을 수 있다(`to_dict` / `from_dict`).
읽기 단계(S1)만 파일 형식(hwpx, 나중에 xml)을 알고, 그 뒤 단계는 이 모델만 본다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# 노드 종류와 깊이. lead는 조 본문 첫 문장(항 번호 없음).
NODE_RANK = {"lead": 0, "항": 1, "호": 2, "목": 3}


def nonspace(text: str) -> int:
    """공백이 아닌 글자 수. 글자 보존 검사에 쓴다(공백 정리는 글자 손실이 아니다)."""
    return sum(1 for ch in text if not ch.isspace())


# --- S1 읽기 결과 -------------------------------------------------------------------


@dataclass
class RawParagraph:
    index: int
    text: str  # 앞쪽 공백을 지우지 않는다. 문단 안 줄바꿈은 '\n'

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawParagraph":
        return cls(index=d["index"], text=d["text"])


@dataclass
class SourceMeta:
    law_name: str
    law_alias: str | None = None
    effective_date: str | None = None  # YYYY-MM-DD
    promulgation_no: str | None = None  # 예: "고용노동부령 제450호"
    promulgation_date: str | None = None  # YYYY-MM-DD
    amendment_type: str | None = None  # 예: "일부개정"
    source_file: str = ""
    source_format: str = ""
    source_sha256: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceMeta":
        return cls(**d)


@dataclass
class SourceDoc:
    meta: SourceMeta
    paragraphs: list[RawParagraph]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"stage": "source", **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceDoc":
        return cls(
            meta=SourceMeta.from_dict(d["meta"]),
            paragraphs=[RawParagraph.from_dict(p) for p in d["paragraphs"]],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# --- S2/S3 구조 ---------------------------------------------------------------------


@dataclass
class Node:
    kind: str  # lead | 항 | 호 | 목
    marker: str  # "" | "①" | "1." | "가."
    text: str
    lines: list[str] = field(default_factory=list)  # 번호 없는 이어지는 줄
    children: list["Node"] = field(default_factory=list)
    deleted: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            kind=d["kind"],
            marker=d["marker"],
            text=d["text"],
            lines=list(d.get("lines", [])),
            children=[cls.from_dict(c) for c in d.get("children", [])],
            deleted=d.get("deleted", False),
        )


@dataclass
class Heading:
    level: str  # config.HEADING_LEVELS 중 하나(편 | 장 | 절 | 관 | 속)
    number: str  # "10", "3의2"
    title: str
    raw: str
    src: int
    label: str = ""  # S3에서 채움: "제10장 밀폐공간 작업으로 …"
    deleted: bool = False
    amend_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Heading":
        return cls(**d)


@dataclass
class Article:
    order: int
    key: str  # "241-2"
    label: str  # "제241조의2"
    title: str | None
    status: str  # ACTIVE | DELETED | EMPTY
    path_idx: list[int]  # law.headings 안의 위치(편→관 순)
    inline_first: bool  # 조 줄 안에 첫 항/본문이 붙어 있었는지
    nodes: list[Node]
    notes: list[str]  # [본조신설 …] 같은 별도 줄
    amend_tags: list[str]  # 본문에서 떼어 낸 <개정 …> 표시
    raw_lines: list[str]  # 원본 줄(빈 줄 제외)
    src_start: int
    src_end: int
    ledger: dict[str, int] = field(default_factory=dict)  # S3에서 채움

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Article":
        d = dict(d)
        d["nodes"] = [Node.from_dict(n) for n in d["nodes"]]
        return cls(**d)


@dataclass
class ParsedLaw:
    meta: SourceMeta
    stage: str  # parsed | cleaned
    headings: list[Heading]
    articles: list[Article]
    prelude: list[str]
    addenda: list[str]
    body_src_start: int
    body_src_end: int
    addenda_src_start: int | None
    orphans: list[str] = field(default_factory=list)  # 어느 조에도 속하지 못한 줄
    warnings: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParsedLaw":
        return cls(
            meta=SourceMeta.from_dict(d["meta"]),
            stage=d["stage"],
            headings=[Heading.from_dict(h) for h in d["headings"]],
            articles=[Article.from_dict(a) for a in d["articles"]],
            prelude=list(d["prelude"]),
            addenda=list(d["addenda"]),
            body_src_start=d["body_src_start"],
            body_src_end=d["body_src_end"],
            addenda_src_start=d.get("addenda_src_start"),
            orphans=list(d.get("orphans", [])),
            warnings=list(d.get("warnings", [])),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# --- S4 청킹 결과 -------------------------------------------------------------------


@dataclass
class Piece:
    article_key: str
    part: int
    parts_total: int
    lines: list[str]
    repeated_lead_lines: int  # 앞의 몇 줄이 이전 조각에서 되풀이한 도입부인지
    span: str | None  # "제1항~제2항" 등. 없으면 None
    oversize: bool = False  # 상한을 넘지만 더 못 나눈 경우

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Piece":
        return cls(**d)
