"""적재 API 입력 모델과 필드 간 일관성 검증 규칙을 정의한다."""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from app.chunking.accident_ingest.profiles.sif import (
    CATE_ALLOWED as SIF_CATEGORIES,
    TAG_CODES as SIF_TAG_CODES,
)
from app.rag.filters import COLLECTIONS


def _validate_point_id(value: object) -> str | int:
    if isinstance(value, bool):
        raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.") from exc
    raise ValueError("point_id는 UUID 문자열 또는 음이 아닌 정수여야 합니다.")


class IngestChunk(BaseModel):
    point_id: str | int
    text: str = Field(min_length=1, max_length=32768)
    payload: dict[str, Any]

    @field_validator("point_id", mode="before")
    @classmethod
    def validate_point_id(cls, value: object) -> str | int:
        return _validate_point_id(value)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("빈 청크는 적재할 수 없습니다.")
        return value


class SourceCompletion(BaseModel):
    source_id: StrictInt = Field(gt=0)
    chunk_count: StrictInt = Field(gt=0)


class IngestRequest(BaseModel):
    """장부 ID가 준비된 청크를 검증하고 선택적으로 완료 처리한다."""

    collection: Literal["kb_accidents", "kb_guides"]
    chunks: list[IngestChunk] = Field(min_length=1, max_length=32)
    finalize_sources: list[SourceCompletion] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_payloads(self) -> IngestRequest:
        required_fields = {"kb_source_id", "kb_chunk_id", "source_type", "site_id", "flagged"}
        source_types: dict[int, str] = {}
        chunk_ids: set[int] = set()
        for chunk in self.chunks:
            payload = chunk.payload
            if not required_fields <= payload.keys():
                raise ValueError("payload에 RAG 필수 필드가 없습니다.")
            if (
                isinstance(payload["kb_source_id"], bool)
                or not isinstance(payload["kb_source_id"], int)
                or payload["kb_source_id"] <= 0
                or isinstance(payload["kb_chunk_id"], bool)
                or not isinstance(payload["kb_chunk_id"], int)
                or payload["kb_chunk_id"] <= 0
                or isinstance(payload["site_id"], bool)
                or not isinstance(payload["site_id"], int)
                or payload["site_id"] < 0
                or not isinstance(payload["flagged"], bool)
                or payload["source_type"] not in COLLECTIONS[self.collection]
            ):
                raise ValueError("payload 값이 collection의 RAG 스키마와 맞지 않습니다.")
            if payload["source_type"] != "NEAR_MISS" and (
                not isinstance(payload.get("text"), str) or not payload["text"].strip()
            ):
                raise ValueError("공공 지식 payload에 원문 text가 필요합니다.")
            if payload["kb_chunk_id"] in chunk_ids:
                raise ValueError("요청에 중복된 kb_chunk_id가 있습니다.")
            chunk_ids.add(payload["kb_chunk_id"])
            source_types[payload["kb_source_id"]] = payload["source_type"]
            if payload["source_type"] == "NEAR_MISS":
                self._validate_near_miss(chunk)
        completion_ids = [source.source_id for source in self.finalize_sources]
        if len(set(completion_ids)) != len(completion_ids):
            raise ValueError("finalize_sources에 중복된 kb_source_id가 있습니다.")
        for source_id in self.finalize_sources:
            source_type = source_types.get(source_id.source_id)
            if source_type is None:
                raise ValueError("완료 처리할 출처는 요청 청크에 포함되어야 합니다.")
            if source_type == "NEAR_MISS":
                raise ValueError("NEAR_MISS는 agent_kb 완료 처리 대상이 아닙니다.")
        return self


    @staticmethod
    def _validate_near_miss(chunk: IngestChunk) -> None:
        payload = chunk.payload
        required_fields = {
            "severity",
            "accident_type",
            "hazard",
            "work_type",
            "cas",
            "ghs",
            "year",
            "occurred_at",
            "zone_id",
            "title",
            "text",
        }
        if not required_fields <= payload.keys():
            raise ValueError("NEAR_MISS payload에 필수 분류·원문 필드가 없습니다.")

        source_id = payload["kb_source_id"]
        if source_id <= 0 or payload["kb_chunk_id"] != source_id:
            raise ValueError("NEAR_MISS의 kb_source_id와 kb_chunk_id는 같은 양의 업무 ID여야 합니다.")
        expected_point_id = str(uuid5(NAMESPACE_URL, f"NEAR_MISS:{source_id}:0"))
        if chunk.point_id != expected_point_id:
            raise ValueError("NEAR_MISS point_id가 UUIDv5 재적재 규칙과 맞지 않습니다.")
        if payload["site_id"] <= 0 or payload["flagged"] is not False:
            raise ValueError("NEAR_MISS는 양수 site_id와 flagged=false가 필요합니다.")

        if not isinstance(payload["severity"], str) or not payload["severity"].strip():
            raise ValueError("NEAR_MISS severity는 비어 있지 않은 문자열이어야 합니다.")
        for field in ("accident_type", "hazard", "work_type", "cas", "ghs"):
            values = payload[field]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"NEAR_MISS {field}는 문자열 배열이어야 합니다.")
        if payload["ghs"]:
            raise ValueError("현재 NEAR_MISS 계약에서 ghs는 빈 배열이어야 합니다.")

        occurred_at = payload["occurred_at"]
        if not isinstance(occurred_at, str):
            raise ValueError("NEAR_MISS occurred_at은 ISO 형식 문자열이어야 합니다.")
        try:
            occurred_year = datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).year
        except ValueError as exc:
            raise ValueError("NEAR_MISS occurred_at은 ISO 형식이어야 합니다.") from exc
        year = payload["year"]
        if isinstance(year, bool) or not isinstance(year, int) or year != occurred_year:
            raise ValueError("NEAR_MISS year는 occurred_at의 연도와 같은 정수여야 합니다.")

        zone_id = payload["zone_id"]
        if zone_id is not None and (
            isinstance(zone_id, bool) or not isinstance(zone_id, int) or zone_id < 0
        ):
            raise ValueError("NEAR_MISS zone_id는 음이 아닌 정수 또는 null이어야 합니다.")
        if not isinstance(payload["title"], str) or not payload["title"].strip():
            raise ValueError("NEAR_MISS title은 비어 있지 않은 문자열이어야 합니다.")
        if payload["text"] != chunk.text:
            raise ValueError("NEAR_MISS payload.text와 청크 text는 같아야 합니다.")


class LocalVectorSaveRequest(IngestRequest):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    chunks: list[IngestChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_unique_point_ids(self) -> LocalVectorSaveRequest:
        point_ids = [chunk.point_id for chunk in self.chunks]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("요청에 중복된 point_id가 있습니다.")
        return self


class LocalChunkRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    natural_key: dict[str, Any]
    text: str = Field(min_length=1, max_length=32768)
    embed_text: str = Field(min_length=1, max_length=32768)
    collection: Literal["kb_accidents", "kb_guides"] = "kb_guides"

    @model_validator(mode="after")
    def validate_chunk(self) -> LocalChunkRecord:
        key = self.natural_key
        if self.collection == "kb_guides":
            part = key.get("part")
            if (
                not isinstance(key.get("law_key"), str)
                or not key["law_key"].strip()
                or not isinstance(key.get("article_key"), str)
                or not key["article_key"].strip()
                or isinstance(part, bool)
                or not isinstance(part, int)
                or part < 0
            ):
                raise ValueError("법령 natural_key에 law_key, article_key, part가 필요합니다.")
        else:
            extra = self.model_extra or {}
            external_id = extra.get("external_id")
            chunk_index = extra.get("chunk_index")
            if (
                key.get("dataset") != "sif"
                or not isinstance(key.get("record"), str)
                or not key["record"].strip()
                or not isinstance(key.get("section"), str)
                or not key["section"].strip()
                or isinstance(key.get("part"), bool)
                or not isinstance(key.get("part"), int)
                or key["part"] < 0
                or not isinstance(external_id, str)
                or external_id != key.get("record")
                or isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
                or extra.get("source_type") != "SIF"
            ):
                raise ValueError("SIF 청크 natural_key, external_id, chunk_index, source_type가 올바르지 않습니다.")
        if not self.text.strip() or not self.embed_text.strip():
            raise ValueError("청크 text와 embed_text는 비어 있을 수 없습니다.")
        return self


class LocalChunkFileRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    chunk_file: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jsonl$")
    data_type: Literal["accidents", "guides"] | None = None
    subdir: str = Field(default="law", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    validation_file: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.json$")


class LocalVectorLoadRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    collection: Literal["kb_accidents", "kb_guides"]


class LocalQdrantPoint(BaseModel):
    id: str | int
    vector: dict[str, Any]
    payload: dict[str, Any]

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: object) -> str | int:
        return _validate_point_id(value)

    @model_validator(mode="after")
    def validate_vector(self) -> LocalQdrantPoint:
        dense = self.vector.get("dense")
        if (
            not isinstance(dense, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in dense
            )
        ):
            raise ValueError("dense 벡터 형식이 올바르지 않습니다.")
        sparse = self.vector.get("sparse")
        if sparse is not None:
            if not isinstance(sparse, dict):
                raise ValueError("sparse 벡터 형식이 올바르지 않습니다.")
            indices, values = sparse.get("indices"), sparse.get("values")
            if (
                not isinstance(indices, list)
                or not isinstance(values, list)
                or len(indices) != len(values)
                or any(type(index) is not int or index < 0 for index in indices)
                or indices != sorted(set(indices))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError("sparse 벡터 형식이 올바르지 않습니다.")
        return self


class DeletePointsRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    point_ids: list[str | int] = Field(min_length=1, max_length=32)

    @field_validator("point_ids", mode="before")
    @classmethod
    def validate_point_ids(cls, value: object) -> list[str | int]:
        if not isinstance(value, list):
            raise ValueError("point_ids는 UUID 문자열 또는 음이 아닌 정수의 배열이어야 합니다.")
        return [_validate_point_id(point_id) for point_id in value]


class PublicSourceMetadata(BaseModel):
    external_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    url: str | None = Field(default=None, max_length=500)
    license: str | None = Field(default=None, max_length=50)
    publisher: str | None = Field(default=None, max_length=100)
    published_on: date | None = None
    file_path: str | None = Field(default=None, max_length=500)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    flagged: bool = False

    @model_validator(mode="after")
    def validate_catalog_fields(self) -> PublicSource:
        if not self.external_id.strip() or not self.title.strip():
            raise ValueError("external_id와 title은 공백일 수 없습니다.")
        if not (self.url and self.url.strip()) and not (self.file_path and self.file_path.strip()):
            raise ValueError("공공 source에는 url 또는 file_path가 필요합니다.")
        return self


class PublicSource(PublicSourceMetadata):
    source_type: Literal[
        "ACCIDENT_REPORT", "SIF", "KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"
    ]


class PublicSourceChunkBase(BaseModel):
    chunk_index: StrictInt = Field(ge=0)
    text: str = Field(min_length=1, max_length=32768)
    embedding_text: str | None = Field(default=None, min_length=1, max_length=32768)
    locator: str | None = Field(default=None, max_length=100)
    token_count: StrictInt | None = Field(default=None, gt=0)

    @field_validator("text", "embedding_text")
    @classmethod
    def validate_nonblank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("청크 본문은 공백일 수 없습니다.")
        return value


class PublicSourceChunk(PublicSourceChunkBase):
    payload: dict[str, Any] = Field(default_factory=dict)


class PublicAccidentSource(PublicSourceMetadata):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["ACCIDENT_REPORT", "SIF"]


class PublicGuideSource(PublicSourceMetadata):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"]


class SifNaturalKey(BaseModel):
    dataset: Literal["sif"]
    record: str = Field(min_length=1, max_length=100)
    section: str = Field(min_length=1, max_length=100)
    part: StrictInt = Field(ge=0)


class LawNaturalKey(BaseModel):
    law_key: str = Field(min_length=1, max_length=100)
    article_key: str = Field(min_length=1, max_length=100)
    part: StrictInt = Field(ge=0)


class LawMetadata(BaseModel):
    effective_date: str
    promulgation_no: str
    promulgation_date: str
    amendment_type: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicAccidentTags(BaseModel):
    """사고 컬렉션의 검색 태그와 기존 SIF payload 전용 필드를 검증한다."""

    model_config = ConfigDict(extra="forbid")

    severity: str | None = None
    accident_type: list[str] = Field(default_factory=list)
    hazard: list[str] = Field(default_factory=list)
    work_type: list[str] = Field(default_factory=list)
    cas: list[str] = Field(default_factory=list)
    ghs: list[str] = Field(default_factory=list)
    year: StrictInt | None = None
    natural_key: SifNaturalKey | None = None
    tier: Literal["A", "B", "C", "X"] | None = None
    cate: str | None = None
    disaster_type: str | None = None
    situation: str | None = None

    @model_validator(mode="after")
    def validate_tag_codes(self) -> PublicAccidentTags:
        if self.severity is not None and self.severity not in SIF_TAG_CODES["SEVERITY"]:
            raise ValueError("severity가 기존 사고 태그 코드에 없습니다.")
        for field, axis in (
            ("accident_type", "ACCIDENT_TYPE"),
            ("hazard", "HAZARD"),
            ("work_type", "WORK_TYPE"),
        ):
            values = getattr(self, field)
            if any(value not in SIF_TAG_CODES[axis] for value in values):
                raise ValueError(f"{field}에 기존 태그 코드가 아닌 값이 있습니다.")
        return self


class PublicGuideTags(BaseModel):
    """지침 태그와 기존 법령 청크 payload 필드를 검증한다."""

    model_config = ConfigDict(extra="forbid")

    hazard: list[str] = Field(default_factory=list)
    work_type: list[str] = Field(default_factory=list)
    cas: list[str] = Field(default_factory=list)
    year: StrictInt | None = None
    guide_no: str | None = None
    natural_key: LawNaturalKey | None = None
    article_label: str | None = None
    article_title: str | None = None
    parts_total: StrictInt | None = Field(default=None, gt=0)
    repeated_lead_lines: StrictInt | None = Field(default=None, ge=0)
    oversize: bool | None = None
    path: list[str] | None = None
    hierarchy: dict[str, str] | None = None
    law_meta: LawMetadata | None = None
    article_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_tag_codes(self) -> PublicGuideTags:
        for field, axis in (("hazard", "HAZARD"), ("work_type", "WORK_TYPE")):
            values = getattr(self, field)
            if any(value not in SIF_TAG_CODES[axis] for value in values):
                raise ValueError(f"{field}에 기존 태그 코드가 아닌 값이 있습니다.")
        return self


class PublicAccidentChunk(PublicSourceChunkBase):
    model_config = ConfigDict(extra="forbid")

    payload: PublicAccidentTags = Field(default_factory=PublicAccidentTags)


class PublicGuideChunk(PublicSourceChunkBase):
    model_config = ConfigDict(extra="forbid")

    payload: PublicGuideTags = Field(default_factory=PublicGuideTags)


class PublicSourceIngestRequest(BaseModel):
    collection: Literal["kb_accidents", "kb_guides"]
    source: PublicSource
    chunks: list[PublicSourceChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_source_collection(self) -> PublicSourceIngestRequest:
        if self.source.source_type not in COLLECTIONS[self.collection]:
            raise ValueError("source_type이 collection과 맞지 않습니다.")
        indexes = [chunk.chunk_index for chunk in self.chunks]
        if len(set(indexes)) != len(indexes):
            raise ValueError("요청에 중복된 chunk_index가 있습니다.")
        return self


class PublicAccidentIngestRequest(BaseModel):
    """사고 전용 계약으로 요청을 검증하며 컬렉션은 라우트에서 고정한다."""

    model_config = ConfigDict(extra="forbid")

    source: PublicAccidentSource
    chunks: list[PublicAccidentChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_chunk_indexes(self) -> PublicAccidentIngestRequest:
        indexes = [chunk.chunk_index for chunk in self.chunks]
        if len(set(indexes)) != len(indexes):
            raise ValueError("요청에 중복된 chunk_index가 있습니다.")
        if self.source.source_type == "SIF":
            required_sif_fields = {
                "natural_key",
                "tier",
                "cate",
                "disaster_type",
                "situation",
            }
            for chunk in self.chunks:
                missing = required_sif_fields - chunk.payload.model_fields_set
                if missing:
                    raise ValueError(
                        f"SIF payload에 기존 필드가 없습니다: {', '.join(sorted(missing))}"
                    )
                natural_key = chunk.payload.natural_key
                if (
                    natural_key is None
                    or natural_key.record != self.source.external_id
                    or chunk.payload.tier is None
                    or chunk.payload.cate not in SIF_CATEGORIES
                ):
                    raise ValueError("SIF natural_key 또는 cate가 source와 맞지 않습니다.")
        return self


class PublicGuideIngestRequest(BaseModel):
    """지침·법령 전용 계약으로 요청을 검증하며 컬렉션은 라우트에서 고정한다."""

    model_config = ConfigDict(extra="forbid")

    source: PublicGuideSource
    chunks: list[PublicGuideChunk] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_chunk_indexes(self) -> PublicGuideIngestRequest:
        indexes = [chunk.chunk_index for chunk in self.chunks]
        if len(set(indexes)) != len(indexes):
            raise ValueError("요청에 중복된 chunk_index가 있습니다.")
        if self.source.source_type == "LAW":
            required_law_fields = {
                "natural_key",
                "article_label",
                "article_title",
                "parts_total",
                "repeated_lead_lines",
                "oversize",
                "path",
                "hierarchy",
                "hazard",
                "work_type",
                "cas",
                "year",
                "law_meta",
                "article_sha256",
            }
            source_hashes: set[str] = set()
            for chunk in self.chunks:
                missing = required_law_fields - chunk.payload.model_fields_set
                if missing:
                    raise ValueError(
                        f"LAW payload에 기존 필드가 없습니다: {', '.join(sorted(missing))}"
                    )
                natural_key = chunk.payload.natural_key
                if natural_key is None or natural_key.law_key != self.source.external_id:
                    raise ValueError("LAW natural_key.law_key와 source.external_id가 다릅니다.")
                if chunk.payload.law_meta is not None:
                    source_hashes.add(chunk.payload.law_meta.source_sha256)
            if len(source_hashes) != 1:
                raise ValueError("LAW 청크의 law_meta.source_sha256가 하나의 원본으로 일치해야 합니다.")
            if self.source.content_sha256 not in {None, next(iter(source_hashes))}:
                raise ValueError("LAW source.content_sha256와 law_meta.source_sha256가 다릅니다.")
        return self
