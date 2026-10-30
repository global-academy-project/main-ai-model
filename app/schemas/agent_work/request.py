from typing import Literal, Annotated
from pydantic import Field
from app.schemas import CamelModel
from app.schemas.forms import MeasurementItem

MAX_UTTERANCES = 10
PositiveId = Annotated[int, Field(strict=True, gt=0)]

class WorkSlots(CamelModel):
    """되묻기 체크리스트 항목. 응답으로 받은 값을 다음 요청에 그대로 다시 보낸다.
    화면에서 직접 채운 값(예: 화재감시자 이름)도 여기에 넣어 보내면 된다."""

    utterances: list[str] = Field(default_factory=list, max_length=MAX_UTTERANCES, description="지금까지 받은 작업책임자 발화")
    location: str | None = None
    hot_work_kind: str | None = None
    nearby_materials: list[str] = Field(default_factory=list)
    nearby_materials_none: bool = Field(default=False, description="'주변 물질 없음'이라고 답했다")
    material_locations: dict[str, str] = Field(default_factory=dict, description="물질 → 보관 위치")
    headcount: int | None = Field(default=None, ge=1, le=500)
    workDate: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    equipment: list[str] = Field(default_factory=list)
    excluded_equipment: list[str] = Field(default_factory=list, description="'크레인 안 씀'처럼 쓰지 않는다고 한 장비")
    equipment_none: bool = False
    measurements: dict[MeasurementItem, float] = Field(default_factory=dict)
    fire_watcher: str | None = None
    attendant: str | None = None
    asked_keys: list[str] = Field(default_factory=list, description="직전에 되물은 항목 key. 다음 발화의 짧은 답('3', '없음')을 읽는 데 쓴다")

class WorkRegisterRequest(CamelModel):
    message: str = Field(min_length=1, max_length=2000, description="이번 작업책임자 발화")
    slots: WorkSlots | None = Field(default=None, description="이전 응답의 slots. 첫 요청이면 비운다")
    mode: Literal["VERIFY", "GUIDE"] = Field(default="VERIFY", description="sites.mode. GUIDE면 쉬운 말 안내를 붙인다")
    run_id: PositiveId | None = None
    site_id: PositiveId | None = None
    permit_id: PositiveId | None = None