from fastapi import Depends, Form
from typing import Annotated
from app.schemas.extract import ExtractOptions
from app.core.config import Settings, Engine, get_settings
from app.exceptions.business_errors import EngineNotAllowedError

SettingsDep = Annotated[Settings, Depends(get_settings)]

def get_engine(
    settings: SettingsDep,
    engine: Annotated[Engine | None, Form()] = None,
) -> Engine:
    """요청의 engine을 검증해 반환한다. 없으면 설정의 기본 엔진을 쓴다."""
    selected = engine or settings.engine
    try:
        settings.check_engine(selected)
    except ValueError:
        raise EngineNotAllowedError(selected_engine=selected)
    return selected

EngineDep = Annotated[Engine, Depends(get_engine)]

def get_extract_options(
    engine: EngineDep,
    extract_images: Annotated[bool, Form(description="이미지 추출 여부")] = False,
    lang: Annotated[list[str], Form(description="선호 언어")] = ["ko", "en"]
) -> ExtractOptions:
    return ExtractOptions(
        engine=engine, 
        extract_images=extract_images,
        lang=lang
    )
