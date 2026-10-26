from functools import lru_cache
from fastapi import Depends
from typing import Annotated
from app.core.config import Settings, get_settings
from app.services.engines_service import EnginesService
from app.services.summary_service import SummaryService
from app.services.pdf.base import PdfBackend
from app.services.pdf.registry import create_pdf_backend
from app.services.extract_service import ExtractService
from app.services.ocr.registry import OcrEngineFactory
from app.services.summary_service import SummaryService
from app.core.dependecies.client_dep import get_summary_service

SettingsDep = Annotated[Settings, Depends(get_settings)]
        
def get_engines_service(
    settings: SettingsDep,
) -> EnginesService:
    return EnginesService(settings)

def get_pdf_backend(settings: SettingsDep) -> PdfBackend:
    return create_pdf_backend(settings.pdf_backend)

PdfBackendDep = Annotated[PdfBackend, Depends(get_pdf_backend)]

@lru_cache
def _ocr_engine_factory() -> OcrEngineFactory:
    # 요청마다 새로 만들면 캐시가 의미 없으므로 프로세스당 하나만 둔다
    return OcrEngineFactory(get_settings())

def get_ocr_engine_factory() -> OcrEngineFactory:
    return _ocr_engine_factory()

OcrEngineFactoryDep = Annotated[OcrEngineFactory, Depends(get_ocr_engine_factory)]

def get_extract_service(
    settings: SettingsDep,
    pdf_backend: PdfBackendDep,
    engine_factory: OcrEngineFactoryDep,
    summary_service: Annotated[SummaryService, Depends(get_summary_service)],
) -> ExtractService:
    return ExtractService(
        settings=settings,
        pdf_backend=pdf_backend,
        engine_factory=engine_factory,
        summary_service=summary_service,
    )

ExtractServiceDep = Annotated[ExtractService, Depends(get_extract_service)]

