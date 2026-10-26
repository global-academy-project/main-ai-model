from fastapi import APIRouter, Depends, status
from typing import Annotated
from app.schemas.base import DataResponse
from app.schemas.engines import EnginesData
from app.schemas.extract import(
    PdfInput, ExtractData, ExtractOptions
)
from app.core.dependecies.pdf_dep import get_pdf_input
from app.core.dependecies.engine_dep import get_extract_options
from app.core.dependecies.dependencies import(
    get_engines_service,
    get_summary_service,
)
from app.services.engines_service import EnginesService
from app.services.summary_service import SummaryService
from app.core.dependecies.dependencies import ExtractServiceDep
from app.core.dependecies.client_dep import get_summary_service

router = APIRouter(prefix="/ocr", tags=["OCR"])
EnginesServiceDep = Annotated[EnginesService, Depends(get_engines_service)]
PdfDep = Annotated[PdfInput, Depends(get_pdf_input)]
SummaryServiceDep = Annotated[SummaryService, Depends(get_summary_service)]
ExtractOptionsDep = Annotated[ExtractOptions, Depends(get_extract_options)]

@router.get(
    "/engines",
    response_model=DataResponse[EnginesData],
    status_code=status.HTTP_200_OK,    
)
def get_engines(service: EnginesServiceDep) -> DataResponse[EnginesData]:
    """가용한 엔진 목록을 반환합니다."""
    engines_data = service.get_engines()
    return DataResponse(data=engines_data)

@router.post(
    "/extract",
    response_model=DataResponse[ExtractData],
    status_code=status.HTTP_200_OK,
)
async def extract_pdf(
    valid_pdf: PdfDep,
    options: ExtractOptionsDep,
    service: ExtractServiceDep
    ) -> DataResponse[ExtractData]:
    """업로드된 PDF 파일에서 텍스트를 추출한 뒤 요약 결과를 반환합니다."""
    extrart_data = await service.extract(valid_pdf, options)
    return DataResponse(data=extrart_data)