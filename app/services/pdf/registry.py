from functools import lru_cache
from app.schemas.error import ErrorDetail
from app.exceptions.app_errors import AppError
from app.services.pdf.base import PdfBackend

@lru_cache(maxsize=None)
def create_pdf_backend(name: str) -> PdfBackend:
    if name == "PDFIUM":
        from app.services.pdf.pdfium_backend import PdfiumBackend
        return PdfiumBackend()
    
    raise AppError(
        code="PDF_BACKEND_NOT_SUPPORTED", 
        message="지원하지 않는 PDF 백엔드입니다", 
        status_code=500, 
        details=[ErrorDetail(
            filedfield=None, 
            reason=f"지원하지 않는 PDF 백엔드: {name}")
            ]
        )