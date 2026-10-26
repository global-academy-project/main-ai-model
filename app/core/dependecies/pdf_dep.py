from fastapi import File, UploadFile
from io import BytesIO
from pypdf import PdfReader
from app.schemas.extract import PdfInput
from app.exceptions.business_errors import(
    FileTooLargeError, 
    UnsupportedFileTypeError,
    PageLimitExceededError)
from app.schemas.extract import PdfInput

MAX_FILE_SIZE_MB = 20
MAX_PDF_PAGE_NUM = 50

async def get_pdf_input(
    file: UploadFile = File(
        ..., 
        description = f"요약할 PDF 파일. 최대 용량 {MAX_FILE_SIZE_MB}MB")    
) -> PdfInput:
    """업로드된 파일이 올바른 PDF인지 검증하고 PdfInput으로 변환합니다.
    파일 크기, PDF 시그니처(%PDF),PDF 페이지 수 순으로 검사합니다.
    
    Args:
        file: multipart/form-data로 업로드된 파일.

    Returns:
        검증을 통과한 파일명과 바이트 내용을 담은 PdfInput.
        파일명이 없으면 "unknownFileName"으로 대체됩니다.

    Raises:
        검증 실패 시 아래 상태 코드로 발생합니다.
            - FileTooLargeError: 파일 제한 크기를 초과한 경우
            - UnsupportedFileTypeError: 파일이 %PDF 시그니처로 시작하지 않는 경우
            - PageLimitExceededError: PDF 페이지 수 제한을 초과한 경우
    """
    
    # MAX_FILE_SIZE_MB를 실제 MB로 변환
    max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read(max_file_size + 1)
    
    if len(content) > max_file_size:
        raise FileTooLargeError(
            max_size=MAX_FILE_SIZE_MB
        )
    if not content.startswith(b"%PDF"):
        raise UnsupportedFileTypeError(
            file_type="PDF"
        )
    reader = PdfReader(BytesIO(content))
    
    if len(reader.pages) > MAX_PDF_PAGE_NUM :
        raise PageLimitExceededError(
            page_limit=MAX_PDF_PAGE_NUM
        )
    
    return PdfInput(
        filename=file.filename or "unknownFileName",
        content=content,
        pages=len(reader.pages)
    )
