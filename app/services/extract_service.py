import asyncio
import time
from typing import Callable
import openai
from app.exceptions.app_errors import AppError
from app.schemas.extract import (
    ExtractData, ExtractOptions, ImageInfo, PageInfo, PageSource, PdfInput,
)
from app.services.image_utils import image_to_base64_png
from app.services.ocr.base import OcrEngine
from app.services.ocr.policy import ensure_engine_allowed
from app.services.pdf.base import PdfBackend, PdfOpenError
from app.services.summary_service import SummaryCommand, SummaryService

TEXT_LAYER_ENGINES = {"PYMUPDF", "PDFIUM"}
TEXT_LAYER_MIN_CHARS = 20
OCR_DPI = 300
MAX_CONCURRENT_OCR = 2   # 동시 추출 수 제한 (CPU/외부 API 보호)
MAX_SUMMARY_INPUT_CHARS = 20_000

_ocr_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR)

EngineFactory = Callable[[str], OcrEngine]

def choose_engine(
    requested: str, layer_text: str, default_ocr: str, text_layer_name: str
) -> str:
    """AUTO만 페이지마다 달라진다. 지정 엔진은 전체 페이지에 적용."""
    if requested == "AUTO":
        return text_layer_name if len(layer_text) >= TEXT_LAYER_MIN_CHARS else default_ocr
    return requested

def _fit_to_limit(text: str, limit: int) -> str:
    """limit을 넘으면 줄 경계에서 자른다."""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut < limit // 2:          # 줄바꿈이 너무 앞에 있으면 limit에서 그냥 자른다
        cut = limit
    return text[:cut].rstrip()

class ExtractService:
    def __init__(
        self,
        *, 
        settings, 
        pdf_backend: PdfBackend, 
        engine_factory: EngineFactory,
        summary_service: SummaryService
    ) -> None:
        self._settings = settings
        self._pdf = pdf_backend
        self._factory = engine_factory
        self._summary = summary_service
        self._default_ocr = settings.default_ocr_engine

    async def extract(self, pdf: PdfInput, opts: ExtractOptions) -> ExtractData:
        network_mode = self._settings.network_env
        ensure_engine_allowed(opts.engine)
        if opts.engine == "AUTO":
            # AUTO는 스캔 페이지에서 기본 OCR 엔진을 고르므로 그 엔진도 허용되는지 확인
            ensure_engine_allowed(self._default_ocr)
        if self._settings.engine == "OPENAI":
            # 요약이 OpenAI로 나가는 설정이면, 오래 걸리는 OCR을 하기 전에 먼저 막는다
            ensure_engine_allowed(self._settings.engine)

        started = time.perf_counter()

        try:
            async with _ocr_semaphore:
                # OCR은 동기 CPU/IO 작업이라 스레드로 보내 이벤트 루프를 막지 않는다
                pages, images, engine_name = await asyncio.to_thread(
                    self._run, pdf.content, opts
                )
        except PdfOpenError as exc:
            # 상태 코드/코드값은 팀 에러 규칙(§2)에 맞춰 조정
            raise AppError(
                code= "PDF_UNREADABLE", 
                message=str(exc), 
                status_code=422
            ) from exc

        summary = await self._summarize(pages, opts)

        confs = [
            p.confidence for p in pages
            if p.source == PageSource.OCR and p.confidence is not None
        ]
        return ExtractData(
            engine=engine_name,
            network_mode=network_mode,
            page_count=len(pages),
            total_ms=int((time.perf_counter() - started) * 1000),
            avg_confidence=sum(confs) / len(confs) if confs else None,
            pages=pages,
            images=images,
            summary=summary
        )
    
    async def _summarize(self, pages: list[PageInfo], opts: ExtractOptions) -> str:
        text = "\n\n".join(p.text for p in pages if p.text.strip())
        if not text.strip():
            raise AppError(
                422, "NO_TEXT_EXTRACTED", "PDF에서 요약할 텍스트를 찾지 못했습니다."
            )
        try:
            result = await self._summary.summarize(
                SummaryCommand(
                    text=_fit_to_limit(text, MAX_SUMMARY_INPUT_CHARS),
                    length="medium",
                    tone="neutral",
                )
            )

        except openai.OpenAIError as exc:
            # 키나 원문이 응답에 새지 않도록 상세 메시지는 노출하지 않는다
            raise AppError(
                status_code=503, code="SUMMARY_PROVIDER_ERROR", message="요약 요청에 실패했습니다."
            ) from exc
        except (ValueError, RuntimeError) as exc:
            # 빈 요약 결과 등 SummaryService/클라이언트가 던지는 예외
            raise AppError(
                status_code=502, code="SUMMARY_FAILED", message="요약 결과를 생성하지 못했습니다."
            ) from exc
        return result.summary

    def _run(self, content: bytes, opts: ExtractOptions):
        pages: list[PageInfo] = []
        images: list[ImageInfo] = []
        used_ocr: str | None = None

        with self._pdf.open(content) as doc:
            for page_no, page in enumerate(doc.pages(), start=1):  # 페이지 순서대로
                t0 = time.perf_counter()
                layer_text = page.extract_text()
                chosen = choose_engine(
                    opts.engine, layer_text, self._default_ocr, self._pdf.name
                )

                rendered = None
                if chosen in TEXT_LAYER_ENGINES:
                    text, conf, source = layer_text, None, PageSource.TEXT_LAYER
                else:
                    rendered = page.render(dpi=OCR_DPI)
                    text, conf = self._factory(chosen).recognize(
                        rendered, lang=opts.lang
                    )
                    source, used_ocr = PageSource.OCR, chosen

                pages.append(PageInfo(
                    page_no=page_no, source=source, text=text, confidence=conf,
                    processing_ms=int((time.perf_counter() - t0) * 1000),
                ))

                if opts.extract_images:
                    # 가정: 페이지 렌더링 이미지 (명세에 정의 없음, 팀 확인 필요)
                    b64, w, h = image_to_base64_png(rendered or page.render(dpi=OCR_DPI))
                    images.append(ImageInfo(
                        page_no=page_no, format="png",
                        width=w, height=h, data_base64=b64,
                    ))

        # 응답의 engine은 실제 사용된 주 엔진.
        # AUTO에서 OCR이 한 번도 없었으면 텍스트 레이어 백엔드 이름
        if opts.engine == "AUTO":
            engine_name = used_ocr or self._pdf.name
        elif opts.engine in TEXT_LAYER_ENGINES:
            engine_name = self._pdf.name
        else:
            engine_name = opts.engine
        return pages, images, engine_name