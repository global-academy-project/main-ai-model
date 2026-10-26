"""테스트용 PDF 생성 도구. 외부 파일 없이 메모리에서 PDF 를 만든다."""
from io import BytesIO

from pypdf import PdfWriter

TEXT_LAYER_SENTENCE = "This page has a real text layer."   # 20자 이상


def blank_pdf(pages: int) -> bytes:
    """텍스트 레이어가 없는 빈 페이지 n 장짜리 PDF"""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def text_pdf(text: str = TEXT_LAYER_SENTENCE) -> bytes:
    """텍스트 레이어가 있는 1페이지 PDF (기본 폰트 Helvetica)"""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return bytes(out)


def make_settings(**overrides):
    """.env 와 무관하게 Settings 를 만든다. 필요한 값만 overrides 로 바꾼다."""
    from app.core.config import Settings

    values = dict(
        app_name="test", app_version="0.0.0", app_env="test",
        network_env="OPEN", engine="OPENAI", default_ocr_engine="TESSERACT",
        openai_api_key="sk-test-not-a-real-key", openai_model="test-model",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)
