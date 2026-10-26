import base64
import io

def image_to_base64_png(image) -> tuple[str, int, int]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), image.width, image.height