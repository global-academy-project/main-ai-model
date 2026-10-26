from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image


class OcrEngine(Protocol):
    name: str

    def recognize(
        self, image: "Image", *, lang: list[str]
    ) -> tuple[str, float | None]:
        """(텍스트, 신뢰도 0.0~1.0 또는 None)을 반환. 신뢰도 정규화는 구현체 책임."""
        ...