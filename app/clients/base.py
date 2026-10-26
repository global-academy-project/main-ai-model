from typing import Protocol

class TextGenerationClient(Protocol):
    async def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        ...