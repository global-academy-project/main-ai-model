class FakeTextGenerationClient:
    def __init__(
        self,
        *,
        response_text: str = "테스트 AI 응답",
    ) -> None:
        self._response_text = response_text
        self.last_instructions: str | None = None
        self.last_input_text: str | None = None

    async def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        self.last_instructions = instructions
        self.last_input_text = input_text

        return self._response_text