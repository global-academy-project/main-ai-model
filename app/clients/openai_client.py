from openai import AsyncOpenAI

class OpenAITextGenerationClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens

        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def generate(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        response = await self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=self._max_output_tokens,
        )

        output_text = response.output_text.strip()

        if not output_text:
            raise RuntimeError(
                "AI Provider 응답이 비어 있습니다."
            )

        return output_text