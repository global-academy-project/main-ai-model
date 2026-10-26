from dataclasses import dataclass
from app.clients.base import TextGenerationClient
from app.schemas.ai_summary import (
    SummaryLength,
    SummaryTone,
)

@dataclass(frozen=True)
class SummaryCommand:
    text: str
    length: SummaryLength
    tone: SummaryTone

@dataclass(frozen=True)
class SummaryResult:
    summary: str
    length: SummaryLength
    tone: SummaryTone

class SummaryService:
    def __init__(
        self,
        *,
        client: TextGenerationClient,
    ) -> None:
        self._client = client
        
    async def summarize(
        self,
        command: SummaryCommand,
    ) -> SummaryResult:
        instructions = self._build_instructions(
            length=command.length,
            tone=command.tone,
        )

        input_text = self._prepare_input(command.text)

        generated_text = (
            await self._client.generate(
                instructions=instructions,
                input_text=input_text,
            )
        )

        summary = self._validate_output(generated_text)

        return SummaryResult(
            summary=summary,
            length=command.length,
            tone=command.tone,
        )

    def _build_instructions(
        self,
        *,
        length: SummaryLength,
        tone: SummaryTone,
    ) -> str:
        length_instruction = {
            "short": "핵심 내용을 3문장 이내로 요약하세요.",
            "medium": "핵심 내용을 5개 문장 안팎으로 요약하세요.",
            "long": (
                "핵심 논리와 중요한 세부사항을 포함하여 "
                "충분히 자세히 요약하세요."
            ),
        }[length]

        tone_instruction = {
            "neutral": "중립적이고 객관적인 문체를 사용하세요.",
            "professional": (
                "업무 보고서에 적합한 전문적인 문체를 사용하세요."
            ),
            "friendly": (
                "비전문가도 이해하기 쉬운 친근한 문체를 사용하세요."
            ),
        }[tone]

        return "\n".join(
            [
                "당신은 문서 요약을 수행하는 AI입니다.",
                length_instruction,
                tone_instruction,
                "원문에 없는 사실을 추가하지 마세요.",
                "요약 결과만 반환하세요.",
            ]
        )

    def _prepare_input(self, text: str) -> str:
        return text.strip()

    def _validate_output(self, output: str) -> str:
        normalized = output.strip()

        if not normalized:
            raise ValueError(
                "생성된 요약 결과가 비어 있습니다."
            )

        return normalized