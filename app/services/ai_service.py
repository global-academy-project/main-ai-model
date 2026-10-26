from typing import Protocol

class AIServiceProtocol(Protocol):
    """PDF 요약을 수행하는 AI 서비스의 인터페이스.

    실제 AI 모델 연동 구현체와 테스트용 가짜 구현체가 이 규약을 따릅니다.
    """
    async def summarize(self, content: bytes) -> str:
        """PDF 바이트 내용을 요약한 텍스트를 반환합니다.

        Args:
            content: 검증을 통과한 PDF 파일의 바이트 내용.

        Returns:
            요약된 텍스트.
        """
        ...

class FakeAIService:
    """실제 요약 없이 고정된 문자열을 반환하는 테스트용 구현체.

    AI 모델 연동 전 API 구조와 업로드 흐름을 확인하는 용도입니다.
    입력 내용과 관계없이 항상 같은 값을 반환합니다.
    """
    async def summarize(self, content: bytes) -> str:
        """입력과 무관하게 고정된 문자열을 반환합니다.

        Args:
            content: 사용하지 않습니다.

        Returns:
            고정된 테스트용 문자열.
        """
        return "This file's type is PDF"