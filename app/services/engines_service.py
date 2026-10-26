import logging
from app.schemas.engines import EnginesData, EngineInfo
from app.core.config import Settings
from app.core.config import EXTERNAL_ENGINES

logger = logging.getLogger(__name__)

class EnginesService:
    """가용한 엔진 목록 조회 요청을 처리하는 서비스"""
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_engines(self) -> EnginesData:
        """가용 엔진 목록이 포함된 결과를 반환합니다.
        
        Returns:
            현재 네트워크 모드와 가용 엔진 목록을 담은 DataResponse.
        """
        engines = [self._build_info(name) for name 
                   in self._settings.available_engines]
        logger.info("engines 조회 완료")
        
        return EnginesData(
            network_mode=self._settings.network_env,
            default_engine=self._settings.engine,
            engines=engines,
        )
    
    def _build_info(self, name: str) -> EngineInfo:
        external = name in EXTERNAL_ENGINES
        reason = self._unavailable_reason(name, external)
        return EngineInfo(
            name=name,
            available=reason is None,
            external=external,
            reason=reason,
        )

    def _unavailable_reason(self, name: str, external: bool) -> str | None:
        """사용할 수 없는 사유를 반환하고, 사용 가능하면 None을 반환한다."""
        s = self._settings
        if external and s.network_env == "CLOSED":
            return "폐쇄망 모드에서는 사용할 수 없습니다."
        if name == "OPENAI":
            if s.openai_api_key is None:
                return "OPENAI_API_KEY가 설정되지 않았습니다."
            if not s.openai_model:
                return "OPENAI_MODEL이 설정되지 않았습니다."
        return None