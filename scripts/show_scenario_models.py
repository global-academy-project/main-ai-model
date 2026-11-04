"""시나리오별로 실제 쓰일 엔진·모델을 보여 준다. 모델은 내려받거나 메모리에 올리지 않는다.

    python scripts/show_scenario_models.py

app/core/config.py의 SCENARIO_MODELS 표(.env에 SCENARIO_ENGINES가 있으면 그 값), NETWORK_ENV,
AI_MAX_OUTPUT_TOKENS, 각 시나리오 모듈(app/llm/engines/llm_<키>.py)의 SPEC을 합친 결과다.
팀원과 결과가 다르면 이 출력부터 비교한다. 한 키의 설정이 틀려도 나머지 키는 계속 보여 준다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import get_settings  # noqa: E402
from app.core.scenario_keys import DEFAULT_KEY, MODEL_KEYS, SCENARIO_CODES  # noqa: E402
from app.exceptions import LLMError  # noqa: E402
from app.llm.engine_names import MODEL_ID_ENGINES  # noqa: E402
from app.llm.llm_factory import build_llm_client  # noqa: E402
from app.llm.llm_provider import LLMClientProvider  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    settings = get_settings()
    provider = LLMClientProvider.from_settings(settings)
    print(f"NETWORK_ENV={settings.network_env}")
    print(f"{'키':<20} {'계획서':<4} {'엔진':<20} {'모델':<32} {'장치':<6} {'토큰':<5} {'로컬만':<6} 클래스")
    for key in MODEL_KEYS:
        if key == DEFAULT_KEY:
            continue
        binding = provider.binding_for(key)
        code = SCENARIO_CODES.get(key) or "—"
        if binding.engine in MODEL_ID_ENGINES:
            try:
                client = build_llm_client(settings=settings, selected_engine=binding.engine, model_id=binding.model_id)
            except LLMError as exc:  # 이 키만 설정이 틀렸다. 이유를 보여 주고 다음 키로
                print(f"{key:<20} {code:<4} {binding.engine:<20} 오류: {exc}")
                continue
            cfg = client.config
            print(f"{key:<20} {code:<4} {binding.engine:<20} {cfg.model_id:<32} {cfg.device:<6} "
                  f"{cfg.max_new_tokens:<5} {str(cfg.local_files_only):<6} {type(client).__name__}")
        else:
            print(f"{key:<20} {code:<4} {binding.engine:<20} {binding.model_id or '-':<32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
