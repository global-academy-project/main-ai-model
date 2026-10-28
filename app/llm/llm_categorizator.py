"""CATEGORIZATOR 엔진: transformers로 받은 로컬 모델로 `LLMClient`를 구현한다.

- 모델은 처음 `chat()`을 부를 때 한 번만 읽고(지연 로딩) 프로세스 안에서 공유한다.
- `transformers`·`torch`는 지연 임포트한다. 설치돼 있지 않아도 앱은 뜬다.
- 도구 호출(tool use)은 지원하지 않는다. 1주차 루프는 도구를 코드가 부르고,
  LLM은 JSON 텍스트만 돌려주면 된다.
- 오류 정책(지침 4장): `LLMConfigError`에는 직접 쓴 고정 문구만 넘기고,
  transformers가 낸 오류 원문은 로그에만 남긴다. 호출 실패는 `LLMError`(502).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from app.exceptions import LLMConfigError, LLMError

from .llm_base import LLMClient
from .llm_types import LLMResponse, Message, ToolSpec, Usage

logger = logging.getLogger(__name__)

ENGINE_NAME = "CATEGORIZATOR"
DEFAULT_MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class CategorizatorConfig:
    """모델을 어디서 어떻게 읽을지. 해시 가능해서 로딩 캐시 키로 쓴다."""

    model_id: str  # 허브 저장소 이름 또는 로컬 디렉터리 경로
    device: str = "cpu"  # "cpu" | "cuda" | "cuda:0" | "mps" | "auto"(accelerate 필요)
    local_files_only: bool = False  # True면 네트워크 없이 캐시·로컬 파일만 사용
    revision: str | None = None  # 재현성을 위해 커밋 해시 고정 권장
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    # Qwen3처럼 추론(<think>) 모드가 있는 채팅 템플릿용. False면 추론을 끈다 (짧은 JSON 출력이 잘리지 않게).
    # 이 변수를 쓰지 않는 템플릿은 무시한다. None이면 템플릿에 넘기지 않는다.
    enable_thinking: bool | None = False


class _LoadedModel:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model


_cache: dict[CategorizatorConfig, _LoadedModel] = {}
_cache_lock = threading.Lock()
_generate_lock = threading.Lock()  # 한 모델 인스턴스에서 동시 generate를 막는다


def _load(config: CategorizatorConfig) -> _LoadedModel:
    """설정별로 모델을 한 번만 읽는다."""
    with _cache_lock:
        if config in _cache:
            return _cache[config]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            logger.error("transformers/torch를 불러오지 못했습니다.", exc_info=exc, extra={"engine": ENGINE_NAME})
            raise LLMConfigError(
            "transformers 또는 torch를 불러오지 못했습니다. 설치와 실행 환경을 확인하세요.", engine=ENGINE_NAME
        ) from exc
        try:
            common = {"local_files_only": config.local_files_only, "revision": config.revision}
            tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
            if config.device == "auto":
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id, torch_dtype="auto", device_map="auto", **common
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id, torch_dtype="auto", **common
                ).to(config.device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - 외부 라이브러리 오류를 한 곳에서 감싼다
            logger.error(
                "모델 로딩 실패: model=%s local_only=%s",
                config.model_id,
                config.local_files_only,
                exc_info=exc,
                extra={"engine": ENGINE_NAME},
            )
            raise LLMError("모델을 불러오지 못했습니다.", engine=ENGINE_NAME) from exc
        loaded = _LoadedModel(tokenizer, model)
        _cache[config] = loaded
        return loaded


def clear_model_cache() -> None:
    """테스트나 모델 교체 때 읽어 둔 모델을 버린다."""
    with _cache_lock:
        _cache.clear()


class CategorizatorClient(LLMClient):
    name = ENGINE_NAME

    def __init__(self, config: CategorizatorConfig) -> None:
        if not config.model_id:
            raise ValueError("model_id가 비어 있습니다.")
        if config.max_new_tokens <= 0:
            raise ValueError("max_new_tokens는 1 이상이어야 합니다.")
        self._config = config

    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if tools:
            raise LLMConfigError("CATEGORIZATOR 엔진은 도구 호출을 지원하지 않습니다.", engine=ENGINE_NAME)

        chat_messages: list[dict[str, str]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool" or m.tool_calls:
                raise LLMConfigError(
                    "CATEGORIZATOR 엔진은 도구 메시지를 지원하지 않습니다.", engine=ENGINE_NAME
                )
            chat_messages.append({"role": m.role, "content": m.content})

        loaded = _load(self._config)
        limit = max_tokens or self._config.max_new_tokens

        try:
            template_kwargs: dict[str, Any] = {}
            if self._config.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._config.enable_thinking
            inputs = loaded.tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
            ).to(loaded.model.device)
            prompt_len = int(inputs["input_ids"].shape[-1])
            with _generate_lock:
                output = loaded.model.generate(
                    **inputs,
                    max_new_tokens=limit,
                    do_sample=False,  # 분류·JSON 출력은 결정적으로
                )
            new_ids = output[0][prompt_len:]
            text = loaded.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            generated = int(new_ids.shape[-1])
            eos = loaded.tokenizer.eos_token_id
            ended = eos is not None and generated > 0 and int(new_ids[-1]) == eos
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("모델 생성 실패", exc_info=exc, extra={"engine": ENGINE_NAME})
            raise LLMError("모델 호출에 실패했습니다.", engine=ENGINE_NAME) from exc

        stop_reason = "end" if ended or generated < limit else "max_tokens"
        return LLMResponse(
            text=text,
            stop_reason=stop_reason,
            usage=Usage(input_tokens=prompt_len, output_tokens=generated),
            model=self._config.model_id,
        )


def build_categorizator_client(*, settings: Any, model_id: str | None = None) -> CategorizatorClient:
    """`Settings`에서 설정을 읽어 클라이언트를 만든다 (모델은 첫 호출 때 읽는다).

    `model_id`를 주면 `CATEGORIZATOR_MODEL_ID` 대신 그 모델을 쓴다 (시나리오별 모델).
    `NETWORK_ENV=CLOSED`면 허브 다운로드를 시도하지 않고 로컬 파일만 쓴다.
    """
    model_id = model_id or getattr(settings, "categorizator_model_id", None)
    if not model_id:
        raise LLMConfigError("CATEGORIZATOR_MODEL_ID가 설정되지 않았습니다.", engine=ENGINE_NAME)
    local_only = bool(getattr(settings, "categorizator_local_files_only", False)) or (
        settings.network_env == "CLOSED"
    )
    return CategorizatorClient(
        CategorizatorConfig(
            model_id=model_id,
            device=getattr(settings, "categorizator_device", "cpu"),
            local_files_only=local_only,
            revision=getattr(settings, "categorizator_revision", None),
            max_new_tokens=getattr(settings, "categorizator_max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            enable_thinking=getattr(settings, "categorizator_enable_thinking", False),
        )
    )