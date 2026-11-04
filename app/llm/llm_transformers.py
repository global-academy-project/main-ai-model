"""transformers 로컬 모델로 글자를 생성하는 공용 부분 (`LLMClient.chat()` 기본 구현이 부른다).

- 모델은 처음 `chat()`을 부를 때 한 번만 읽고(지연 로딩) 프로세스 안에서 공유한다. 같은 설정(`TransformersConfig`)이면
  엔진이 달라도 같은 모델을 쓴다.
- `transformers`·`torch`는 지연 임포트한다. 설치돼 있지 않아도 앱은 뜬다.
- 도구 호출(tool use)은 지원하지 않는다. 도구는 코드가 정한 순서로 부르고, LLM은 JSON 텍스트만 돌려준다.
- 오류 정책(지침 4장): `LLMConfigError`에는 직접 쓴 고정 문구만 넘기고, transformers가 낸 오류 원문은 로그에만 남긴다.
  호출 실패는 `LLMError`(502). 손질 지점(`prepare_messages`·`generate_options`·`finish_text`)의 예외는 감싸지 않는다.
- 오류·로그의 엔진 이름은 호출한 클라이언트의 `name`을 쓴다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.exceptions import LLMConfigError, LLMError

from .llm_types import LLMResponse, Message, ToolSpec, Usage

if TYPE_CHECKING:
    from .llm_base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_NAME = "CATEGORIZATOR"

@dataclass(frozen=True)
class TransformersConfig:
    """모델을 어디서 어떻게 읽을지. 해시 가능해서 로딩 캐시 키로 쓴다."""

    model_id: str  # 허브 저장소 이름 또는 로컬 디렉터리 경로
    max_new_tokens: int  # 최대 생성 길이. 모든 엔진이 Settings.ai_max_output_tokens(AI_MAX_OUTPUT_TOKENS) 하나를 쓴다
    device: str = "cpu"  # "cpu" | "cuda" | "cuda:0" | "mps" | "auto"(accelerate 필요)
    local_files_only: bool = False  # True면 네트워크 없이 캐시·로컬 파일만 사용
    revision: str | None = None  # 재현성을 위해 커밋 해시 고정 권장
    # Qwen3처럼 추론(<think>) 모드가 있는 채팅 템플릿용. False면 추론을 끈다 (짧은 JSON 출력이 잘리지 않게).
    # 이 변수를 쓰지 않는 템플릿은 무시한다. None이면 템플릿에 넘기지 않는다.
    enable_thinking: bool | None = False


class _LoadedModel:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model


_cache: dict[TransformersConfig, _LoadedModel] = {}
_cache_lock = threading.Lock()
_generate_lock = threading.Lock()  # 한 모델 인스턴스에서 동시 generate를 막는다


def _load(config: TransformersConfig, engine: str = DEFAULT_ENGINE_NAME) -> _LoadedModel:
    """설정별로 모델을 한 번만 읽는다. 같은 설정이면 시나리오 엔진끼리도 같은 모델을 공유한다."""
    with _cache_lock:
        if config in _cache:
            return _cache[config]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            logger.error("transformers/torch를 불러오지 못했습니다.", exc_info=exc, extra={"engine": engine})
            raise LLMConfigError(
            "transformers 또는 torch를 불러오지 못했습니다. 설치와 실행 환경을 확인하세요.", engine=engine
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
                extra={"engine": engine},
            )
            raise LLMError("모델을 불러오지 못했습니다.", engine=engine) from exc
        loaded = _LoadedModel(tokenizer, model)
        _cache[config] = loaded
        return loaded


def clear_model_cache() -> None:
    """테스트나 모델 교체 때 읽어 둔 모델을 버린다."""
    with _cache_lock:
        _cache.clear()


def generate_chat(
    client: LLMClient,
    config: TransformersConfig,
    messages: list[Message],
    *,
    system: str = "",
    tools: list[ToolSpec] | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """`client`의 손질 지점을 거쳐 transformers로 한 번 생성한다."""
    engine = client.name
    if tools:
        raise LLMConfigError(f"{engine} 엔진은 도구 호출을 지원하지 않습니다.", engine=engine)

    chat_messages: list[dict[str, str]] = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "tool" or m.tool_calls:
            raise LLMConfigError(
                f"{engine} 엔진은 도구 메시지를 지원하지 않습니다.", engine=engine
            )
        chat_messages.append({"role": m.role, "content": m.content})
    chat_messages = client.prepare_messages(chat_messages)
    extra = client.generate_options()
    if {"max_new_tokens", "do_sample"} & set(extra):
        raise LLMConfigError(f"{engine} 엔진의 generate_options에 max_new_tokens·do_sample은 넣지 않습니다.", engine=engine)

    loaded = _load(config, engine)
    # 최대 생성 길이: 설정값(ai_max_output_tokens)이 모든 엔진의 상한이다. 호출자가 더 작게 요청하면(라우터 200 등) 그 값을 쓴다
    limit = min(max_tokens, config.max_new_tokens) if max_tokens else config.max_new_tokens

    try:
        template_kwargs: dict[str, Any] = {}
        if config.enable_thinking is not None:
            template_kwargs["enable_thinking"] = config.enable_thinking
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
                **extra,
            )
        new_ids = output[0][prompt_len:]
        raw_text = loaded.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        generated = int(new_ids.shape[-1])
        eos = loaded.tokenizer.eos_token_id
        ended = eos is not None and generated > 0 and int(new_ids[-1]) == eos
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("모델 생성 실패", exc_info=exc, extra={"engine": engine})
        raise LLMError("모델 호출에 실패했습니다.", engine=engine) from exc

    # 손질 지점은 try 밖에서 부른다 (시나리오 엔진 코드의 버그를 "모델 호출 실패"로 숨기지 않게)
    text = client.finish_text(raw_text)
    stop_reason = "end" if ended or generated < limit else "max_tokens"
    return LLMResponse(
        text=text,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=prompt_len, output_tokens=generated),
        model=config.model_id,
    )
