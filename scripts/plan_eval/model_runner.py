"""transformers로 서버와 같은 방식으로 한 번 생성한다 (app/llm/llm_transformers.py generate_chat 과 같은 호출).

같게 맞춘 것: 채팅 템플릿(system + user, add_generation_prompt, enable_thinking=False), 그리디(do_sample=False),
skip_special_tokens 디코딩, 끊김 판정(마지막 토큰이 eos가 아니고 생성 수가 한도에 닿으면 max_tokens).
다른 것: 서버는 모델을 `torch_dtype="auto"`로 올린다. 여기서는 `--dtype`으로 고른다 (auto | float32 | bfloat16 | float16).
torch·transformers는 이 파일 안에서만, 모델을 읽을 때 불러온다 (하니스·시험은 둘 없이 돈다).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.agent.plan_brief import BRIEF_MAX_TOKENS  # 서버가 계획 호출에 쓰는 최대 생성 길이

from .harness import GenResult

DTYPES = ("auto", "float32", "bfloat16", "float16")


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Runner:
    model_id: str
    dtype: str
    device: str
    tokenizer: Any
    model: Any
    max_new_tokens: int
    load_seconds: float

    @property
    def revision(self) -> str:
        """읽은 모델의 허브 커밋 해시 (어댑터는 학습한 판에만 붙으므로 기록해 둔다). 모르면 빈 문자열."""
        return str(getattr(getattr(self.model, "config", None), "_commit_hash", None) or "")

    def _inputs(self, system: str, user: str) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False,
        ).to(self.model.device)

    def generate(self, system: str, user: str, *, max_new_tokens: int | None = None) -> GenResult:
        import torch

        limit = max_new_tokens or self.max_new_tokens
        inputs = self._inputs(system, user)
        prompt_len = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        seconds = time.perf_counter() - started
        new_ids = output[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        generated = int(new_ids.shape[-1])
        eos = self.tokenizer.eos_token_id
        ended = eos is not None and generated > 0 and int(new_ids[-1]) == eos
        return GenResult(text=text, prompt_tokens=prompt_len, gen_tokens=generated, seconds=seconds,
                         hit_max=not (ended or generated < limit))

    def __call__(self, system: str, user: str) -> GenResult:
        return self.generate(system, user)


def load_runner(model_id: str, *, dtype: str = "auto", device: str = "auto", max_new_tokens: int = BRIEF_MAX_TOKENS) -> Runner:
    if dtype not in DTYPES:
        raise ValueError(f"dtype은 {', '.join(DTYPES)} 중 하나여야 합니다: {dtype}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = pick_device(device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict[str, Any] = {}
    # 서버는 torch_dtype="auto". 최신 transformers는 dtype=을 받고 torch_dtype=을 경고하므로 둘 다 시도한다.
    value: Any = "auto" if dtype == "auto" else getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=value, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=value, **kwargs)
    model = model.to(resolved)
    model.eval()
    return Runner(model_id, dtype, resolved, tokenizer, model, max_new_tokens, time.perf_counter() - started)
