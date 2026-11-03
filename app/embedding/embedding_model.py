from __future__ import annotations

from collections import Counter
from threading import Lock

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import EmbeddingServerSettings
from .types import HybridEmbedding, SparseVector


class BgeM3Encoder:
    def __init__(
        self,
        settings: EmbeddingServerSettings,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
    ) -> None:
        self.settings = settings
        self.tokenizer = tokenizer
        self.model = model
        self._lock = Lock()

    def encode(self, texts: list[str]) -> tuple[list[HybridEmbedding], int]:
        if not 1 <= len(texts) <= self.settings.embedding_max_batch_size:
            raise ValueError("임베딩 요청의 문장 개수가 허용 범위를 벗어났습니다.")
        with self._lock, torch.inference_mode():
            tokens = self.tokenizer(
                texts, padding=True, truncation=False, return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] > self.settings.embedding_max_tokens:
                raise ValueError("입력이 최대 토큰 수를 초과했습니다. 청크를 나누어 요청하세요.")
            token_count = int(tokens["attention_mask"].sum().item())
            tokens = tokens.to(self.settings.embedding_device)
            output = self.model(**tokens)
            # BGE-M3 dense retrieval uses the normalized CLS token, not mean pooling.
            vectors = output.last_hidden_state[:, 0].float()
            if vectors.shape != (len(texts), self.settings.embedding_dim):
                raise RuntimeError("임베딩 모델의 출력 차원이 설정과 다릅니다.")
            norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
            if (
                not torch.isfinite(vectors).all().item()
                or not torch.isfinite(norms).all().item()
                or (norms == 0).any().item()
            ):
                raise RuntimeError("임베딩 모델이 유효하지 않은 벡터를 반환했습니다.")
            dense_vectors = (vectors / norms).cpu().tolist()
            special_tokens = set(self.tokenizer.all_special_ids)
            if self.tokenizer.unk_token_id is not None:
                special_tokens.discard(self.tokenizer.unk_token_id)
            sparse_vectors: list[SparseVector] = []
            for input_ids, attention_mask in zip(
                tokens["input_ids"].tolist(), tokens["attention_mask"].tolist(), strict=True,
            ):
                term_counts = Counter(
                    token_id
                    for token_id, attended in zip(input_ids, attention_mask, strict=True)
                    if attended and token_id not in special_tokens
                )
                if not term_counts:
                    raise ValueError("입력에 임베딩할 토큰이 없습니다.")
                indices = sorted(term_counts)
                sparse_vectors.append(
                    SparseVector(
                        indices=indices,
                        values=[float(term_counts[index]) for index in indices],
                    )
                )
            return (
                [
                    HybridEmbedding(dense=dense, sparse=sparse)
                    for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
                ],
                token_count,
            )


def load_encoder(settings: EmbeddingServerSettings) -> BgeM3Encoder:
    if settings.embedding_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("EMBEDDING_DEVICE=cuda이지만 CUDA를 사용할 수 없습니다.")
    source = settings.embedding_model_path or settings.embedding_model
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=settings.embedding_local_files_only,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        source, local_files_only=settings.embedding_local_files_only,
        trust_remote_code=False,
    )
    if model.config.hidden_size != settings.embedding_dim:
        raise ValueError("모델 hidden_size와 EMBEDDING_DIM이 일치하지 않습니다.")
    model.to(settings.embedding_device)
    model.eval()
    encoder = BgeM3Encoder(settings, tokenizer, model)
    encoder.encode(["ready"])
    return encoder
