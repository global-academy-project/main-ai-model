import json
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

from app.rag import http
from app.rag.config import EmbeddingServerSettings, RagSettings
from app.rag.embedding import embed_text
from app.rag.embedding_api import create_app
from app.rag.embedding_model import load_encoder


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    """Use a tiny random local model, never download production weights."""
    directory = tmp_path_factory.mktemp("embedding-model")
    tokenizer = Tokenizer(models.WordLevel(
        {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "fire": 4, "safe": 5},
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>",
        cls_token="<s>", sep_token="</s>",
    ).save_pretrained(directory)
    model = XLMRobertaModel(XLMRobertaConfig(
        vocab_size=6, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
    ))
    model.save_pretrained(directory)
    settings = EmbeddingServerSettings(
        _env_file=None, embedding_model_path=str(directory),
        embedding_model="BAAI/bge-m3", embedding_dim=8,
        embedding_device="cpu", embedding_local_files_only=True,
        embedding_max_tokens=8, embedding_max_batch_size=2,
    )
    return load_encoder(settings)


def test_encoder_normalizes_cls_and_counts_non_padding_tokens(encoder):
    vectors, count = encoder.encode(["fire", "fire safe"])
    tokens = encoder.tokenizer(["fire", "fire safe"], padding=True, return_tensors="pt")
    with torch.inference_mode():
        cls = encoder.model(**tokens).last_hidden_state[:, 0]
        expected = torch.nn.functional.normalize(cls, p=2, dim=1)
    assert torch.allclose(torch.tensor(vectors), expected)
    assert count == 7
    assert all(len(vector) == 8 for vector in vectors)
    assert torch.allclose(torch.tensor(vectors).norm(dim=1), torch.ones(2))
    assert not encoder.model.training


def test_encoder_rejects_long_inputs_without_truncating(encoder):
    with pytest.raises(ValueError, match="토큰"):
        encoder.encode(["fire " * 8])
    with pytest.raises(ValueError, match="개수"):
        encoder.encode(["fire"] * 3)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_encoder_rejects_invalid_vectors(encoder, monkeypatch, value):
    def forward(**tokens):
        return SimpleNamespace(
            last_hidden_state=torch.full((1, 3, 8), value),
        )

    monkeypatch.setattr(encoder.model, "forward", forward)
    with pytest.raises(RuntimeError, match="유효하지 않은"):
        encoder.encode(["fire"])


def test_loader_rejects_dimension_mismatch(encoder):
    settings = encoder.settings.model_copy(update={"embedding_dim": 1024})
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        load_encoder(settings)


def test_loader_fails_when_local_model_is_missing(tmp_path):
    settings = EmbeddingServerSettings(
        _env_file=None, embedding_model_path=str(tmp_path / "missing"),
        embedding_local_files_only=True,
    )
    with pytest.raises((OSError, ValueError)):
        load_encoder(settings)


def test_api_loads_once_and_supports_single_and_batch(encoder):
    loaded = []

    def factory(settings):
        loaded.append(settings)
        return encoder

    app = create_app(encoder.settings, encoder_factory=factory)
    with TestClient(app) as client:
        assert client.get("/ready").json()["dimensions"] == 8
        single = client.post("/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": "fire",
        })
        assert single.status_code == 200
        body = single.json()
        assert body["object"] == "list"
        assert body["model"] == "BAAI/bge-m3"
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert len(body["data"][0]["embedding"]) == 8
        assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
        batch = client.post("/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": ["fire", "safe"],
        })
        assert batch.status_code == 200
        assert [item["index"] for item in batch.json()["data"]] == [0, 1]
    assert len(loaded) == 1
    assert app.state.encoder is None


@pytest.mark.parametrize("body,status", [
    ({"model": "wrong", "input": "fire"}, 400),
    ({"model": "BAAI/bge-m3", "input": ""}, 422),
    ({"model": "BAAI/bge-m3", "input": " "}, 422),
    ({"model": "BAAI/bge-m3", "input": []}, 422),
    ({"model": "BAAI/bge-m3", "input": ["fire", " "]}, 422),
    ({"model": "BAAI/bge-m3", "input": [1]}, 422),
    ({"model": "BAAI/bge-m3", "input": ["fire"] * 3}, 422),
    ({"model": "BAAI/bge-m3", "input": "fire " * 8}, 422),
    ({"model": "BAAI/bge-m3", "input": "f" * 32769}, 422),
    ({"model": "BAAI/bge-m3", "input": "fire", "encoding_format": "base64"}, 422),
])
def test_invalid_api_requests(encoder, body, status):
    app = create_app(encoder.settings, encoder_factory=lambda settings: encoder)
    with TestClient(app) as client:
        assert client.post("/v1/embeddings", json=body).status_code == status


def test_optional_bearer_authentication(encoder):
    settings = EmbeddingServerSettings(
        _env_file=None, embedding_api_key="test-key", embedding_dim=8,
    )
    app = create_app(settings, encoder_factory=lambda settings: encoder)
    with TestClient(app) as client:
        body = {"model": "BAAI/bge-m3", "input": "fire"}
        assert client.get("/ready").status_code == 200
        assert client.post("/v1/embeddings", json=body).status_code == 401
        assert client.post(
            "/v1/embeddings", json=body,
            headers={"Authorization": "Bearer wrong"},
        ).status_code == 401
        assert client.post(
            "/v1/embeddings", json=body,
            headers={"Authorization": "Bearer test-key"},
        ).status_code == 200


def test_not_ready_and_startup_failure(encoder):
    app = create_app(encoder.settings, encoder_factory=lambda settings: encoder)
    with TestClient(app) as client:
        app.state.encoder = None
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": "fire",
        }).status_code == 503

    def broken(settings):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        with TestClient(create_app(encoder.settings, encoder_factory=broken)):
            pass


def test_inference_failure_returns_503(encoder, monkeypatch):
    def fail(texts):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(encoder, "encode", fail)
    app = create_app(encoder.settings, encoder_factory=lambda settings: encoder)
    with TestClient(app) as client:
        result = client.post("/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": "fire",
        })
        assert result.status_code == 503
        assert "inference failed" not in result.text


def test_existing_search_client_can_read_api_response(encoder, monkeypatch):
    app = create_app(encoder.settings, encoder_factory=lambda settings: encoder)
    with TestClient(app) as client:
        def respond(request, timeout):
            response = client.post(
                "/v1/embeddings", json=json.loads(request.data),
            )
            assert response.status_code == 200
            return BytesIO(response.content)

        monkeypatch.setattr(http, "urlopen", respond)
        settings = RagSettings(
            embedding_url="http://embedding/v1/embeddings",
            qdrant_url="http://qdrant:6333",
            embedding_model="BAAI/bge-m3", embedding_dim=8, timeout_seconds=15,
            embedding_api_key=None, qdrant_api_key=None,
        )
        vector = embed_text(settings, "fire")
        assert len(vector) == 8
        assert sum(value * value for value in vector) == pytest.approx(1.0)
