# Main AI Model

## Docker

The Dockerfile provides two CPU-only runtime targets:

| Target | Application | Port |
| --- | --- | --- |
| `api` (default) | General AI API | 60100 |
| `embedding` | BGE-M3 embedding API | 60101 |

Both targets share the dependency builder because the general API also runs
local Transformers models. Only the installed virtual environment and application
source are copied into runtime images; local `.env` files are not included.
The previous single-stage specification is preserved as comments in the Dockerfile.

```powershell
docker build --target api -t main-ai-model:api .
docker build --target embedding -t main-ai-model:embedding .
```

Run both services on the same Docker network. Replace the model directory below
with an existing local BGE-M3 model directory containing its weights, configuration,
and tokenizer files.

```powershell
docker network create main-ai-model
docker run -d --name embedding --network main-ai-model -p 60101:60101 --mount "type=bind,source=C:\models\bge-m3,target=/models/bge-m3,readonly" -e EMBEDDING_MODEL_PATH=/models/bge-m3 main-ai-model:embedding
docker run -d --name ai-api --network main-ai-model -p 60100:60100 --env-file .env -e EMBEDDING_URL=http://embedding:60101 main-ai-model:api
```

Use a valid `.env` for the general API, including `NETWORK_ENV=OPEN` or
`NETWORK_ENV=CLOSED`. Set `RAG_BACKEND=app.rag.tool_backend:build_search_backend`
and `QDRANT_URL` when enabling RAG; setting `EMBEDDING_URL` alone does not enable it.
If embedding authentication is required, pass the same `EMBEDDING_API_KEY` to
both containers.

The embedding server defaults to local-only model loading. For an online model
download instead of a bind mount, omit the mount and `EMBEDDING_MODEL_PATH`, and
set `EMBEDDING_LOCAL_FILES_ONLY=false`. Model initialization may take time;
`http://localhost:60101/ready` reports readiness after initialization.
These images contain CPU-only PyTorch and do not support `EMBEDDING_DEVICE=cuda`.
