# Prometheus metrics wrapper

This directory provides an isolated Prometheus instrumentation layer for the existing AI FastAPI application.
It does not modify the existing application entrypoint or dependency manifests.

Run locally:

```powershell
python -m pip install -r metrix\requirements.txt
python -m uvicorn metrix.main:app --host 127.0.0.1 --port 60100
```

The infrastructure Compose configuration builds `python-ai-model` using `metrix/Dockerfile`.
Rebuild and restart the service from `main-infra-manage\docker` after configuring its
required environment variables:

```powershell
docker compose up -d --build python-ai-model
```

The existing Prometheus configuration scrapes `python-ai-model:60100/metrics` on the
Compose network. Check the `python-ai-model` target in Prometheus after it is running.
Only the AI model service is instrumented by this wrapper; the embedding and ingest
services need their own instrumentation.

The metrics are process-local. Run one Uvicorn worker per container; multiple workers
need Prometheus multiprocess configuration, which this wrapper does not provide.
