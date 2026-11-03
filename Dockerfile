FROM python:3.12.15-slim AS base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

#-------------------------[Multi-stage build base image]---------------------------------------
# Download the ML wheels once; BuildKit keeps pip's download cache outside image layers.
FROM base AS ml-wheelhouse

COPY requirements.txt requirements.release.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip download --only-binary=:all: --dest /tmp/wheels \
        -r requirements.release.txt

# Install ML dependencies offline from the wheelhouse.
FROM base AS ml-dependencies

COPY requirements.txt requirements.release.txt ./
COPY --from=ml-wheelhouse /tmp/wheels /tmp/wheels
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --no-index \
        --find-links=/tmp/wheels -r requirements.release.txt \
    && /opt/venv/bin/python -m pip check \
    && rm -rf /tmp/wheels

#-------------------------[Multi-stage build AI image]---------------------------------------
FROM ml-dependencies AS ai

COPY app ./app

EXPOSE 60100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60100"]

#-------------------------[Multi-stage build Embedding image]---------------------------------------
FROM ml-dependencies AS embedding

COPY app ./app

EXPOSE 60110

CMD ["uvicorn", "app.embedding.embedding_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "60110"]

#-------------------------[Multi-stage build Ingest image]---------------------------------------
FROM base AS ingest-dependencies

COPY requirements.ingest.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.ingest.txt \
    && /opt/venv/bin/python -m pip check

FROM ingest-dependencies AS ingest

COPY app ./app

EXPOSE 8001

CMD ["uvicorn", "app.ingest_main:app", "--host", "0.0.0.0", "--port", "8001"]