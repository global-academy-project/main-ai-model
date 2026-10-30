# Previous single-stage specification (preserved for reference).
# FROM python:3.12.15-slim
#
# RUN echo "Updating apt and installing system dependencies..."
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     && rm -rf /var/lib/apt/lists/*
#
# WORKDIR /app
# COPY . .
#
# RUN echo "Copy requirements.txt to /app"
# COPY requirements.release.txt .
#
# RUN echo "Creating virtual environment at /opt/venv"
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
#
# RUN echo "Installing Python dependencies..."
# RUN python -m pip download --no-cache-dir --only-binary=:all: \
#         --dest /tmp/wheels -r ./requirements.release.txt \
#     && for wheel in /tmp/wheels/*.whl; do \
#         echo "Verifying $wheel"; \
#         python -c "import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); bad = archive.testzip(); archive.close(); sys.exit('Corrupt wheel member: ' + bad if bad else 0)" "$wheel" || exit 1; \
#     done \
#     && python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
#         -r ./requirements.release.txt \
#     && python -m pip check \
#     && rm -rf /tmp/wheels
#
# EXPOSE 60100
#
# RUN echo "Starting the application..."
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60100"]

FROM python:3.12.15-slim AS base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

FROM base AS dependencies

COPY requirements.txt requirements.release.txt ./
RUN python -m venv /opt/venv
RUN python -m pip download --no-cache-dir --only-binary=:all: \
        --dest /tmp/wheels -r ./requirements.release.txt \
    && for wheel in /tmp/wheels/*.whl; do \
        echo "Verifying $wheel"; \
        python -c "import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); bad = archive.testzip(); archive.close(); sys.exit('Corrupt wheel member: ' + bad if bad else 0)" "$wheel" || exit 1; \
    done \
    && python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
        -r ./requirements.release.txt \
    && python -m pip check \
    && rm -rf /tmp/wheels

FROM base AS runtime

COPY --from=dependencies /opt/venv /opt/venv
COPY app ./app

FROM runtime AS embedding

EXPOSE 60101
CMD ["uvicorn", "app.rag.embedding_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "60101"]

# Keep the API as the default target for existing build commands.
FROM runtime AS api

EXPOSE 60100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60100"]