FROM python:3.12.15-slim AS base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.release.txt requirements.txt ./
COPY metrix/requirements.txt ./metrix/requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.release.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r metrix/requirements.txt \
    && /opt/venv/bin/python -m pip check \
    && mkdir -p /models/adapters /models/yolox /models/huggingface \
    && chown -R 65003:65003 /models/adapters /models/yolox /models/huggingface

FROM base AS ai

COPY app ./app
COPY metrix ./metrix
COPY adapters/extract/ /models/adapters/extract/
COPY adapters/work_register/ /models/adapters/work_register/
COPY adapters/helmet_nano.onnx /models/yolox/helmet_nano.onnx
COPY adapters/yolox_nano.onnx /models/yolox/yolox_nano.onnx

USER 65003:65003

EXPOSE 60100

CMD ["uvicorn", "metrix.main:app", "--host", "0.0.0.0", "--port", "60100"]
