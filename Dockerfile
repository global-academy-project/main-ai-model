FROM python:3.12.15-slim

RUN echo "Updating apt and installing system dependencies..."
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN echo "Copy requirements.txt to /app"
COPY requirements.txt .

COPY . .

RUN echo "Creating virtual environment at /opt/venv"
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN echo "Installing Python dependencies..."
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 60100

RUN echo "Starting the application..."
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60100"]