from fastapi import FastAPI

app = FastAPI(title="KB Ingest API")

@app.get("/ready")
def ready():
    return {"status": "ok"}