from fastapi import FastAPI
from app.api.rag import router as rag_router
from app.core.logger import setup_logging

import logging

setup_logging()

log = logging.getLogger("app")

app = FastAPI(title="HABR RAG API")

app.include_router(rag_router, prefix="/api/v1")

@app.get("/health")
def health():
    log.info("Ping to /health endpoint received")
    return {"status": "ok"}