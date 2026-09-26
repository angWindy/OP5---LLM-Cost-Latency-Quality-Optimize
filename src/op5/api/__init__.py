"""OP5 FastAPI service — exposes Track 1 extraction, Track 2 RAG, health + inspect endpoints.

Usage:
    conda activate vsf
    pip install fastapi uvicorn pydantic httpx
    uvicorn scripts.api.serve:app --host 0.0.0.0 --port 8000 --reload
    # Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from op5.api.routers import extract, health, inspect, rag

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="OP5 Live Demo API",
        description=(
            "Wraps the Phase 03 pipeline (OCR → redact → LLM → score) and exposes "
            "endpoints for Track 1 (extraction) and Track 2 (RAG). See /docs for the "
            "OpenAPI schema."
        ),
        version="0.1.0",
    )

    app.include_router(health.router)
    app.include_router(extract.router)
    app.include_router(rag.router)
    app.include_router(inspect.router)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "service": "op5-live-api",
            "version": app.version,
            "docs": "/docs",
            "health": "/healthz",
        }

    return app


__all__ = ["create_app"]
