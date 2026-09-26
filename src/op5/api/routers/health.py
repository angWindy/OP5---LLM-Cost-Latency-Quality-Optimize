"""Health probe endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

from op5.api.deps import API_LLM_MODE, chroma_reachable
from op5.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    keys = 0
    multi = os.getenv("GOOGLE_API_KEYS", "").strip()
    if multi:
        keys = len([k for k in multi.split(",") if k.strip()])
    else:
        single = os.getenv("GOOGLE_API_KEY", "").strip()
        if single:
            keys = 1

    try:
        from op5.storage import STORAGE

        backend_name = STORAGE().backend_name()
    except Exception:
        backend_name = "unknown"

    try:
        from op5.rag.embedding import MultilingualEmbedder

        MultilingualEmbedder()
        embedder_status = "loaded"
    except Exception as exc:
        embedder_status = f"error: {type(exc).__name__}"

    return HealthResponse(
        status="ok",
        keys_configured=keys,
        chroma_reachable=chroma_reachable(),
        storage_backend=backend_name,
        embedder_status=embedder_status,
    )


@router.get("/healthz/llm-mode")
def llm_mode() -> dict[str, str]:
    return {"mode": API_LLM_MODE}
