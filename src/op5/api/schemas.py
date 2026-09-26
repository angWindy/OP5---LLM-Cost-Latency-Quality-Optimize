"""Pydantic request/response schemas for the OP5 API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    chroma_reachable: bool
    storage_backend: str
    embedder_status: str


class Track1ExtractRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to a scanned contract PDF.")
    config: str = Field(default="D", description="Track 1 config: A | B | D | B+D | D-strong")
    expected_fields: list[str] | None = Field(default=None, description="Optional override for fields to extract.")
    use_ocr: bool = Field(default=True, description="If False, skip OCR and use the GT-derived text (smoke-test path).")


class Track1Field(BaseModel):
    name: str
    value: str | None


class Track1ExtractResponse(BaseModel):
    case_id: str
    config: str
    redacted_text: str
    extracted_fields: dict[str, Any]
    model: str
    cost_usd: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


class Track2AskRequest(BaseModel):
    case_id: str = Field(..., description="SYNTH case_id, e.g. contract_synth-ctr-001")
    question: str = Field(..., description="User question in Vietnamese or English")
    config: str = Field(default="A", description="Track 2 config: A | B | C | D | B+D | C+D | B+C+D")
    top_k: int = Field(default=8, ge=1, le=32)


class Track2AskResponse(BaseModel):
    case_id: str
    question: str
    answer: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    config: str
    model: str
    cost_usd: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    context_chunks: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class InspectResponse(BaseModel):
    case_id: str
    track1: list[dict[str, Any]] = Field(default_factory=list)
    track2: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "HealthResponse",
    "Track1ExtractRequest",
    "Track1ExtractResponse",
    "Track1Field",
    "Track2AskRequest",
    "Track2AskResponse",
    "InspectResponse",
]
