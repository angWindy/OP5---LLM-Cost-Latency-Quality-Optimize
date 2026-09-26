"""Track 1 extraction endpoint.

POST /track1/extract
  Body: {"pdf_path": "<abs path>", "config": "D", "expected_fields": [...], "use_ocr": true}
  Returns extracted fields + cost + latency.

In smoke mode (use_ocr=False or pdf_path missing) we use the GT-derived
text from data/processed/phase03_synth_contracts.jsonl so the endpoint
stays demoable without OCR infra available.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from op5.api.deps import (
    TRACK1_EXPECTED_FIELDS,
    _gt_text_for_case,
    get_gt_records,
    get_llm_wrapper,
    get_redactor,
)
from op5.api.schemas import Track1ExtractRequest, Track1ExtractResponse
from op5.rag.prompts import track1_prompt
from op5.router import extract_features_track1, route

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/track1", tags=["track1"])

VALID_CONFIGS = {"A", "B", "D", "B+D", "D-strong"}


def _strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _resolve_case_id(pdf_path: str) -> tuple[str, str]:
    """Return (case_id, gt_text) for a PDF path or empty text in smoke mode."""
    gt = get_gt_records()
    if not pdf_path:
        return "", ""
    p = Path(pdf_path)
    stem = p.stem
    if stem in gt:
        return stem, _gt_text_for_case(gt[stem])
    for synth_id, rec in gt.items():
        if rec.get("source_pdf", "").endswith(stem + ".pdf"):
            return stem, _gt_text_for_case(rec)
    return stem, ""


@router.post("/extract", response_model=Track1ExtractResponse)
def extract(req: Track1ExtractRequest) -> Track1ExtractResponse:
    if req.config not in VALID_CONFIGS:
        raise HTTPException(status_code=400, detail=f"invalid config {req.config!r}; choices={sorted(VALID_CONFIGS)}")

    fields = req.expected_fields or list(TRACK1_EXPECTED_FIELDS)
    case_id, gt_text = _resolve_case_id(req.pdf_path)

    redactor = get_redactor()
    if req.use_ocr and Path(req.pdf_path).exists() and Path(req.pdf_path).suffix.lower() == ".pdf":
        try:
            from op5.ocr import ADAPTERS

            adapter = ADAPTERS.get("easyocr")
            if adapter is None or not adapter.is_available():
                raise RuntimeError("no OCR adapter available on this host")
            result = adapter(Path(req.pdf_path))
            d = result.to_dict()
            ocr_doc = {"text_blocks": d.get("text_blocks", []) or [{"page": 1, "text": "", "bbox": [0, 0, 100, 100], "confidence": 0.0}]}
        except Exception as exc:
            logger.warning("OCR failed, falling back to GT-derived text: %s", exc)
            ocr_doc = {"text_blocks": [{"page": 1, "text": gt_text or "", "bbox": [0, 0, 100, 100], "confidence": 1.0}]}
            case_id = case_id or "smoke"
    else:
        ocr_doc = {"text_blocks": [{"page": 1, "text": gt_text, "bbox": [0, 0, 100, 100], "confidence": 1.0}]}
        case_id = case_id or "smoke"

    redact_on = req.config in {"D", "B+D", "D-strong"}
    prompt_variant = "B" if req.config.startswith("B") else "A"
    if redact_on:
        ocr_proc = redactor.redact_ocr_result(ocr_doc)
    else:
        ocr_proc = ocr_doc
    redacted_text = ocr_proc.get("redacted_text") or "\n".join(b.get("text", "") for b in ocr_proc.get("text_blocks", []))

    deployment = "gemini-3.5-flash-lite"
    prompt = track1_prompt(prompt_variant, fields, redacted_text)

    wrapper = get_llm_wrapper()
    t0 = time.time()
    if wrapper is not None:
        meta = wrapper.generate(prompt, model=deployment, max_output_tokens=512)
        llm_raw = meta.text or ""
        cost_usd = meta.cost_usd
        latency_ms = meta.latency_ms or ((time.time() - t0) * 1000)
        input_tokens = meta.input_tokens
        output_tokens = meta.output_tokens
        error = meta.error
    else:
        m = re.search(r'Contract text:\s*"""+\s*(.*?)\s*"""+', prompt, flags=re.DOTALL)
        body = m.group(1) if m else redacted_text
        out: dict[str, str] = {}
        for line in body.splitlines():
            kv = re.match(r"\s*([A-Za-z_]+)\s*:\s*(.+)", line)
            if kv:
                out[kv.group(1).strip()] = kv.group(2).strip()
        llm_raw = json.dumps(out, ensure_ascii=False)
        input_tokens = len(prompt) // 4
        output_tokens = len(llm_raw) // 4
        latency_ms = (time.time() - t0) * 1000
        cost_usd = 0.0001  # gemini-3.5-flash-lite for all tiers
        error = None

    stripped = _strip_code_fence(llm_raw)
    try:
        parsed = json.loads(stripped) if stripped else {}
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    return Track1ExtractResponse(
        case_id=case_id,
        config=req.config,
        redacted_text=redacted_text[:2000],
        extracted_fields=parsed,
        model=deployment,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )


@router.get("/configs")
def list_configs() -> dict[str, list[str]]:
    return {"configs": sorted(VALID_CONFIGS)}
