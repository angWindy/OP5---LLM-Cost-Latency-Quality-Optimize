"""Lazy singletons shared across FastAPI request handlers.

Singletons:
- LLM wrapper (`LLMWrapper`) — Gemini client + pricing
- Redactor (`Redactor`) — applied to OCR output before LLM
- Track 1 expected fields list — mirrors scripts/phase-03/run_extraction_track1.py
- Index helper — builds Track-2 retrieval index over the SYNTH corpus

Selection between live Gemini and offline stub is driven by env var
`OP5_API_LLM` (default `gemini`). When `stub`, `LLMWrapper` is not built and
routers use the offline deterministic fallback. This keeps the API
runnable on a host with no API key.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_LLM_MODE = os.getenv("OP5_API_LLM", "gemini").strip().lower()


TRACK1_EXPECTED_FIELDS = [
    "contract_no",
    "sign_date",
    "seller_name",
    "seller_tax_id",
    "buyer_name",
    "buyer_phone",
    "buyer_email",
    "model",
    "version",
    "color",
    "vin",
    "unit_price_vnd",
    "total_price_words_vi",
]


_GT_PATH = Path("data/processed/phase03_synth_contracts.jsonl")
_QA_PATH = Path("data/processed/phase03_mini_qa.jsonl")


def _gt_text_for_case(rec: dict[str, Any]) -> str:
    """Build a synthetic OCR text from ground-truth field values (mirrors runners)."""
    fields = rec.get("ground_truth_fields", {}) or {}
    lines: list[str] = []
    ordered = [
        "contract_no", "sign_date", "seller_name", "seller_address",
        "seller_tax_id", "seller_rep", "seller_rep_title", "buyer_name",
        "buyer_address", "buyer_phone", "buyer_email", "model", "version",
        "color", "vin", "unit_price_vnd", "total_price_words_vi",
        "delivery_date", "delivery_place", "special_offer_a",
        "special_offer_b", "special_offer_c", "special_offer_d",
    ]
    for k in ordered:
        v = fields.get(k)
        if v is None or v == "":
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _load_gt_records() -> dict[str, dict]:
    if not _GT_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    with _GT_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["case_id"]] = rec
    return out


def _load_qa_records() -> list[dict]:
    if not _QA_PATH.exists():
        return []
    with _QA_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_GT_CACHE: dict[str, dict] | None = None
_QA_CACHE: list[dict] | None = None
_INDEX_LOCK = threading.Lock()
_INDEX_BUILT = False
_RETRIEVER: Any = None
_RERANKER: Any = None


def get_gt_records() -> dict[str, dict]:
    global _GT_CACHE
    if _GT_CACHE is None:
        _GT_CACHE = _load_gt_records()
    return _GT_CACHE


def get_qa_records() -> list[dict]:
    global _QA_CACHE
    if _QA_CACHE is None:
        _QA_CACHE = _load_qa_records()
    return _QA_CACHE


@lru_cache(maxsize=1)
def get_redactor():
    from op5.redact import Redactor, default_policy
    return Redactor(default_policy())


@lru_cache(maxsize=1)
def get_llm_wrapper():
    if API_LLM_MODE != "gemini":
        logger.info("OP5_API_LLM=%s: not building live LLM wrapper", API_LLM_MODE)
        return None
    try:
        from op5.llm import make_llm_wrapper
        return make_llm_wrapper()
    except RuntimeError as exc:
        logger.warning("Live LLM wrapper unavailable (%s); API will return stub responses", exc)
        return None


def get_track2_pipeline(force_rebuild: bool = False):
    """Lazily build a Track 2 RAG pipeline index over the SYNTH corpus.

    Returns (retriever, reranker) tuple, or (None, None) if build fails.
    """
    global _INDEX_BUILT, _RETRIEVER, _RERANKER
    if _INDEX_BUILT and not force_rebuild:
        return _RETRIEVER, _RERANKER

    with _INDEX_LOCK:
        if _INDEX_BUILT and not force_rebuild:
            return _RETRIEVER, _RERANKER
        try:
            from op5.rag.chunking import chunk_documents
            from op5.rag.embedding import MultilingualEmbedder
            from op5.rag.rerank import BM25Reranker
            from op5.rag.retrieval import Retriever

            redactor = get_redactor()
            gt = get_gt_records()
            docs: list[tuple[str, str]] = []
            for synth_id, rec in gt.items():
                redacted = redactor.redact_ocr_result(
                    {"text_blocks": [{"page": 1, "text": _gt_text_for_case(rec), "bbox": [0, 0, 100, 100], "confidence": 1.0}]}
                )
                text = redacted.get("redacted_text", "")
                case_id = (
                    rec.get("source_pdf", "").split("/")[-1].replace(".pdf", "")
                    or f"contract_synth-{synth_id}"
                )
                docs.append((case_id, text))

            chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=40)
            retriever = Retriever(
                collection="op5_api_track2",
                embedder=MultilingualEmbedder(),
            )
            try:
                retriever.index(chunks)
            except Exception as e:
                logger.warning("Track 2 index build failed (%s); using deterministic fallback", e)
            _RETRIEVER = retriever
            _RERANKER = BM25Reranker()
            _INDEX_BUILT = True
            return retriever, _RERANKER
        except Exception as exc:
            logger.exception("Track 2 pipeline init failed: %s", exc)
            return None, None


def chroma_reachable() -> bool:
    """Probe whether the configured ChromaDB is reachable."""
    try:
        from op5.storage import STORAGE
        backend = STORAGE().backend_name()
        if backend == "docker":
            STORAGE().chroma().count("op5_api_track2")
            return True
        return True
    except Exception as e:
        logger.debug("Chroma probe failed: %s", e)
        return False


def reset_caches_for_tests() -> None:
    global _GT_CACHE, _QA_CACHE, _INDEX_BUILT, _RETRIEVER, _RERANKER
    _GT_CACHE = None
    _QA_CACHE = None
    _INDEX_BUILT = False
    _RETRIEVER = None
    _RERANKER = None
    get_redactor.cache_clear()
    get_llm_wrapper.cache_clear()


__all__ = [
    "API_LLM_MODE",
    "TRACK1_EXPECTED_FIELDS",
    "get_gt_records",
    "get_qa_records",
    "get_redactor",
    "get_llm_wrapper",
    "get_track2_pipeline",
    "chroma_reachable",
    "reset_caches_for_tests",
    "_gt_text_for_case",
]
