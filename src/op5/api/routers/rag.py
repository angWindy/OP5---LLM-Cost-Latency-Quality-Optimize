"""Track 2 RAG endpoint.

POST /track2/ask
  Body: {"case_id": "contract_synth-ctr-001", "question": "...", "config": "A", "top_k": 8}
  Returns answer + citations + cost + latency.

If the Track 2 index hasn't been built yet, lazily build it from the SYNTH
corpus (data/processed/phase03_synth_contracts.jsonl). If ChromaDB is
unavailable, fall back to deterministic in-process scoring.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException

from op5.api.deps import (
    _gt_text_for_case,
    get_gt_records,
    get_llm_wrapper,
    get_qa_records,
    get_redactor,
    get_track2_pipeline,
)
from op5.api.schemas import Track2AskRequest, Track2AskResponse
from op5.rag.chunking import chunk_documents
from op5.rag.prompts import track2_prompt
from op5.rag.retrieval import RetrievedChunk
from op5.router import extract_features_track2, route

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/track2", tags=["track2"])

VALID_CONFIGS = {"A", "B", "C", "D", "B+D", "C+D", "B+C+D"}


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


@router.post("/ask", response_model=Track2AskResponse)
def ask(req: Track2AskRequest) -> Track2AskResponse:
    if req.config not in VALID_CONFIGS:
        raise HTTPException(status_code=400, detail=f"invalid config {req.config!r}; choices={sorted(VALID_CONFIGS)}")

    cfg_meta = {
        "A": {"prompt_variant": "A", "use_rerank": False, "use_router": False, "tier": "cheap"},
        "B": {"prompt_variant": "B", "use_rerank": False, "use_router": False, "tier": "cheap"},
        "C": {"prompt_variant": "A", "use_rerank": True,  "use_router": False, "tier": "cheap"},
        "D": {"prompt_variant": "A", "use_rerank": False, "use_router": True,  "tier": "cheap"},
        "B+D": {"prompt_variant": "B", "use_rerank": False, "use_router": True,  "tier": "cheap"},
        "C+D": {"prompt_variant": "A", "use_rerank": True,  "use_router": True,  "tier": "cheap"},
        "B+C+D": {"prompt_variant": "B", "use_rerank": True, "use_router": True,  "tier": "cheap"},
    }[req.config]

    gt = get_gt_records()
    qa_records = get_qa_records()
    synth_id_match = next((k for k, v in gt.items() if v.get("source_pdf", "").endswith(req.case_id + ".pdf")), None)
    if not synth_id_match and req.case_id in gt:
        synth_id_match = req.case_id
    if not synth_id_match:
        raise HTTPException(status_code=404, detail=f"case_id {req.case_id!r} not found in SYNTH corpus")

    gt_record = gt[synth_id_match]
    is_refusable = False
    for q in qa_records:
        if q["case_id"] == synth_id_match and q.get("question", "") == req.question:
            is_refusable = q.get("is_refusable", False)
            break
    if not is_refusable and qa_records:
        is_refusable = any(t in req.question.lower() for t in ("bảo hiểm", "tư vấn pháp luật", "insurance"))

    retriever, reranker = get_track2_pipeline()
    retrieved: list[RetrievedChunk] = []
    if retriever is not None:
        try:
            retrieved = retriever.query(req.question, top_k=req.top_k, case_id=req.case_id)
        except Exception as e:
            logger.warning("Retriever query failed: %s; using fallback", e)

    if not retrieved:
        gt_text = _gt_text_for_case(gt_record)
        redactor = get_redactor()
        redacted = redactor.redact_ocr_result(
            {"text_blocks": [{"page": 1, "text": gt_text, "bbox": [0, 0, 100, 100], "confidence": 1.0}]}
        )
        docs = [(req.case_id, redacted.get("redacted_text", ""))]
        chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=40)
        retrieved = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                case_id=c.get("case_id", req.case_id),
                text=c["text"],
                distance=0.0,
            )
            for c in chunks[: req.top_k]
        ]

    chunks_ranked = reranker.rerank(req.question, retrieved) if (cfg_meta["use_rerank"] and reranker is not None) else retrieved

    features = extract_features_track2(req.question, [c.text for c in chunks_ranked], is_refusable)
    routing = route(features, "track2")
    deployment = "gemini-3.5-flash-lite"

    prompt = track2_prompt(cfg_meta["prompt_variant"], req.question, [(c.chunk_id, c.text) for c in chunks_ranked[: req.top_k]])

    wrapper = get_llm_wrapper()
    t0 = time.time()
    if wrapper is not None:
        meta = wrapper.generate(prompt, model=deployment, max_output_tokens=512)
        raw = meta.text or ""
        cost_usd = meta.cost_usd
        latency_ms = meta.latency_ms or ((time.time() - t0) * 1000)
        input_tokens = meta.input_tokens
        output_tokens = meta.output_tokens
        error = meta.error
    else:
        refused = any(t in req.question.lower() for t in ("bảo hiểm", "tư vấn pháp luật", "có thể mua"))
        answer = ""
        citations: list[str] = []
        if not refused and chunks_ranked:
            answer = chunks_ranked[0].text
            citations = [c.chunk_id for c in chunks_ranked[:3]]
        raw = json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "refused": refused,
                "refusal_reason": "out_of_scope" if refused else None,
            },
            ensure_ascii=False,
        )
        input_tokens = len(prompt) // 4
        output_tokens = len(raw) // 4
        latency_ms = (time.time() - t0) * 1000
        cost_usd = 0.0001  # gemini-3.5-flash-lite for all tiers
        error = None

    stripped = _strip_code_fence(raw)
    try:
        parsed = json.loads(stripped) if stripped else {"answer": stripped, "citations": [], "refused": False}
        if not isinstance(parsed, dict):
            parsed = {"answer": stripped, "citations": [], "refused": False}
    except Exception:
        parsed = {"answer": stripped, "citations": [], "refused": False}

    return Track2AskResponse(
        case_id=req.case_id,
        question=req.question,
        answer=str(parsed.get("answer", "")),
        citations=list(parsed.get("citations") or []),
        refused=bool(parsed.get("refused", False)),
        refusal_reason=parsed.get("refusal_reason"),
        config=req.config,
        model=deployment,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_chunks=[
            {"chunk_id": c.chunk_id, "case_id": c.case_id, "distance": c.distance}
            for c in chunks_ranked[: req.top_k]
        ],
        error=error,
    )


@router.get("/configs")
def list_configs() -> dict[str, list[str]]:
    return {"configs": sorted(VALID_CONFIGS)}


@router.get("/cases")
def list_cases() -> dict[str, list[str]]:
    """Return the SYNTH case_ids the API has indexed."""
    cases: list[str] = []
    for synth_id, rec in get_gt_records().items():
        pdf = rec.get("source_pdf", "")
        stem = pdf.split("/")[-1].replace(".pdf", "")
        if stem.startswith("contract_synth-"):
            cases.append(stem)
    return {"cases": sorted(set(cases))}
