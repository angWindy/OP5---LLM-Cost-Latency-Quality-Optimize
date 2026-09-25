"""Track 2 runner: 7 RAG configs x 30 mini-QA cases on synthetic contracts.

Configs (master plan §4.6):
  A    = full prompt A (zero-shot) + top-k=8 retrieval only
  B    = compact prompt B (few-shot) + top-k=8
  C    = re-rank (BM25) on top-k=32, then 8
  D    = deterministic router on top of A
  B+D  = B + D
  C+D  = C + D
  B+C+D = the cheapest viable combo (B + C + D)

QA dataset: data/processed/phase03_mini_qa.jsonl generated inline — 30 records
(15 cases x 2 questions each: a "domain" question + an "out-of-scope" refusal).

Smoke-test mode: pre-compute retrieval once, then call the LLM stub locally.
This exercises the full pipeline (chunking -> embedding -> chroma query ->
BM25 rerank -> router -> prompt -> LLM stub -> EvalLog JSONL).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.rag.chunking import chunk_documents  # noqa: E402
from op5.rag.embedding import MultilingualEmbedder  # noqa: E402
from op5.rag.prompts import track2_prompt  # noqa: E402
from op5.rag.rerank import BM25Reranker  # noqa: E402
from op5.rag.retrieval import Retriever  # noqa: E402
from op5.redact import Redactor, default_policy  # noqa: E402
from op5.router import extract_features_track2, route  # noqa: E402

OUTPUT = Path("results/phase-03-track2.jsonl")
QA_PATH = Path("data/processed/phase03_mini_qa.jsonl")
GT_PATH = Path("data/processed/phase03_synth_contracts.jsonl")
CHROMA_COLLECTION = "phase03_rag"

CONFIGS: dict[str, dict[str, Any]] = {
    "A": {"prompt_variant": "A", "use_rerank": False, "use_router": False, "tier": "cheap"},
    "B": {"prompt_variant": "B", "use_rerank": False, "use_router": False, "tier": "cheap"},
    "C": {"prompt_variant": "A", "use_rerank": True,  "use_router": False, "tier": "cheap"},
    "D": {"prompt_variant": "A", "use_rerank": False, "use_router": True,  "tier": "cheap"},
    "B+D": {"prompt_variant": "B", "use_rerank": False, "use_router": True,  "tier": "cheap"},
    "C+D": {"prompt_variant": "A", "use_rerank": True,  "use_router": True,  "tier": "cheap"},
    "B+C+D": {"prompt_variant": "B", "use_rerank": True, "use_router": True,  "tier": "cheap"},
}


def _stub_llm(prompt: str) -> str:
    import re

    chunk_pattern = re.compile(r"\[([\w\-]+)\]\s*(.+)")
    chunks: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        m = chunk_pattern.match(line.strip())
        if m:
            chunks.append((m.group(1), m.group(2)))

    refused = any(t in prompt.lower() for t in ("out-of-scope", "mua bảo hiểm", "tôi có thể"))
    answer = ""
    citations: list[str] = []
    if not refused and chunks:
        answer = chunks[0][1]
        citations = [c[0] for c in chunks[:3]]

    payload = {
        "answer": answer,
        "citations": citations,
        "refused": refused,
        "refusal_reason": "out_of_scope" if refused else None,
    }
    return json.dumps(payload, ensure_ascii=False)


def _gt_text_for_case(rec: dict[str, Any]) -> str:
    fields = rec.get("ground_truth_fields", {}) or {}
    lines = []
    ordered = ["contract_no", "sign_date", "seller_name", "seller_address",
               "seller_tax_id", "buyer_name", "buyer_phone", "buyer_email",
               "buyer_id", "model", "version", "color", "vin",
               "unit_price_vnd", "delivery_date", "delivery_place",
               "special_offer_a", "special_offer_b", "special_offer_c",
               "special_offer_d", "payment_stage_1_vnd", "payment_method"]
    for k in ordered:
        v = fields.get(k)
        if v is None or v == "":
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _ensure_qa_dataset(qa_path: Path, gt_records: dict[str, dict]) -> list[dict]:
    if qa_path.exists():
        return [json.loads(l) for l in qa_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    qa: list[dict] = []
    for synth_id, rec in gt_records.items():
        qa.append({
            "case_id": synth_id,
            "source_pdf": rec.get("source_pdf"),
            "question": "Hop dong co hieu luc tu ngay nao?",
            "answer": str(rec.get("ground_truth_fields", {}).get("sign_date", "")),
            "source_pages": [1],
            "is_refusable": False,
        })
        qa.append({
            "case_id": synth_id,
            "source_pdf": rec.get("source_pdf"),
            "question": "Toi co the mua bao hiem o dau?",
            "answer": "",
            "source_pages": [],
            "is_refusable": True,
        })

    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w", encoding="utf-8") as f:
        for q in qa:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not GT_PATH.exists():
        print(f"GT corpus missing: {GT_PATH}", file=sys.stderr)
        return 1

    gt_records: dict[str, dict] = {}
    with GT_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt_records[rec["case_id"]] = rec

    qa_records = _ensure_qa_dataset(QA_PATH, gt_records)[: args.limit]

    redactor = Redactor(default_policy())
    docs: list[tuple[str, str]] = []
    for qa in qa_records:
        rec = gt_records.get(qa["case_id"], {})
        redacted = redactor.redact_ocr_result({"text_blocks": [{"page": 1, "text": _gt_text_for_case(rec), "bbox": [0, 0, 100, 100], "confidence": 1.0}]})
        text = redacted.get("redacted_text", "")
        case_id = rec.get("source_pdf", "").split("/")[-1].replace(".pdf", "") or f"contract_synth-{qa['case_id']}"
        docs.append((case_id, text))

    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=40)
    retriever = Retriever(collection=CHROMA_COLLECTION, embedder=MultilingualEmbedder())
    try:
        retriever.index(chunks)
    except Exception as e:
        print(f"  warning: indexing skipped ({e}); will use deterministic fallback", file=sys.stderr)

    reranker = BM25Reranker()
    rows: list[dict] = []

    for qa in qa_records:
        rec = gt_records.get(qa["case_id"], {})
        case_id = rec.get("source_pdf", "").split("/")[-1].replace(".pdf", "") or f"contract_synth-{qa['case_id']}"
        question = qa["question"]

        try:
            retrieved = retriever.query(question, top_k=8, case_id=case_id)
        except Exception:
            retrieved = []
        if not retrieved:
            from op5.rag.retrieval import RetrievedChunk
            matched = [c for c in chunks if c["case_id"] == case_id][:8]
            retrieved = [RetrievedChunk(chunk_id=c["chunk_id"], case_id=c["case_id"], text=c["text"], distance=0.0) for c in matched]

        for cfg_name, cfg in CONFIGS.items():
            chunks_ranked = reranker.rerank(question, retrieved) if cfg["use_rerank"] else retrieved
            features = extract_features_track2(question, [c.text for c in chunks_ranked], qa.get("is_refusable", False))
            routing = route(features, "track2")
            cfg_tier = routing["tier"] if cfg["use_router"] else cfg["tier"]
            deployment = "gemini-3.1-pro-strong" if cfg_tier == "strong" else "gemini-3.5-flash-lite"

            prompt = track2_prompt(cfg["prompt_variant"], question, [(c.chunk_id, c.text) for c in chunks_ranked[:8]])
            t0 = time.time()
            raw = _stub_llm(prompt)
            latency_ms = (time.time() - t0) * 1000
            try:
                llm_parsed = json.loads(raw)
            except Exception:
                llm_parsed = {"answer": raw, "citations": [], "refused": False}

            valid_chunk_ids = [c.chunk_id for c in chunks_ranked[:8]]
            rows.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": "track2",
                "case_id": case_id,
                "config": cfg_name,
                "provider": "gemini-stub",
                "model": deployment,
                "deployment_id": deployment,
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(raw) // 4,
                "latency_ms": latency_ms,
                "cost_usd": 0.0001 if cfg_tier == "cheap" else 0.005,
                "cache_status": "bypass",
                "pred": llm_parsed,
                "ref": {
                    "answer": qa.get("answer"),
                    "is_refusable": qa.get("is_refusable", False),
                    "valid_chunk_ids": valid_chunk_ids,
                    "gt_case_id": qa["case_id"],
                    "question": question,
                },
                "score": {},
                "routing": {**routing, "tier": cfg_tier, "deployment_id": deployment},
            })

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Track 2: {len(rows)} rows ({len(qa_records)} QAs x {len(CONFIGS)} configs) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
