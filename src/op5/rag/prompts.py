"""Prompt templates for Phase 03 RAG (Track 2) + extraction (Track 1).

Two prompt variants per track:
- A: zero-shot (no examples).
- B: few-shot (one example appended).
"""

from __future__ import annotations

TRACK1_PROMPT_A = """You are a contract field extractor. Read the OCR text below and extract the following fields.
Output strict JSON with EXACTLY these keys (use null if not found):

{fields}

Contract text:
\"\"\"
{redacted_text}
\"\"\"

Output schema:
{{"{schema}"}}

Output JSON only, no prose."""

TRACK1_PROMPT_B = """You are a contract field extractor. Read the OCR text below and extract the following fields.
Output strict JSON with EXACTLY these keys (use null if not found).

Example:
Input text: "Hop dong so ABC ky ngay 2026-01-15 giua ong Nguyen Van A va Cong ty VINFAST. Gia: 700000000 VND."
Output: {{"contract_no": "ABC", "sign_date": "2026-01-15", "buyer_name": "Nguyen Van A", "total_price_vnd": 700000000}}

Now extract from this contract:
{fields}

Contract text:
\"\"\"
{redacted_text}
\"\"\"

Output schema:
{{"{schema}"}}

Output JSON only, no prose."""


def track1_prompt(variant: str, fields: list[str], redacted_text: str) -> str:
    schema = '", "'.join(fields)
    schema = '"' + schema + '"'
    if variant.upper() == "A":
        return TRACK1_PROMPT_A.format(fields=", ".join(fields), redacted_text=redacted_text, schema=schema)
    return TRACK1_PROMPT_B.format(fields=", ".join(fields), redacted_text=redacted_text, schema=schema)


TRACK2_PROMPT_A = """You are a contract Q&A assistant. Use ONLY the retrieved context below to answer the question.
Cite the chunk IDs in square brackets at the end of each fact, e.g. [case-001-c003].
If the answer is not in the context, respond: "I cannot answer this from the provided documents."
If the question is out-of-scope (legal advice, financial advice, personal opinions), refuse politely.

Context:
{chunks}

Question: {question}

Output JSON:
{{"answer": "...", "citations": ["chunk-id-1", "..."], "refused": false, "refusal_reason": null}}"""


TRACK2_PROMPT_B = """You are a contract Q&A assistant. Use ONLY the retrieved context below to answer the question.

Example:
Context: [case-001-c000] Hop dong so ABC ky ngay 2026-01-15.
Question: "Hop dong co hieu luc tu ngay nao?"
Output: {{"answer": "Hop dong co hieu luc tu ngay 2026-01-15.", "citations": ["case-001-c000"], "refused": false, "refusal_reason": null}}

Now answer:

Context:
{chunks}

Question: {question}

Output JSON:
{{"answer": "...", "citations": ["chunk-id-1", "..."], "refused": false, "refusal_reason": null}}"""


def track2_prompt(variant: str, question: str, chunks: list[tuple[str, str]]) -> str:
    chunk_text = "\n\n".join(f"[{cid}] {text}" for cid, text in chunks)
    if variant.upper() == "A":
        return TRACK2_PROMPT_A.format(question=question, chunks=chunk_text)
    return TRACK2_PROMPT_B.format(question=question, chunks=chunk_text)
