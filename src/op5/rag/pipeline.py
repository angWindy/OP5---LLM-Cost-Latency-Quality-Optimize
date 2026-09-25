"""Top-level RAG pipeline for Phase 03 Track 2.

Pipeline: question -> embed_query -> chroma top-k -> BM25 rerank -> LLM prompt + call -> JSON.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Callable

from op5.rag.prompts import track2_prompt
from op5.rag.rerank import BM25Reranker
from op5.rag.retrieval import Retriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    chunks: list[RetrievedChunk]
    raw_llm_response: str | None = None


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, reranker: BM25Reranker | None = None, llm_call: Callable[[str], str] | None = None, top_k: int = 8) -> None:
        self.retriever = retriever or Retriever()
        self.reranker = reranker or BM25Reranker()
        self.llm_call = llm_call
        self.top_k = top_k

    def ask(self, question: str, case_id: str | None = None, variant: str = "A") -> RAGAnswer:
        chunks = self.retriever.query(question, top_k=self.top_k, case_id=case_id)
        chunks = self.reranker.rerank(question, chunks)
        prompt = track2_prompt(variant, question, [(c.chunk_id, c.text) for c in chunks])

        if self.llm_call is None:
            return RAGAnswer(
                question=question,
                answer="[no-llm-set]",
                citations=[c.chunk_id for c in chunks[:3]],
                refused=False,
                refusal_reason=None,
                chunks=chunks,
            )

        raw = self.llm_call(prompt)
        answer = raw
        citations: list[str] = []
        refused = False
        refusal_reason = None
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
            citations = list(parsed.get("citations", []))
            refused = bool(parsed.get("refused", False))
            refusal_reason = parsed.get("refusal_reason")
        except Exception:
            pass
        return RAGAnswer(
            question=question,
            answer=answer,
            citations=citations,
            refused=refused,
            refusal_reason=refusal_reason,
            chunks=chunks,
            raw_llm_response=raw,
        )


def build_default_pipeline(llm_call: Callable[[str], str] | None = None) -> RAGPipeline:
    return RAGPipeline(llm_call=llm_call)
