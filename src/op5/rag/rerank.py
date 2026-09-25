"""BM25 reranker (rank-bm25) for Phase 03 RAG."""

from __future__ import annotations

from op5.rag.retrieval import RetrievedChunk


class BM25Reranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re as _re

        return _re.findall(r"\w+", text.lower(), flags=_re.UNICODE)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            return chunks

        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        scores = bm25.get_scores(self._tokenize(query))
        scored = sorted(zip(scores, chunks), key=lambda x: -x[0])
        out: list[RetrievedChunk] = []
        for s, c in scored:
            c.distance = float(-s)
            out.append(c)
        return out
