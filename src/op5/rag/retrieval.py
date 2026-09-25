"""Retrieval (ChromaDB cosine top-k) for Phase 03 RAG."""

from __future__ import annotations

from dataclasses import dataclass

from op5.rag.embedding import MultilingualEmbedder
from op5.storage import STORAGE, EmbeddingItem


@dataclass
class RetrievedChunk:
    chunk_id: str
    case_id: str
    text: str
    distance: float | None


class Retriever:
    def __init__(self, collection: str = "phase03_rag", embedder: MultilingualEmbedder | None = None) -> None:
        self.collection = collection
        self.embedder = embedder or MultilingualEmbedder()
        self._storage = STORAGE()

    def index(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)
        items = [
            EmbeddingItem(
                id=c["chunk_id"],
                embedding=vec,
                document=c["text"],
                metadata={"case_id": c.get("case_id", "")},
            )
            for c, vec in zip(chunks, vectors)
        ]
        self._storage.chroma().add(self.collection, items)

    def query(self, question: str, top_k: int = 8, case_id: str | None = None) -> list[RetrievedChunk]:
        qvec = self.embedder.embed_query(question)
        where = {"case_id": case_id} if case_id else None
        raw = self._storage.chroma().query(self.collection, qvec, top_k=top_k, where=where)
        return [
            RetrievedChunk(
                chunk_id=r["id"],
                case_id=r.get("metadata", {}).get("case_id", ""),
                text=r["document"],
                distance=r.get("distance"),
            )
            for r in raw
        ]
