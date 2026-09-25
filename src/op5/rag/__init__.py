"""RAG skeleton for Phase 03 Track 2."""

from op5.rag.chunking import chunk_documents, recursive_split
from op5.rag.embedding import EMBED_DIM, MultilingualEmbedder
from op5.rag.pipeline import RAGAnswer, RAGPipeline, build_default_pipeline
from op5.rag.prompts import track1_prompt, track2_prompt
from op5.rag.rerank import BM25Reranker
from op5.rag.retrieval import Retriever, RetrievedChunk

__all__ = [
    "BM25Reranker",
    "EMBED_DIM",
    "MultilingualEmbedder",
    "RAGAnswer",
    "RAGPipeline",
    "RetrievedChunk",
    "Retriever",
    "build_default_pipeline",
    "chunk_documents",
    "recursive_split",
    "track1_prompt",
    "track2_prompt",
]
