"""Chunking strategies for Phase 03 RAG.

Provides a simple RecursiveCharacterTextSplitter-compatible interface without
needing langchain.
"""

from __future__ import annotations

import re


def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50, separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")) -> list[str]:
    """Recursively split text using the given separator list, preserving order.

    chunk_size is in characters (approximate; we aim for chunk_size +/- 25%).
    """
    if not text or chunk_size <= 0:
        return [text] if text else []

    chunks: list[str] = []
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    if sep == "":
        pieces = list(text)
    else:
        parts = text.split(sep)
        pieces = []
        for i, p in enumerate(parts):
            if i > 0:
                pieces.append(sep)
            pieces.append(p)

    current = ""
    for piece in pieces:
        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size and len(separators) > 1:
                sub = recursive_split(piece, chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators[1:])
                if current and chunks:
                    chunks[-1] = chunks[-1] + sub[0][-chunk_overlap:]
                chunks.extend(sub)
                current = ""
            else:
                if len(piece) > chunk_size:
                    chunks.append(piece[:chunk_size])
                    current = piece[chunk_size:]
                else:
                    current = piece
    if current:
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = chunks[i - 1]
                tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                overlapped.append(tail + ch)
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def chunk_documents(docs: list[tuple[str, str]], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    out: list[dict] = []
    for case_id, text in docs:
        for i, c in enumerate(recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            out.append({"case_id": case_id, "chunk_id": f"{case_id}-c{i:03d}", "text": c})
    return out
