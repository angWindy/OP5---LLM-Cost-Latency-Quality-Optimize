#!/usr/bin/env python3
"""
Smoke test v2: run LLMLingua-2 compression on 1 row from llmlingua_test5.jsonl through
both paths and compare (using CORRECT API per LLMLingua series documentation):

  - Path A: pip-direct `PromptCompressor(use_llmlingua2=True)`
  - Path B: LangChain wrapper over the same pip-direct compressor

Key fixes vs v1:
  - Pass `use_llmlingua2=True` to activate the LLMLingua-2 model
    (without this flag, PromptCompressor uses the original LLMLingua model
    which gave ratio=100% on our data)
  - Do NOT manually chunk — compress_prompt has internal iterative_size=200
    chunking. Passing pre-chunked text causes position-boundary truncation.
  - Use `force_tokens=["!", ".", "?", "\n"]` to preserve sentence structure
    (required for MC-QA tasks where A./B./C./D. matter)

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/smoke_both_paths.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


# Default LLMLingua-2 config from the paper (MeetingBank, rate=0.33)
DEFAULT_RATE = 0.33
DEFAULT_FORCE_TOKENS = ["!", ".", "?", "\n"]
COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


# ---------------------------------------------------------------------------
# Shared compressor instance (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_llmlingua2_pc = None


def _get_llmlingua2() -> "PromptCompressor":
    """Lazily create and cache the LLMLingua-2 PromptCompressor."""
    global _llmlingua2_pc
    if _llmlingua2_pc is None:
        from llmlingua import PromptCompressor
        _llmlingua2_pc = PromptCompressor(
            model_name=COMPRESSOR_MODEL,
            use_llmlingua2=True,  # <-- THE CRITICAL FLAG
            device_map="cpu",
        )
    return _llmlingua2_pc


# ---------------------------------------------------------------------------
# Path A: pip-direct PromptCompressor
# ---------------------------------------------------------------------------
def compress_pip_direct(
    context: str,
    question: str = "",
    rate: float = DEFAULT_RATE,
    force_tokens: list[str] | None = None,
) -> tuple[str, float, dict]:
    """
    Path A: LLMLingua-2 via pip-direct PromptCompressor.
    Does NOT chunk manually — compress_prompt handles that internally.
    Returns (compressed_text, elapsed_ms, meta_dict).
    """
    pc = _get_llmlingua2()
    ft = force_tokens or DEFAULT_FORCE_TOKENS
    t0 = time.perf_counter()
    result = pc.compress_prompt(
        context,
        question=question,
        rate=rate,
        force_tokens=ft,
        drop_consecutive=True,
        return_word_label=False,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    compressed = result["compressed_prompt"]
    meta = {
        "origin_tokens": result.get("origin_tokens"),
        "compressed_tokens": result.get("compressed_tokens"),
        "ratio_str": result.get("ratio"),  # e.g. "3.0x"
        "origin_chars": len(context),
        "compressed_chars": len(compressed),
    }
    return compressed, elapsed, meta


# ---------------------------------------------------------------------------
# Path B: LangChain-compatible wrapper around the same PromptCompressor
# ---------------------------------------------------------------------------
class LLMLingua2Compressor:
    """
    A minimal LangChain-compatible document compressor that wraps LLMLingua-2
    (pip-direct). The official `LLMLinguaCompressor` in langchain-community
    does NOT pass `use_llmlingua2=True`, so it uses the original LLMLingua
    model and crashes with AttributeError on newer transformers versions.
    This wrapper fixes both issues.
    """

    def __init__(
        self,
        model_name: str = COMPRESSOR_MODEL,
        rate: float = DEFAULT_RATE,
        force_tokens: list[str] | None = None,
        device_map: str = "cpu",
    ):
        from llmlingua import PromptCompressor

        self.pc = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,  # <-- the fix
            device_map=device_map,
        )
        self.rate = rate
        self.force_tokens = force_tokens or DEFAULT_FORCE_TOKENS

    def compress_documents(self, documents, query: str, callbacks=None):
        """Compress a list of Document objects, question-aware."""
        if not documents:
            return []

        # Format with ref markers (same pattern as langchain-community impl)
        formatted = []
        for i, doc in enumerate(documents):
            content = doc.page_content.replace("\n\n", "\n")
            formatted.append(f"\n\n<#ref{i}#> {content} <#ref{i}#>\n\n")

        context_str = "".join(formatted)
        t0 = time.perf_counter()
        result = self.pc.compress_prompt(
            context_str,
            question=query,
            rate=self.rate,
            force_tokens=self.force_tokens,
            drop_consecutive=True,
            return_word_label=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        compressed = result["compressed_prompt"]

        # Split back on ref markers and clean up
        parts = re.split(r"<#ref\d+#?>", compressed)
        compressed_docs = []
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                from langchain_core.documents import Document

                compressed_docs.append(Document(page_content=cleaned))

        # Store compression metadata on the first doc's metadata
        if compressed_docs:
            compressed_docs[0].metadata["llmlingua2_meta"] = {
                "origin_tokens": result.get("origin_tokens"),
                "compressed_tokens": result.get("compressed_tokens"),
                "ratio_str": result.get("ratio"),
                "compressor_ms": elapsed,
            }
        return compressed_docs

    def compress(self, context: str, question: str = "") -> tuple[str, float, dict]:
        """Standalone compression of a plain string (not Documents)."""
        t0 = time.perf_counter()
        result = self.pc.compress_prompt(
            context,
            question=question,
            rate=self.rate,
            force_tokens=self.force_tokens,
            drop_consecutive=True,
            return_word_label=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        meta = {
            "origin_tokens": result.get("origin_tokens"),
            "compressed_tokens": result.get("compressed_tokens"),
            "ratio_str": result.get("ratio"),
        }
        return result["compressed_prompt"], elapsed, meta


def compress_langchain(
    context: str,
    question: str = "",
    rate: float = DEFAULT_RATE,
    force_tokens: list[str] | None = None,
) -> tuple[str, float, dict]:
    """
    Path B: LLMLingua-2 via our custom LangChain-compatible wrapper.
    Returns (compressed_text, elapsed_ms, meta_dict).
    """
    # Simulate "one doc per page" so the wrapper's ref markers work
    class _Doc:
        def __init__(self, text):
            self.page_content = text
            self.metadata = {}

    wrapper = LLMLingua2Compressor(rate=rate, force_tokens=force_tokens)
    docs = [_Doc(context)]
    compressed_docs = wrapper.compress_documents(docs, query=question)
    # The wrapper stores meta on the first doc
    meta = (compressed_docs[0].metadata.get("llmlingua2_meta") or {}) if compressed_docs else {}
    compressed = " ".join(d.page_content for d in compressed_docs)
    elapsed = meta.get("compressor_ms", 0.0)
    return compressed, elapsed, meta


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def chunk_context_for_info_only(context: str, chunk_chars: int = 1500) -> list[str]:
    """
    Only used for DISPLAY purposes (context char count estimation).
    NOT passed to compress_prompt — that function handles its own internal chunking.
    """
    if len(context) <= chunk_chars:
        return [context]
    return [context[i:i + chunk_chars] for i in range(0, len(context), chunk_chars)]


def print_snippet(label: str, text: str, limit: int = 220) -> None:
    s = text.replace("\n", " ")
    print(f"  {label}: {s[:limit]}{'...' if len(s) > limit else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    test_path = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run `setup_dataset.py` first.", file=sys.stderr)
        return 2

    import json

    with test_path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        print("ERROR: test file empty.", file=sys.stderr)
        return 3

    # Smoke on the SHORTEST row so it finishes quickly
    rows.sort(key=lambda r: len(r.get("context", "")))
    row = rows[0]
    context = row.get("context", "")
    question = row.get("question", "")
    approx_chunks = len(chunk_context_for_info_only(context))

    print("=== Smoke v2: 1 row, both paths (correct API) ===")
    print(f"  case:   _id={row.get('_id', '?')}")
    print(f"  domain: {row.get('domain', '?')} / {row.get('sub_domain', '?')}")
    print(f"  q:      {question[:120]}")
    print(f"  gold:   {row.get('answer', '?')}")
    print(f"  context: {len(context):,} chars (≈{approx_chunks} chunks @ 1500 — NOT passed to compress)")
    print(f"  config:  rate={DEFAULT_RATE}, force_tokens={DEFAULT_FORCE_TOKENS}")
    print()

    # Path A
    print("--- Path A: pip-direct PromptCompressor(use_llmlingua2=True) ---")
    try:
        ca, ta, ma = compress_pip_direct(context, question)
        ratio_a = ma.get("ratio_str", f"{len(ca)/max(len(context),1)*100:.1f}%")
        print(f"  origin_tokens : {ma.get('origin_tokens')} -> compressed_tokens: {ma.get('compressed_tokens')}")
        print(f"  ratio_str    : {ratio_a}  (from paper, not char-based)")
        print(f"  elapsed      : {ta:.0f} ms")
        print(f"  chars in/out : {ma.get('origin_chars'):,} -> {ma.get('compressed_chars'):,}")
        print_snippet("preview", ca)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"  Path A crashed: {exc}")
    print()

    # Path B
    print("--- Path B: LangChain wrapper (LLMLingua2Compressor) ---")
    try:
        cb, tb, mb = compress_langchain(context, question)
        ratio_b = mb.get("ratio_str", f"{len(cb)/max(len(context),1)*100:.1f}%")
        print(f"  origin_tokens : {mb.get('origin_tokens')} -> compressed_tokens: {mb.get('compressed_tokens')}")
        print(f"  ratio_str    : {ratio_b}")
        print(f"  elapsed      : {tb:.0f} ms")
        print(f"  chars in/out : {len(context):,} -> {len(cb):,}")
        print_snippet("preview", cb)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"  Path B crashed: {exc}")
    print()

    print("OK Smoke v2 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
