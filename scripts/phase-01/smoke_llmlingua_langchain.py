#!/usr/bin/env python3
"""
Quick test: LangChain + LLMLingua on ~10 LongBench-v2 samples.
Show before/after context length, compression ratio, and a preview snippet.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/smoke_llmlingua_langchain.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN", "")


def load_samples(n: int = 10) -> list[dict]:
    """Load N samples from cached LongBench-v2 JSON."""
    cache = Path("/tmp/longbench_full.json")
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} rows from cache")
        return data[:n]

    from datasets import load_dataset

    print(f"Downloading {n} rows from HuggingFace...")
    ds = load_dataset("THUDM/LongBench-v2", split="test", token=HF_TOKEN)
    return [ds[i] for i in range(min(n, len(ds)))]


def main() -> int:
    # Use LLMLingua PromptCompressor directly (LangChain wrapper has accelerate bug)
    print("Loading LLMLingua compressor (direct PromptCompressor, bypassing LangChain wrapper)...")
    t_load = time.perf_counter()
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    try:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu",
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"ERROR: Cannot init PromptCompressor: {exc!r}", file=sys.stderr)
        return 4
    print(f"Compressor loaded in {(time.perf_counter()-t_load)*1000:.0f} ms\n")

    # 2. Load samples
    samples = load_samples(10)

    # 3. Compress each and report
    total_orig = 0
    total_comp = 0
    total_time = 0.0

    print(f"{'#':<3} {'orig_chars':>10} {'comp_chars':>10} {'ratio%':>7} {'ms':>6}  question snippet")
    print("-" * 80)

    for i, sample in enumerate(samples):
        context: str = sample.get("context", "")
        question: str = sample.get("question", "")
        answer: str = sample.get("answer", "")

        # LongBench-v2 contexts can be >500k chars (~125k tokens).
        # LLMLingua's xlm-roberta is 512-token max; chunk into ~1500-char pieces
        # (~375 tokens) so the per-chunk forward pass fits.
        CHUNK_CHARS = 1500
        if len(context) > CHUNK_CHARS:
            context_chunks = [context[i : i + CHUNK_CHARS] for i in range(0, len(context), CHUNK_CHARS)]
        else:
            context_chunks = [context]

        t0 = time.perf_counter()
        comp_texts = []
        try:
            for ci, chunk in enumerate(context_chunks):
                try:
                    result = compressor.compress_prompt(
                        chunk,
                        question=question,
                        rate=0.5,
                    )
                    comp_texts.append(result.get("compressed_prompt", ""))
                except Exception as chunk_err:
                    # Skip bad chunks; keep going for the rest.
                    print(f"[{i+1}] chunk#{ci} skipped: {chunk_err}")
                    continue
            comp_text = " ".join(t for t in comp_texts if t)
            if not comp_text:
                raise RuntimeError("all chunks failed")
        except Exception as exc:
            print(f"[{i+1}] ERROR compression failed: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        orig_len = len(context)
        comp_len = len(comp_text)
        ratio = comp_len / max(orig_len, 1) * 100

        total_orig += orig_len
        total_comp += comp_len
        total_time += elapsed_ms

        q_snippet = question[:55].replace("\n", " ")
        print(
            f"{i+1:<3} {orig_len:>10,} {comp_len:>10,} {ratio:>6.1f}% {elapsed_ms:>6.0f}  {q_snippet}..."
        )

    print("-" * 80)
    if total_orig > 0:
        overall_ratio = total_comp / total_orig * 100
        avg_time = total_time / max(len(samples), 1)
        print(f"{'TOTAL':<3} {total_orig:>10,} {total_comp:>10,} {overall_ratio:>6.1f}% {total_time:>6.0f} ms")
        print(f"\nAvg compression time per sample: {avg_time:.0f} ms")
        print(f"Overall compression ratio: {overall_ratio:.1f}%")

    print("\nOK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
