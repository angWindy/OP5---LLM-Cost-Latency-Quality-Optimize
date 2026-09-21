#!/usr/bin/env python3
"""
Smoke test: 1 basic Gemini request.

Purpose: Verify GOOGLE_API_KEY is set, the model `gemini-3.5-flash-lite` is available,
and we can read a response end-to-end. Run BEFORE the full PoC harness to fail fast
on auth / model-name issues.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/smoke_gemini.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load .env from repo root (script lives in scripts/phase-01/)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    print("WARNING: python-dotenv not installed. Falling back to os.environ only.",
          file=sys.stderr)

MODEL_NAME = os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite")
PROMPT = (
    "Trong mot cau, ban hieu gi viec nen prompt (prompt compression) trong LLM?"
)


def main() -> int:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY chua duoc set. Them vao file .env o repo root.",
              file=sys.stderr)
        return 2

    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai chua duoc cai. "
              "Chay: pip install google-generativeai", file=sys.stderr)
        return 3

    genai.configure(api_key=api_key)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT!r}")
    print("---")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as exc:
        print(f"ERROR: Khong tao duoc model '{MODEL_NAME}': {exc}", file=sys.stderr)
        return 4

    t0 = time.perf_counter()
    try:
        response = model.generate_content(
            PROMPT,
            generation_config={"temperature": 0.0, "max_output_tokens": 256},
        )
    except Exception as exc:
        print(f"ERROR: Request failed: {exc}", file=sys.stderr)
        return 5
    elapsed_ms = (time.perf_counter() - t0) * 1000

    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)

    print(f"Response ({elapsed_ms:.0f} ms):")
    print(text)
    print("---")
    if usage is not None:
        print("Usage:")
        print(f"  prompt_tokens     = {getattr(usage, 'prompt_token_count', '?')}")
        print(f"  completion_tokens = {getattr(usage, 'candidates_token_count', '?')}")
        print(f"  total_tokens      = {getattr(usage, 'total_token_count', '?')}")
    else:
        print("Usage metadata khong co trong response (OK, khong bat buoc).")

    print("---")
    print("OK Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
