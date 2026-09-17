#!/usr/bin/env python3
"""
Smoke test: verify HuggingFace token can authenticate and read LongBench-v2.

Purpose: Confirm HF_TOKEN is set correctly and the dataset is accessible before
running the full PoC harness. Run BEFORE the PoC scripts to fail fast on auth.

Usage (from repo root):
    conda activate po5
    python scripts/phase-01/smoke_huggingface.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from repo root
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    print(
        "WARNING: python-dotenv not installed. Falling back to os.environ only.",
        file=sys.stderr,
    )

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET = "THUDM/LongBench-v2"


def main() -> int:
    if not HF_TOKEN:
        print(
            "ERROR: HF_TOKEN chua duoc set. Them vao file .env o repo root.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERROR: huggingface_hub chua duoc cai. "
            "Chay: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 3

    print(f"Dataset: {HF_DATASET}")
    print(f"Token:   {HF_TOKEN[:4]}...{HF_TOKEN[-4:]} (masked)")
    print("---")

    # 1. Verify token is valid
    api = HfApi(token=HF_TOKEN)
    try:
        who = api.whoami()
        print(f"Authenticated as: {who.get('name', who.get('id', '?'))} ({who.get('email', 'no email')})")
    except Exception as exc:
        print(f"ERROR: Token khong hop le hoac bi thu hoi: {exc}", file=sys.stderr)
        return 4

    # 2. Check dataset is accessible (info call, no full download)
    try:
        info = api.dataset_info(repo_id=HF_DATASET, token=HF_TOKEN)
        print(f"Dataset: {info.id}")
        print(f"  Files: {[s.path for s in info.siblings]}")
        print(f"  Gated: {info.gated}")
    except Exception as exc:
        print(f"ERROR: Khong truy cap duoc dataset '{HF_DATASET}': {exc}", file=sys.stderr)
        return 5

    print("---")
    print("OK Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
