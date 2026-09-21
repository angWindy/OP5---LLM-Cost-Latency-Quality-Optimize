#!/usr/bin/env python3
"""
Smoke test: verify HuggingFace token can authenticate and read ZeroSCROLLS dataset.

Purpose: Confirm HF_TOKEN is set correctly and the dataset is accessible before
running the full PoC harness. Run BEFORE the PoC scripts to fail fast on auth.

Default dataset is now tau/zero_scrolls (was zai-org/LongBench-v2).
Use --dataset longbench_v2 to test the legacy path.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/smoke_huggingface.py [--dataset zero_scrolls]
"""
from __future__ import annotations

import argparse
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

# Dataset registry (keep in sync with _common.DATASETS)
DATASET_REGISTRY = {
    "zero_scrolls": "tau/zero_scrolls",
    "longbench_v2": "zai-org/LongBench-v2",
}
DEFAULT_DATASET = "zero_scrolls"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test HF dataset access")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASET_REGISTRY.keys()),
                        help=f"Dataset key (default {DEFAULT_DATASET})")
    args = parser.parse_args()

    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_DATASET = DATASET_REGISTRY[args.dataset]

    # ZeroSCROLLS là public → không bắt buộc HF_TOKEN, chỉ cần cho legacy LongBench-v2
    if not HF_TOKEN and args.dataset == "longbench_v2":
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

    print(f"Dataset: {args.dataset}  ({HF_DATASET})")
    print(f"Token:   {HF_TOKEN[:4]}...{HF_TOKEN[-4:] if HF_TOKEN else '(empty)'}")
    print("---")

    # 1. Verify token is valid (nếu có)
    if HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        try:
            who = api.whoami()
            print(f"Authenticated as: {who.get('name', who.get('id', '?'))} ({who.get('email', 'no email')})")
        except Exception as exc:
            print(f"ERROR: Token khong hop le hoac bi thu hoi: {exc}", file=sys.stderr)
            return 4
    else:
        print("No HF_TOKEN set — chỉ kiểm tra public access (OK cho ZeroSCROLLS).")

    # 2. Check dataset is accessible (info call, no full download)
    try:
        if HF_TOKEN:
            info = api.dataset_info(repo_id=HF_DATASET, token=HF_TOKEN)
        else:
            from huggingface_hub import HfApi as _HfApi
            info = _HfApi().dataset_info(repo_id=HF_DATASET)
        print(f"Dataset: {info.id}")
        print(f"  Files: {[s.rfilename for s in info.siblings][:10]}")
        print(f"  Gated: {info.gated}")
    except AttributeError as exc:
        print(f"ERROR: Khong truy cap duoc dataset '{HF_DATASET}': {exc}", file=sys.stderr)
        return 5

    print("---")
    print("OK Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
