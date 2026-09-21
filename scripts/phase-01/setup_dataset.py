#!/usr/bin/env python3
"""
Download LongBench-v2 via HuggingFace, save first 100 rows, then split deterministically:
  - 5 rows  -> data/processed/llmlingua_test5.jsonl  (PoC sanity, read-only after creation)
  - 95 rows -> data/processed/dev_first95.jsonl      (pool cho eval mo rong sau)

Use seed=42 de split co the lap lai.

Requires env HF_TOKEN. Run after `verify_libs.py`.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/setup_dataset.py [--n_total 100]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    print("WARNING: python-dotenv not installed.", file=sys.stderr)


DATASET_NAME = "zai-org/LongBench-v2"
DEFAULT_N_TOTAL = 100
DEFAULT_TEST5_SEED = 42
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"


def download_rows(n: int, split: str = "train") -> list[dict]:
    """Stream first N rows from HF dataset, return list of dicts."""
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN") or True  # fallback to public access
    print(f"[download] Streaming {DATASET_NAME} split='{split}' n={n} ...")
    ds = load_dataset(DATASET_NAME, split=split, streaming=True, token=token)
    rows: list[dict] = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if (i + 1) % 10 == 0:
            print(f"  ...got {i+1} rows")
        if i + 1 >= n:
            break
    print(f"[download] Got {len(rows)} rows total")
    return rows


def split_5_95(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle, first 5 -> test, rest -> dev."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    test_idx = set(idx[:5])
    test_rows = [rows[i] for i in range(len(rows)) if i in test_idx]
    dev_rows = [rows[i] for i in range(len(rows)) if i not in test_idx]
    return test_rows, dev_rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {len(rows)} rows -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LongBench-v2 + split 5/95")
    parser.add_argument("--n_total", type=int, default=DEFAULT_N_TOTAL,
                        help=f"Total rows to download (default {DEFAULT_N_TOTAL})")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default 'train')")
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST5_SEED,
                        help=f"Shuffle seed (default {DEFAULT_TEST5_SEED})")
    args = parser.parse_args()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_path = RAW_DIR / f"longbench_v2_first{args.n_total}_{date_str}.jsonl"
    test_path = PROC_DIR / "llmlingua_test5.jsonl"
    dev_path = PROC_DIR / "dev_first95.jsonl"

    print("=== Dataset setup ===")
    print(f"  dataset:    {DATASET_NAME}")
    print(f"  split:      {args.split}")
    print(f"  n_total:    {args.n_total}")
    print(f"  seed:       {args.seed}")
    print(f"  raw_path:   {raw_path}")
    print(f"  test_path:  {test_path}  (5 rows)")
    print(f"  dev_path:   {dev_path}   (95 rows)")
    print()

    rows = download_rows(args.n_total, split=args.split)
    if len(rows) < 100:
        print(f"WARNING: got only {len(rows)} rows (need 100)", file=sys.stderr)

    write_jsonl(rows, raw_path)
    test_rows, dev_rows = split_5_95(rows, seed=args.seed)
    write_jsonl(test_rows, test_path)
    write_jsonl(dev_rows, dev_path)

    print()
    print("=== Schema of first test row ===")
    if test_rows:
        first = test_rows[0]
        for k, v in first.items():
            preview = repr(v)[:80]
            print(f"  {k:25s} | {type(v).__name__:10s} | {preview}")

    print()
    print("OK Setup done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
