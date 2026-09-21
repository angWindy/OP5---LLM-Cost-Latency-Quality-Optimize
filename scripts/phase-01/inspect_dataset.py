#!/usr/bin/env python3
"""
Inspect LongBench-v2 dataset structure.

Stream first N rows (default 3) to understand the schema: which fields contain the
passage, the question, the answer, the task type, etc. We need this BEFORE writing
the PoC harness.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/inspect_dataset.py --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

DATASET_NAME = "zai-org/LongBench-v2"
DEFAULT_N = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LongBench-v2 dataset")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of rows to inspect (default {DEFAULT_N})")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default 'train')")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets chua duoc cai. pip install datasets", file=sys.stderr)
        return 3

    print(f"Loading dataset: {DATASET_NAME}")
    print(f"  split:  {args.split}")
    print(f"  n:      {args.n}")
    print("---")

    try:
        ds = load_dataset(DATASET_NAME, split=args.split, streaming=True)
    except Exception as exc:
        print(f"ERROR: Khong load duoc dataset: {exc}", file=sys.stderr)
        return 4

    try:
        first_row = next(iter(ds))
    except StopIteration:
        print("ERROR: Dataset rong.", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ERROR: Khong doc duoc row dau: {exc}", file=sys.stderr)
        return 6

    print("First row (field summary):")
    for key, value in first_row.items():
        preview = repr(value)[:150]
        print(f"  {key:30s} | {type(value).__name__:10s} | {preview}")
    print("---")

    print("First row as JSON (truncated):")
    print(json.dumps(_summarize_row(first_row), indent=2, ensure_ascii=False, default=str))
    print("---")

    print(f"Streaming next {args.n - 1} rows to check schema stability...")
    ds_iter = iter(load_dataset(DATASET_NAME, split=args.split, streaming=True))
    next(ds_iter, None)
    for i in range(args.n - 1):
        try:
            row = next(ds_iter)
        except StopIteration:
            print(f"  (het sau {i} rows)")
            break
        keys_match = set(row.keys()) == set(first_row.keys())
        print(f"  Row {i+2}: keys_match={keys_match} | num_fields={len(row)}")

    print("---")
    print("OK Inspect done. Note field names de dung trong poc_track1/track2.")
    return 0


def _summarize_row(row: dict, max_str_len: int = 400) -> dict:
    """Return a JSON-serializable summary with long strings truncated."""
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            out[k] = v if len(v) <= max_str_len else v[:max_str_len] + f"... [+{len(v)-max_str_len}]"
        elif isinstance(v, (list, tuple)):
            out[k] = f"<list len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = f"<dict keys={list(v.keys())[:5]}>"
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    sys.exit(main())
