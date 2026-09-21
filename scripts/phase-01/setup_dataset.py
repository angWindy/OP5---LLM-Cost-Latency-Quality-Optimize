#!/usr/bin/env python3
"""
Download ZeroSCROLLS (tau/zero_scrolls) via HuggingFace, save first N rows, then split
deterministically:
  - 5 rows  -> data/processed/zero_scrolls_test5.jsonl  (PoC sanity, read-only after creation)
  - 95 rows -> data/processed/zero_scrolls_dev95.jsonl  (pool cho eval mo rong sau)

Doi tu LongBench-v2 sang ZeroSCROLLS (2026-09-21) vi:
  - ZeroSCROLLS context trung binh ~10k tokens (LongBench-v2 ~120k)
  - Accuracy ky vong 60-70% (LongBench-v2 chi 25-45%)
  - Multi-domain 10 tasks, co gold answer (F1/EM/Rouge)
  - Public tren HF, khong can token, da duoc LongLLMLingua paper benchmark

Use seed=42 de split co the lap lai.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/setup_dataset.py [--n_total 100] [--dataset zero_scrolls]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    print("WARNING: python-dotenv not installed.", file=sys.stderr)


# Dataset registry (keep in sync with _common.DATASETS)
DATASET_REGISTRY = {
    "zero_scrolls": "tau/zero_scrolls",
    "longbench_v2": "zai-org/LongBench-v2",
}
DEFAULT_DATASET = "zero_scrolls"
DEFAULT_N_TOTAL = 100
DEFAULT_TEST5_SEED = 42
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"


def download_rows(dataset_name: str, n: int, split: str = "test") -> list[dict]:
    """Stream first N rows from HF dataset, return list of dicts.

    ZeroSCROLLS không cần HF token (public). LongBench-v2 thì tùy trường hợp.
    """
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN") or True  # fallback to public access
    print(f"[download] Streaming {dataset_name} split='{split}' n={n} ...")
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
    rows: list[dict] = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if (i + 1) % 10 == 0:
            print(f"  ...got {i+1} rows")
        if i + 1 >= n:
            break
    print(f"[download] Got {len(rows)} rows total")
    return rows


def stratified_sample_by_task(rows: list[dict], per_task: int = 6) -> list[dict]:
    """Stratified sample cho ZeroSCROLLS: lấy đều từ mỗi task để đa dạng miền.

    Nếu row không có field 'task' (LongBench-v2) thì fallback về giữ nguyên rows.
    """
    has_task = any("task" in r for r in rows[:5])
    if not has_task:
        print("[stratify] No 'task' field detected — skip stratified sampling")
        return rows

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("task", "unknown"))].append(r)

    print(f"[stratify] Detected {len(by_task)} tasks: {sorted(by_task.keys())}")
    out: list[dict] = []
    for task, group in by_task.items():
        out.extend(group[:per_task])
    print(f"[stratify] After per-task cap {per_task}: {len(out)} rows")
    return out


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
    parser = argparse.ArgumentParser(
        description="Download dataset (ZeroSCROLLS default) + split 5/95"
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASET_REGISTRY.keys()),
                        help=f"Dataset key (default {DEFAULT_DATASET})")
    parser.add_argument("--n_total", type=int, default=DEFAULT_N_TOTAL,
                        help=f"Total rows to download (default {DEFAULT_N_TOTAL})")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default 'test')")
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST5_SEED,
                        help=f"Shuffle seed (default {DEFAULT_TEST5_SEED})")
    parser.add_argument("--per_task", type=int, default=6,
                        help="Stratified sample per task (ZeroSCROLLS only, default 6)")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Skip stratified sampling (use raw first N rows)")
    args = parser.parse_args()

    dataset_id = DATASET_REGISTRY[args.dataset]
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_name = args.dataset.replace("-", "_")
    raw_path = RAW_DIR / f"{safe_name}_first{args.n_total}_{date_str}.jsonl"
    test_path = PROC_DIR / f"{safe_name}_test5.jsonl"
    dev_path = PROC_DIR / f"{safe_name}_dev95.jsonl"

    print("=== Dataset setup ===")
    print(f"  dataset:    {args.dataset}  ({dataset_id})")
    print(f"  split:      {args.split}")
    print(f"  n_total:    {args.n_total}")
    print(f"  seed:       {args.seed}")
    print(f"  raw_path:   {raw_path}")
    print(f"  test_path:  {test_path}  (5 rows)")
    print(f"  dev_path:   {dev_path}   (95 rows)")
    if args.dataset == "zero_scrolls" and not args.no_stratify:
        print(f"  per_task:   {args.per_task} (stratified sample)")
    print()

    # Stream nhiều hơn n_total để stratified sample có đủ task diversity
    # 10 tasks × 6 = 60 minimum, nên stream tối thiểu n_total nhưng ưu tiên nhiều hơn
    stream_n = args.n_total * 4 if args.dataset == "zero_scrolls" and not args.no_stratify else args.n_total
    rows = download_rows(dataset_id, stream_n, split=args.split)
    if len(rows) < 100:
        print(f"WARNING: got only {len(rows)} rows (need 100)", file=sys.stderr)

    # Apply stratified sampling for ZeroSCROLLS
    if args.dataset == "zero_scrolls" and not args.no_stratify:
        rows = stratified_sample_by_task(rows, per_task=args.per_task)
        # Cắt còn n_total nếu nhiều hơn
        if len(rows) > args.n_total:
            rows = rows[: args.n_total]

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
