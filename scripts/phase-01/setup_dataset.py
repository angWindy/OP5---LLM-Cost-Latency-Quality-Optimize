#!/usr/bin/env python3
"""
Download ZeroSCROLLS (tau/zero_scrolls) task zips via HTTP, stream rows, then split
deterministically:
  - 5 rows  -> data/processed/zero_scrolls_test5.jsonl  (PoC sanity, read-only after creation)
  - 95 rows -> data/processed/zero_scrolls_dev95.jsonl  (pool cho eval mo rong sau)

Doi tu LongBench-v2 sang ZeroSCROLLS (2026-09-21) vi:
  - ZeroSCROLLS context trung binh ~10k tokens (LongBench-v2 ~120k)
  - Accuracy ky vong 60-70% (LongBench-v2 chi 25-45%)
  - Multi-domain 10 tasks, co gold answer (F1/EM/Rouge)
  - Public tren HF, khong can token, da duoc LongLLMLingua paper benchmark

QUAN TRONG (sau khi inspect schema that 2026-09-21):
  - Dataset dung loading script `zero_scrolls.py` nên KHONG load duoc qua
    `datasets.load_dataset(...)` nữa (phiên bản `datasets>=3.4` đã remove
    support cho dataset scripts).
  - Moi task co 1 file `.zip` rieng (qasper.zip, musique.zip, gov_report.zip, ...).
    Moi zip chứa `<task>/test.jsonl` và `<task>/validation.jsonl`.
  - Schema moi: `{id, pid, input, output, document_start_index,
    document_end_index, query_start_index, query_end_index, truncation_seperator}`
    → context extracted từ input[document_start:end], question từ input[query_start:end].

Script này download từng zip, extract rows từ test.jsonl, gán thêm field `task`,
và stratified sample per-task de đa dang mien.

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
import zipfile
import requests
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

# Task list — sync với _common.ZERO_SCROLLS_TASKS (đã loại narrative_qa quá lớn).
DEFAULT_TASKS = [
    "qasper", "musique", "gov_report", "space_digest",
    "summ_screen_fd", "qmsum", "squality", "quality", "book_sum_sort",
]


def download_one_task(task: str, out_dir: Path) -> Path:
    """Download 1 task zip từ HF và extract. Idempotent."""
    jsonl_path = out_dir / task / "test.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path  # cached

    zip_path = out_dir / f"{task}.zip"
    url = f"https://huggingface.co/datasets/tau/zero_scrolls/resolve/main/{task}.zip"
    print(f"[download] {task} from HF...", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
        actual = zip_path.stat().st_size
        print(f"[download]   {actual:,} bytes ({actual/1024/1024:.1f} MB)", flush=True)
    except Exception as exc:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"download {task}: {exc}") from exc

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    except Exception as exc:
        raise RuntimeError(f"extract {task}: {exc}") from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()

    if not jsonl_path.exists():
        raise RuntimeError(f"{task}/test.jsonl not found after extract")
    return jsonl_path


def load_rows_from_tasks(tasks: list[str], cache_dir: Path) -> list[dict]:
    """Download + load rows từ tất cả tasks, gắn field `task`, trả về list."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task in tasks:
        try:
            jsonl_path = download_one_task(task, cache_dir)
        except Exception as exc:
            print(f"[download] WARNING: skip task {task}: {exc}", file=sys.stderr)
            continue
        with open(jsonl_path) as f:
            n = 0
            for line in f:
                raw = json.loads(line)
                raw["task"] = task
                rows.append(raw)
                n += 1
        print(f"[load] {task}: {n} rows", flush=True)
    return rows


def stratified_sample_by_task(rows: list[dict], per_task: int) -> list[dict]:
    """Lấy tối đa `per_task` rows từ mỗi task để đa dạng miền.

    Nếu row không có field 'task' thì trả về rows nguyên (fallback).
    """
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
    parser.add_argument("--per_task", type=int, default=12,
                        help="Stratified sample per task (ZeroSCROLLS only, default 12)")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                        help="Comma-separated task names (default: 9 tasks bỏ narrative_qa)")
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST5_SEED,
                        help=f"Shuffle seed (default {DEFAULT_TEST5_SEED})")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Skip stratified sampling (use raw first N rows)")
    parser.add_argument("--cache_dir", type=str, default="/tmp/zero_scrolls",
                        help="Where to cache downloaded zips")
    args = parser.parse_args()

    if args.dataset == "longbench_v2":
        # Legacy path — delegate to streaming via datasets.load_dataset
        print("[legacy] LongBench-v2 path uses streaming (HF_TOKEN may be needed).")
        from datasets import load_dataset
        token = os.getenv("HF_TOKEN") or True
        ds = load_dataset("zai-org/LongBench-v2", split="train", streaming=True, token=token)
        rows = []
        for i, r in enumerate(ds):
            rows.append(dict(r))
            if i + 1 >= args.n_total:
                break
        raw_path = RAW_DIR / f"longbench_v2_first{args.n_total}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        test_path = PROC_DIR / "longbench_v2_test5.jsonl"
        dev_path = PROC_DIR / "longbench_v2_dev95.jsonl"
        write_jsonl(rows, raw_path)
        test_rows, dev_rows = split_5_95(rows, seed=args.seed)
        write_jsonl(test_rows, test_path)
        write_jsonl(dev_rows, dev_path)
        return 0

    # ZeroSCROLLS path
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print("=== Dataset setup ===")
    print(f"  dataset:    {args.dataset}  (tau/zero_scrolls)")
    print(f"  tasks:      {len(tasks)} ({', '.join(tasks)})")
    print(f"  per_task:   {args.per_task}")
    print(f"  n_total:    {args.n_total}")
    print(f"  seed:       {args.seed}")
    print(f"  cache_dir:  {args.cache_dir}")
    print()

    cache_dir = Path(args.cache_dir)
    rows = load_rows_from_tasks(tasks, cache_dir)
    print(f"[load] Total raw rows: {len(rows)}")
    if len(rows) < args.n_total:
        print(f"WARNING: got only {len(rows)} rows (need {args.n_total})", file=sys.stderr)

    if not args.no_stratify:
        rows = stratified_sample_by_task(rows, per_task=args.per_task)
        if len(rows) > args.n_total:
            rows = rows[: args.n_total]

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_path = RAW_DIR / f"zero_scrolls_first{args.n_total}_{date_str}.jsonl"
    test_path = PROC_DIR / "zero_scrolls_test5.jsonl"
    dev_path = PROC_DIR / "zero_scrolls_dev95.jsonl"

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
            print(f"  {k:30s} | {type(v).__name__:10s} | {preview}")

    print()
    print("OK Setup done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
