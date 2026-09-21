#!/usr/bin/env python3
"""
Inspect ZeroSCROLLS (tau/zero_scrolls) dataset structure.

ZeroSCROLLS không load được qua `datasets.load_dataset` (loading script bị
deprecate). Script này tải và extract các task zips từ HF về `/tmp/zero_scrolls/`,
sau đó đọc 3 rows đầu từ `test.jsonl` của task mặc định (`qasper` — nhỏ nhất).

Default dataset switched from LongBench-v2 → ZeroSCROLLS (2026-09-21) because
ZeroSCROLLS has shorter context (~10k tokens) and gold answers that are easier to
score deterministically.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/inspect_dataset.py [--task qasper]
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

# Default task = qasper (nhỏ nhất, 0.3 MB).
DEFAULT_TASK = "qasper"
CACHE_DIR = Path("/tmp/zero_scrolls")


def _ensure_cached(task: str) -> Path:
    """Make sure <cache>/<task>/test.jsonl exists. Download nếu thiếu."""
    import zipfile
    import requests

    out_dir = CACHE_DIR
    jsonl_path = out_dir / task / "test.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{task}.zip"
    url = f"https://huggingface.co/datasets/tau/zero_scrolls/resolve/main/{task}.zip"
    print(f"Downloading {task} from HF...")
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()
    if not jsonl_path.exists():
        raise RuntimeError(f"{task}/test.jsonl not found after extract")
    return jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect dataset structure (ZeroSCROLLS default)"
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help=f"ZeroSCROLLS task name (default {DEFAULT_TASK})")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of rows to inspect (default 3)")
    args = parser.parse_args()

    print(f"Loading ZeroSCROLLS task: {args.task}")
    print(f"  n:  {args.n}")
    print("---")

    try:
        jsonl_path = _ensure_cached(args.task)
    except Exception as exc:
        print(f"ERROR: Khong download duoc task: {exc}", file=sys.stderr)
        return 4

    try:
        with open(jsonl_path) as f:
            rows: list[dict] = []
            for i, line in enumerate(f):
                rows.append(json.loads(line))
                if i + 1 >= args.n:
                    break
    except Exception as exc:
        print(f"ERROR: Khong doc duoc row: {exc}", file=sys.stderr)
        return 6

    if not rows:
        print("ERROR: Dataset rong.", file=sys.stderr)
        return 5

    first_row = rows[0]
    print("First row (field summary):")
    for key, value in first_row.items():
        preview = repr(value)[:150]
        print(f"  {key:30s} | {type(value).__name__:10s} | {preview}")
    print("---")

    print("First row as JSON (truncated):")
    print(json.dumps(_summarize_row(first_row), indent=2, ensure_ascii=False, default=str))
    print("---")

    print(f"Inspecting {len(rows)} rows for schema stability...")
    for i, row in enumerate(rows[1:], start=2):
        keys_match = set(row.keys()) == set(first_row.keys())
        print(f"  Row {i}: keys_match={keys_match} | num_fields={len(row)}")

    print("---")
    print("OK Inspect done.")
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
