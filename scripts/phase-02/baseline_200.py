"""
Baseline eval on a stratified 200-case sample of ZeroSCROLLS test split.

Pipeline:
  1. Download ZeroSCROLLS test split (9 tasks, ~700 cases total) -> /tmp/zero_scrolls/
  2. Stratified sample 200 cases spread across [min, max] input-token range
     (per-task proportional quota + length-bin stratified).
  3. Run `gemini-3.5-flash-lite` baseline on each case -> results JSONL.

Usage:
  conda activate vsf
  python scripts/phase-02/baseline_200.py

Outputs:
  data/processed/zero_scrolls_200.jsonl  - the stratified 200-case sample
  results/phase-02-baseline-200.jsonl     - baseline Gemini calls
  results/phase-02-baseline-200-summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")
from _http import get_session
from _prompts import format_eval_prompt as _format_eval_prompt

# Process-wide HTTP session reused across all calls.
_SESSION = get_session()

ZEROSCROLLS_CACHE = Path("/tmp/zero_scrolls")
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
OUTPUT_PATH = REPO_ROOT / "results" / "phase-02-baseline-200.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-summary.json"
TASKS = [
    "qasper", "musique", "gov_report", "space_digest",
    "summ_screen_fd", "qmsum", "squality", "quality", "book_sum_sort",
]
PROMPT_TEMPLATE = None  # canonical prompt lives in scripts/_prompts.py

HF_BASE = "https://huggingface.co/datasets/tau/zero_scrolls/resolve/main"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ---------------------------------------------------------------
# 1. Download ZeroSCROLLS
# ---------------------------------------------------------------
def download_task(task: str) -> Path:
    """Download one task zip from HF, extract, return jsonl path. Idempotent."""
    out_dir = ZEROSCROLLS_CACHE / task
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "test.jsonl"
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path

    url = f"{HF_BASE}/{task}.zip"
    zip_path = ZEROSCROLLS_CACHE / f"{task}.zip"
    print(f"  [download] {task} ...", end=" ", flush=True)
    try:
        # Streaming download — keep requests.get directly (not via call_with_retry)
        # because we want chunked iteration, not a buffered response.
        r = _SESSION.get(url, stream=True, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ZEROSCROLLS_CACHE)
        size_mb = jsonl_path.stat().st_size / 1024 / 1024 if jsonl_path.exists() else 0
        print(f"OK ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise
    finally:
        if zip_path.exists():
            zip_path.unlink()
    return jsonl_path


def load_all_cases() -> list[dict]:
    """Load all cases across 9 tasks. Annotate with task name + token estimate."""
    rows = []
    for task in TASKS:
        path = download_task(task)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                r["task"] = task
                rows.append(r)
    # Approximate input tokens = sum of chars/4 of (context + question + instruction)
    for r in rows:
        inp = r.get("input", "")
        ds = int(r.get("document_start_index", 0))
        de = int(r.get("document_end_index", 0))
        qs = int(r.get("query_start_index", 0))
        qe = int(r.get("query_end_index", 0))
        ctx_chars = de - ds
        q_chars = qe - qs
        # Heuristic: ~4 chars/token (English)
        r["_approx_input_tokens"] = (ctx_chars + q_chars + 200) // 4  # +200 for prompt scaffolding
        r["_context_chars"] = ctx_chars
        r["_question_chars"] = q_chars
    return rows


# ---------------------------------------------------------------
# 2. Stratified sample 200 across [min, max] length
# ---------------------------------------------------------------
def stratified_sample_200(rows: list[dict], n_total: int = 200, seed: int = 42) -> list[dict]:
    """Per-task proportional quota, length-bin stratified within each task.

    Length bins: quintiles of approx_input_tokens within each task.
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    # Per-task quota: proportional to task size, min 1
    total = len(rows)
    quotas = {t: max(1, round(n_total * len(v) / total)) for t, v in by_task.items()}
    # Adjust to hit exactly n_total
    diff = n_total - sum(quotas.values())
    if diff > 0:
        for t in sorted(quotas, key=lambda x: -quotas[x]):
            quotas[t] += 1
            diff -= 1
            if diff == 0:
                break
    elif diff < 0:
        for t in sorted(quotas, key=lambda x: quotas[x]):
            quotas[t] -= 1
            diff += 1
            if diff == 0:
                break

    selected = []
    for task, group in by_task.items():
        q = quotas[task]
        if len(group) <= q:
            selected.extend(group)
            continue
        # Quintile bins by length
        group_sorted = sorted(group, key=lambda r: r["_approx_input_tokens"])
        bin_size = len(group_sorted) / 5
        bins = [group_sorted[int(i * bin_size):int((i + 1) * bin_size)] for i in range(5)]
        # Allocate q across 5 bins as evenly as possible
        per_bin = [q // 5] * 5
        for i in range(q % 5):
            per_bin[i] += 1
        for b, k in zip(bins, per_bin):
            if len(b) > k > 0:
                chosen = rng.sample(b, k)
            elif len(b) > 0:
                chosen = b[:k]
            else:
                chosen = []
            selected.extend(chosen)
    selected.sort(key=lambda r: r["_approx_input_tokens"])
    return selected


# ---------------------------------------------------------------
# 3. Run baseline Gemini call
# ---------------------------------------------------------------
def call_gemini_baseline(prompt: str, *, max_retries: int = 3) -> dict:
    """Single Gemini REST call with capped retry/backoff (via shared session).

    Returns dict with keys: status, text, input_tokens, output_tokens,
    latency_ms, error (if any), http_code.
    """
    from _http import call_with_retry

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}
    url = GEMINI_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }

    t0 = time.perf_counter()
    try:
        resp = call_with_retry(
            "POST", url, max_retries=max_retries, timeout=120,
            json=payload, session=_SESSION,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sc = resp.status_code

        if sc == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {
                    "status": "blocked", "http_code": 200,
                    "latency_ms": elapsed_ms,
                    "error": "no candidates (safety block or empty)",
                    "raw": json.dumps(data)[:300],
                }
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            usage = data.get("usageMetadata", {})
            return {
                "status": "ok",
                "text": text,
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "latency_ms": elapsed_ms,
                "http_code": 200,
            }
        # Non-retryable HTTP errors (already past any retries in call_with_retry)
        body = resp.text[:300]
        return {
            "status": "error", "http_code": sc, "latency_ms": elapsed_ms,
            "error": f"HTTP {sc}: {body}",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc),
                "latency_ms": (time.perf_counter() - t0) * 1000.0}


def truncate_context(context: str, question: str, target_tokens: int) -> tuple[str, bool]:
    """Truncate context from the END to fit within target_tokens (approx).

    Returns (truncated_context, was_truncated).
    """
    # Reserve budget for question (~100 tokens) + instruction (~200 tokens)
    char_budget = (target_tokens - 300) * 4
    if len(context) <= char_budget:
        return context, False
    return context[:char_budget] + "\n\n[...truncated...]", True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=8000,
                        help="Approx max input tokens; truncate beyond this")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Sleep seconds between calls (default 0.05)")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-out", type=Path, default=SAMPLE_PATH)
    args = parser.parse_args()

    # 1. Load all cases
    print(f"=== Phase 2 baseline on {args.n} stratified ZeroSCROLLS cases ===\n")
    print("[1/3] Loading ZeroSCROLLS test split...")
    rows = load_all_cases()
    print(f"  Loaded {len(rows)} cases across {len(set(r['task'] for r in rows))} tasks")

    # 2. Stratified sample
    print(f"\n[2/3] Stratified sample to {args.n} cases (seed={args.seed})...")
    sample = stratified_sample_200(rows, n_total=args.n, seed=args.seed)
    print(f"  Selected {len(sample)} cases")
    # Per-task breakdown
    by_task = defaultdict(int)
    for r in sample:
        by_task[r["task"]] += 1
    for t in sorted(by_task):
        print(f"    {t:<18}: {by_task[t]}")

    # Save sample
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sample_out.open("w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote: {args.sample_out}")

    # Token range summary
    tok_min = min(r["_approx_input_tokens"] for r in sample)
    tok_max = max(r["_approx_input_tokens"] for r in sample)
    tok_med = statistics.median(r["_approx_input_tokens"] for r in sample)
    print(f"  Token range: [{tok_min:,}, {tok_max:,}], median={tok_med:,.0f}")

    # 3. Run baseline
    print(f"\n[3/3] Running Gemini {GEMINI_MODEL} baseline...")
    print(f"  Target tokens: {args.target_tokens:,} (truncate beyond)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    all_records = []
    statuses = defaultdict(int)
    total = len(sample)
    for i, r in enumerate(sample):
        ds = int(r.get("document_start_index", 0))
        de = int(r.get("document_end_index", 0))
        qs = int(r.get("query_start_index", 0))
        qe = int(r.get("query_end_index", 0))
        inp = r.get("input", "")
        context = inp[ds:de]
        question = inp[qs:qe]
        gold = r.get("output") or ""

        # Truncate if needed
        context, truncated = truncate_context(context, question, args.target_tokens)
        prompt = _format_eval_prompt(context, question)

        rec = call_gemini_baseline(prompt)
        rec["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["case_id"] = r.get("id", f"C{i:03d}")
        rec["task"] = r["task"]
        rec["question"] = question[:200]
        rec["gold"] = gold[:200]
        rec["approx_input_tokens"] = r["_approx_input_tokens"]
        rec["context_chars"] = len(context)
        rec["truncated"] = truncated
        rec["prompt_chars"] = len(prompt)

        statuses[rec["status"]] += 1

        marker = {"ok": "OK", "error": "ERR", "timeout": "T", "blocked": "BLK"}.get(rec["status"], "?")
        if i % 10 == 0 or rec["status"] != "ok":
            in_tok = rec.get("input_tokens", "?")
            out_tok = rec.get("output_tokens", "?")
            ms = rec.get("latency_ms", 0)
            print(f"  [{i+1:>3}/{total}] {marker:<3} {rec['task']:<14} "
                  f"approx_tok={rec['approx_input_tokens']:>5,} in_tok={str(in_tok):>5} "
                  f"out_tok={str(out_tok):>4} ms={ms:>6.0f} "
                  f"gold={gold[:30]!r:<32}")

        all_records.append(rec)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        time.sleep(args.sleep)

    # Summary
    ok = sum(1 for r in all_records if r["status"] == "ok")
    print(f"\n  Status: {dict(statuses)}")
    print(f"  Total OK: {ok}/{total}")

    if ok > 0:
        ok_records = [r for r in all_records if r["status"] == "ok"]
        in_toks = [r["input_tokens"] for r in ok_records if r.get("input_tokens")]
        out_toks = [r["output_tokens"] for r in ok_records if r.get("output_tokens")]
        lats = [r["latency_ms"] for r in ok_records if r.get("latency_ms")]
        trunc = sum(1 for r in all_records if r.get("truncated"))

        summary = {
            "total": total,
            "ok": ok,
            "statuses": dict(statuses),
            "truncated": trunc,
            "token_range_approx": [tok_min, tok_max],
            "input_tokens": {
                "n": len(in_toks), "min": min(in_toks) if in_toks else None,
                "median": statistics.median(in_toks) if in_toks else None,
                "p95": sorted(in_toks)[int(len(in_toks) * 0.95)] if in_toks else None,
                "max": max(in_toks) if in_toks else None,
            },
            "output_tokens": {
                "n": len(out_toks), "median": statistics.median(out_toks) if out_toks else None,
                "max": max(out_toks) if out_toks else None,
            },
            "latency_ms": {
                "n": len(lats), "min": min(lats) if lats else None,
                "median": statistics.median(lats) if lats else None,
                "p95": sorted(lats)[int(len(lats) * 0.95)] if lats else None,
                "max": max(lats) if lats else None,
            },
            "model": GEMINI_MODEL,
            "target_tokens": args.target_tokens,
        }
        with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote: {args.out}")
        print(f"Wrote: {SUMMARY_PATH}")
        print(f"\nSummary:")
        print(f"  input_tokens: {summary['input_tokens']}")
        print(f"  output_tokens: {summary['output_tokens']}")
        print(f"  latency_ms: {summary['latency_ms']}")
        print(f"  truncated: {trunc}")
    else:
        print("  No successful calls — see error patterns above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
