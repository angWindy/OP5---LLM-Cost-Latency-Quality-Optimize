"""
Rerun the failed cases from the Phase 02 baseline 200 with longer backoff.

Context: scripts/phase-02/baseline_200.py ran 200 stratified ZeroSCROLLS cases
against `gemini-3.5-flash-lite` and got 9 × `exhausted retries (429/5xx)`.
Original retry policy was `max_retries=4` with exponential backoff base 2.0
(2s, 4s, 8s, 16s between attempts) — too short to outwait Gemini's per-minute
rate limit when bursty 429s arrive.

This script:
  1. Reads `results/phase-02-baseline-200.jsonl`, collects case_ids with status != "ok".
  2. Loads the original 200-case sample from `data/processed/zero_scrolls_200.jsonl`
     to rebuild the same context/question that the failed call used.
  3. Re-calls Gemini for ONLY those 9 cases with:
       - max_retries = 12
       - backoff for 429/5xx = 30, 60, 120, 180, 300, 300, ... seconds
         (capped at 5 min — Gemini free-tier rate-limit windows typically reset
         within a few minutes, but we give margin up to 10 min total per call).
  4. Measures latency EXACTLY as the time inside `requests.post()` only —
     backoff sleeps are excluded (user requirement 2026-09-22).
  5. Reports token counts DIRECTLY from Gemini's `usageMetadata`
     (input_tokens = `promptTokenCount`, output_tokens = `candidatesTokenCount`).
     No approximation.
  6. On success, replaces the corresponding error row in the JSONL with the
     successful record (preserving case_id and order). Recomputes summary.

Usage:
  conda activate vsf
  python scripts/phase-02/rerun_failed_9.py

Outputs (in place):
  results/phase-02-baseline-200.jsonl         — error rows replaced with successful ones
  results/phase-02-baseline-200-summary.json  — recomputed
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")
from _prompts import format_eval_prompt as _format_eval_prompt

# Re-use the baseline constants — same model, same prompt, same target_tokens
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
OUTPUT_PATH = REPO_ROOT / "results" / "phase-02-baseline-200.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-summary.json"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT_TEMPLATE = None  # canonical prompt lives in scripts/_prompts.py
TARGET_TOKENS = 8000  # same as baseline_200.py

# --- Long-backoff policy (the actual reason for this script) -------------
# Progressive schedule: 30s, 60s, 120s, 180s, 300s, 300s, 300s, ...
# Capped at 300s (5 min). After retry #6 we keep waiting 5 min each attempt
# up to max_retries=12. Worst-case total wait per call ≈ 36 min if every
# attempt 429s. Realistically 1-2 of the 9 cases will need >2 retries.
BACKOFF_SCHEDULE = [30, 60, 120, 180, 300, 300, 300, 300, 300, 300, 300, 300]  # seconds
MAX_RETRIES = len(BACKOFF_SCHEDULE)


# ----------------------------------------------------------------------
# 1. Locate failed cases in existing JSONL
# ----------------------------------------------------------------------
def collect_failed_cases(jsonl_path: Path) -> list[dict]:
    """Return error records (status != 'ok') preserving original JSONL order."""
    failed = []
    with jsonl_path.open() as f:
        for ln in f:
            rec = json.loads(ln)
            if rec.get("status") != "ok":
                failed.append(rec)
    return failed


# ----------------------------------------------------------------------
# 2. Reload context/question for failed case_ids from the original sample
# ----------------------------------------------------------------------
def load_sample_index(sample_path: Path) -> dict[str, dict]:
    """Build case_id -> full row (with input + index offsets) from saved sample."""
    if not sample_path.exists():
        raise RuntimeError(
            f"Sample file not found: {sample_path}. "
            "Re-run scripts/phase-02/baseline_200.py first."
        )
    idx = {}
    with sample_path.open() as f:
        for line in f:
            r = json.loads(line)
            idx[r["id"]] = r
    return idx


# ----------------------------------------------------------------------
# 3. Single Gemini call with long backoff (strict latency timing)
# ----------------------------------------------------------------------
def truncate_context(context: str, question: str, target_tokens: int) -> tuple[str, bool]:
    """Same logic as baseline_200.py — must produce identical truncated text."""
    char_budget = (target_tokens - 300) * 4  # reserve 300 tokens for Q+instruction
    if len(context) <= char_budget:
        return context, False
    return context[:char_budget] + "\n\n[...truncated...]", True


def call_gemini_long_backoff(prompt: str, *, log_path: Path | None = None) -> dict:
    """Single Gemini call.

    Latency semantics (strict, per user requirement 2026-09-22):
        latency_ms = time spent inside requests.post() only.
        Backoff sleeps are excluded (timer starts AFTER each sleep).

    Returns dict with: status, text, input_tokens, output_tokens,
    latency_ms (HTTP call time only), attempts, http_code, error.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}
    url = GEMINI_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }

    last_error_msg = ""
    last_http_code = None
    for attempt in range(MAX_RETRIES):
        wait_s = BACKOFF_SCHEDULE[attempt] if attempt < len(BACKOFF_SCHEDULE) else 300

        # --- Backoff sleep is OUTSIDE the latency timer ---
        # First attempt: no sleep.
        # Subsequent attempts: sleep BEFORE the HTTP call.
        if attempt > 0:
            if log_path:
                log_attempt(log_path, "backoff", attempt=attempt, wait_s=wait_s)
            print(f"      [backoff] attempt #{attempt+1}: sleeping {wait_s}s before retry",
                  flush=True)
            time.sleep(wait_s)

        # --- HTTP call is the ONLY thing inside the latency timer ---
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            sc = resp.status_code
            last_http_code = sc
        except requests.Timeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            last_error_msg = "timeout"
            if log_path:
                log_attempt(log_path, "timeout", attempt=attempt, latency_ms=elapsed_ms)
            continue
        except requests.ConnectionError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            last_error_msg = f"connection: {exc}"
            if log_path:
                log_attempt(log_path, "connection_error", attempt=attempt,
                            latency_ms=elapsed_ms, err=str(exc)[:200])
            continue

        if sc == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {
                    "status": "blocked",
                    "http_code": 200,
                    "latency_ms": elapsed_ms,
                    "attempts": attempt + 1,
                    "error": "no candidates (safety block or empty)",
                    "raw": json.dumps(data)[:300],
                }
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            usage = data.get("usageMetadata", {}) or {}
            return {
                "status": "ok",
                "text": text,
                # EXACT — from Gemini's usageMetadata, no approximation
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "latency_ms": elapsed_ms,  # call-only, excludes backoff sleeps
                "attempts": attempt + 1,
                "http_code": 200,
            }

        if sc == 429:
            last_error_msg = "HTTP 429 (rate limit)"
            if log_path:
                log_attempt(log_path, "http_429", attempt=attempt, latency_ms=elapsed_ms)
            continue
        if sc in (500, 502, 503, 504):
            last_error_msg = f"HTTP {sc} (transient)"
            if log_path:
                log_attempt(log_path, f"http_{sc}", attempt=attempt, latency_ms=elapsed_ms)
            continue
        if sc in (400, 403):
            body = resp.text[:300]
            return {
                "status": "error", "http_code": sc,
                "latency_ms": elapsed_ms,
                "attempts": attempt + 1,
                "error": f"HTTP {sc}: {body}",
            }
        # Other 4xx — don't retry
        return {
            "status": "error", "http_code": sc,
            "latency_ms": elapsed_ms,
            "attempts": attempt + 1,
            "error": f"HTTP {sc}: {resp.text[:200]}",
        }

    return {
        "status": "error",
        "attempts": MAX_RETRIES,
        "http_code": last_http_code,
        "error": f"exhausted retries (last: {last_error_msg})",
    }


# ----------------------------------------------------------------------
# 4. Logging helpers (per-call ops log)
# ----------------------------------------------------------------------
def log_attempt(log_path: Path, event: str, **fields) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event,
        **fields,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# 5. Summary recomputation (same shape as baseline_200.py)
# ----------------------------------------------------------------------
def recompute_summary(records: list[dict], *, out_path: Path) -> dict:
    ok_records = [r for r in records if r.get("status") == "ok"]
    statuses = defaultdict(int)
    for r in records:
        statuses[r.get("status", "?")] += 1

    in_toks = [r["input_tokens"] for r in ok_records if r.get("input_tokens") is not None]
    out_toks = [r["output_tokens"] for r in ok_records if r.get("output_tokens") is not None]
    lats = [r["latency_ms"] for r in ok_records if r.get("latency_ms") is not None]
    trunc = sum(1 for r in records if r.get("truncated"))

    approx_tokens_all = [r.get("approx_input_tokens", 0) for r in records
                         if r.get("approx_input_tokens") is not None]
    summary = {
        "total": len(records),
        "ok": len(ok_records),
        "statuses": dict(statuses),
        "truncated": trunc,
        "token_range_approx": [min(approx_tokens_all), max(approx_tokens_all)]
                              if approx_tokens_all else None,
        "input_tokens": {
            "n": len(in_toks),
            "min": min(in_toks) if in_toks else None,
            "median": statistics.median(in_toks) if in_toks else None,
            "p95": sorted(in_toks)[int(len(in_toks) * 0.95)] if in_toks else None,
            "max": max(in_toks) if in_toks else None,
        },
        "output_tokens": {
            "n": len(out_toks),
            "median": statistics.median(out_toks) if out_toks else None,
            "max": max(out_toks) if out_toks else None,
        },
        "latency_ms": {
            "n": len(lats),
            "min": min(lats) if lats else None,
            "median": statistics.median(lats) if lats else None,
            "p95": sorted(lats)[int(len(lats) * 0.95)] if lats else None,
            "max": max(lats) if lats else None,
        },
        "model": GEMINI_MODEL,
        "target_tokens": TARGET_TOKENS,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--log", type=Path,
                        default=Path("/tmp/rerun_failed_9.log"),
                        help="Per-attempt ops log (backoff/sleep events)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List failed cases + show plan, don't actually call")
    args = parser.parse_args()

    # 1. Find failures
    failed = collect_failed_cases(args.jsonl)
    if not failed:
        print(f"[OK] No failed cases in {args.jsonl}. Nothing to rerun.")
        return 0

    print(f"=== Rerun failed cases with long backoff ===")
    print(f"  Failed cases: {len(failed)}")
    for f in failed:
        print(f"    - case_id={f['case_id']}  task={f['task']}  "
              f"approx_tok={f.get('approx_input_tokens', '?')}  "
              f"err={f.get('error', '?')[:50]}")

    if args.dry_run:
        print(f"\n[dry-run] Backoff schedule: {BACKOFF_SCHEDULE}")
        print(f"[dry-run] max_retries: {MAX_RETRIES}")
        print(f"[dry-run] Worst-case wait per call: "
              f"{sum(BACKOFF_SCHEDULE)/60:.1f} min")
        print(f"[dry-run] Worst-case total wall time: "
              f"{sum(BACKOFF_SCHEDULE) * len(failed) / 60:.1f} min")
        return 0

    # 2. Load sample index
    sample_idx = load_sample_index(args.sample)
    missing = [f["case_id"] for f in failed if f["case_id"] not in sample_idx]
    if missing:
        print(f"[FATAL] {len(missing)} failed case_ids not in sample: {missing}")
        return 1

    # 3. Clear ops log
    if args.log.exists():
        args.log.unlink()

    # 4. Rerun each failed case
    print(f"\n[1/2] Re-calling Gemini for {len(failed)} cases...")
    print(f"  Backoff schedule (s): {BACKOFF_SCHEDULE}")
    print(f"  max_retries per call: {MAX_RETRIES}")
    print(f"  Latency = HTTP call only (excludes backoff sleep)")
    print(f"  Tokens = exact from usageMetadata\n")

    new_records: list[dict] = []
    for i, f in enumerate(failed):
        case_id = f["case_id"]
        task = f["task"]
        sample_row = sample_idx[case_id]
        ds = int(sample_row["document_start_index"])
        de = int(sample_row["document_end_index"])
        qs = int(sample_row["query_start_index"])
        qe = int(sample_row["query_end_index"])
        inp = sample_row["input"]
        context = inp[ds:de]
        question = inp[qs:qe]
        gold = sample_row.get("output") or ""

        context, truncated = truncate_context(context, question, TARGET_TOKENS)
        prompt = _format_eval_prompt(context, question)

        print(f"  [{i+1}/{len(failed)}] case_id={case_id} task={task} "
              f"approx_tok={sample_row['_approx_input_tokens']}", flush=True)

        t_start = time.perf_counter()
        rec = call_gemini_long_backoff(prompt, log_path=args.log)
        wall_seconds = time.perf_counter() - t_start

        # Inherit metadata from the original (failed) record
        rec["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["case_id"] = case_id
        rec["task"] = task
        rec["question"] = question[:200]
        rec["gold"] = gold[:200]
        rec["approx_input_tokens"] = sample_row["_approx_input_tokens"]
        rec["context_chars"] = len(context)
        rec["truncated"] = truncated
        rec["prompt_chars"] = len(prompt)
        rec["wall_seconds"] = wall_seconds  # includes backoff sleeps (for diagnostics)

        marker = {"ok": "OK", "error": "ERR", "timeout": "T", "blocked": "BLK"}.get(
            rec["status"], "?")
        in_tok = rec.get("input_tokens", "?")
        out_tok = rec.get("output_tokens", "?")
        ms = rec.get("latency_ms", 0)
        attempts = rec.get("attempts", 0)
        print(f"           {marker}  in_tok={in_tok}  out_tok={out_tok}  "
              f"latency_ms={ms:.0f}  attempts={attempts}  "
              f"wall_s={wall_seconds:.1f}")

        new_records.append(rec)

    # 5. Patch the JSONL: replace error rows by case_id
    print(f"\n[2/2] Patching {args.jsonl} ...")
    all_records = []
    with args.jsonl.open() as f:
        for ln in f:
            r = json.loads(ln)
            all_records.append(r)
    replacement = {r["case_id"]: r for r in new_records}
    n_replaced = 0
    for i, r in enumerate(all_records):
        if r.get("status") != "ok" and r["case_id"] in replacement:
            all_records[i] = replacement[r["case_id"]]
            n_replaced += 1
    print(f"  Replaced {n_replaced} error rows")

    # Write back atomically
    tmp = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.jsonl)
    print(f"  Wrote: {args.jsonl}")

    # 6. Recompute summary
    summary = recompute_summary(all_records, out_path=args.summary)
    print(f"  Wrote: {args.summary}")

    # Final report
    ok_new = sum(1 for r in new_records if r["status"] == "ok")
    err_new = sum(1 for r in new_records if r["status"] != "ok")
    print(f"\n=== Rerun summary ===")
    print(f"  Rerun: {ok_new}/{len(new_records)} OK, {err_new} still failing")
    print(f"  Overall: {summary['ok']}/{summary['total']} OK")
    print(f"  Truncated (>=8000 tok): {summary['truncated']}")
    print(f"  Latency (call-only):    "
          f"median={summary['latency_ms']['median']:.0f}ms  "
          f"p95={summary['latency_ms']['p95']:.0f}ms  "
          f"max={summary['latency_ms']['max']:.0f}ms")
    print(f"  Input tokens (exact):   "
          f"median={summary['input_tokens']['median']}  "
          f"p95={summary['input_tokens']['p95']}  "
          f"max={summary['input_tokens']['max']}")
    print(f"  Output tokens (exact):  "
          f"median={summary['output_tokens']['median']}  "
          f"max={summary['output_tokens']['max']}")

    return 0 if err_new == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
