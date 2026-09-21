#!/usr/bin/env python3
"""
Optimization experiment P0: Skip compressor when preselect output is small.

Hypothesis:
  When preselect_top_k() already reduces context to < threshold chars,
  running LLMLingua-2 on top is overhead without benefit:
    - compressor overhead (~7s CPU) >> Gemini token-saving benefit
    - small contexts are fast for Gemini (~900ms regardless)
    - Approach B (preselect only) already achieves 100% accuracy on this case

Method:
  Add adaptive skip: if preselect_chars < SKIP_THRESHOLD, skip compress entirely.
  Compare 3 variants on the same 1 case (Legal 70k chars):
    - baseline
    - C_combo  (always compress after preselect)
    - C_adaptive (skip compress when preselect < threshold)

Metrics: total_ms, accuracy, chars_kept, gemini_input_tokens

Usage:
  conda activate vsf
  python scripts/phase-01/opt_p0_skip_compressor.py [--threshold 8000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT_JSONL = REPO_ROOT / "results" / "phase-01-opt-p0-skip-compressor.jsonl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results" / "phase-01-opt-p0-skip-compressor.json"
SLEEP_BETWEEN = 1.5

# Reuse everything from compare_3_approaches
from compare_3_approaches import (
    load_shortest_case, split_sentences, cosine_sim,
    preselect_top_k, compress_llmlingua,
    build_qa_prompt, call_gemini, normalize_pred,
    now_iso,
)

# ---------------------------------------------------------------
# Adaptive approach: skip compressor when preselect is small
# ---------------------------------------------------------------
def run_adaptive_combo(
    context: str,
    question: str,
    choices: dict[str, str],
    gold: str,
    gemini,
    rate: float,
    k: int,
    skip_threshold_chars: int,
) -> dict:
    """
    Approach C_adaptive:
      - always preselect first
      - if preselect output < threshold: skip compressor
      - else: run compressor
    """
    t_total = time.perf_counter()

    record = {
        "approach": "C_adaptive",
        "rate": rate,
        "k": k,
        "skip_threshold_chars": skip_threshold_chars,
        "context_chars": len(context),
        "question_chars": len(question),
        "gold": gold,
        "status": "ok",
    }

    # Step 1: preselect
    ps = preselect_top_k(context, question, k=k)
    if ps["status"] != "ok":
        record["status"] = "preselect_error"
        record["error"] = ps.get("error", "unknown")
        record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
        return record

    record["preselect_ms"] = ps["total_ms"]
    record["preselect_n_sentences"] = ps["n_sentences"]
    record["preselect_n_kept"] = ps["n_kept"]
    record["preselect_chars"] = ps["selected_chars"]
    record["preselect_char_ratio"] = ps["char_ratio"]
    ctx = ps["selected_text"]

    # Decision: skip compressor?
    skip_compress = len(ctx) < skip_threshold_chars
    record["skip_compress_decision"] = skip_compress
    record["skip_reason"] = (
        f"preselect_chars={len(ctx)} < threshold={skip_threshold_chars}"
        if skip_compress else
        f"preselect_chars={len(ctx)} >= threshold={skip_threshold_chars}"
    )

    # Step 2: compress (conditional)
    if skip_compress:
        record["compress_ms"] = 0
        record["compress_ratio_str"] = "skipped"
        record["compress_char_ratio"] = 1.0
        record["compress_skipped"] = True
    else:
        comp_text, comp_ms, meta = compress_llmlingua(ctx, question, rate=rate)
        if meta.get("status") == "error":
            record["status"] = "compress_error"
            record["error"] = meta.get("error", "")
            record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
            return record
        record["compress_ms"] = round(comp_ms, 1)
        record["compress_ratio_str"] = meta.get("ratio_str", "")
        record["compress_char_ratio"] = meta.get("char_ratio", 1.0)
        record["compress_skipped"] = False
        ctx = comp_text

    # Step 3: Gemini
    prompt = build_qa_prompt(ctx, question, choices)
    gr = call_gemini(gemini, prompt)
    if gr.get("error"):
        record["status"] = "gemini_error"
        record["error"] = gr.get("error", "")
        record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
        return record

    record["gemini_ms"] = round(gr["latency_ms"], 1)
    record["gemini_input_tokens"] = gr.get("input_tokens")
    record["gemini_output_tokens"] = gr.get("output_tokens")
    record["gemini_text"] = gr["text"][:300]

    pred = normalize_pred(gr["text"])
    record["pred"] = pred
    record["correct"] = 1 if pred == gold else 0
    record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)

    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=8000,
                        help="Skip compressor if preselect_chars < threshold (default 8000)")
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    # Load case
    row = load_shortest_case()
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    if gold not in "ABCD":
        gold = gold[0] if gold else "?"
    choices = {
        "A": row.get("choice_A", ""),
        "B": row.get("choice_B", ""),
        "C": row.get("choice_C", ""),
        "D": row.get("choice_D", ""),
    }

    print("=== Opt P0: Adaptive Skip-Compressor ===")
    print(f"  case:        {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:     {len(context):,} chars")
    print(f"  question:    {question[:80]}...")
    print(f"  gold:        {gold}")
    print(f"  threshold:   {args.threshold} chars")
    print(f"  rate={args.rate}, k={args.k}")
    print()

    # Initialize Gemini
    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env")
        return 1
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(
        os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite")
    )

    # Load prior 3-approach results for comparison
    prior_path = REPO_ROOT / "results" / "phase-01-3-approach-1case.json"
    if prior_path.exists():
        with prior_path.open() as fh:
            prior = json.load(fh)
        prior_records = {r["approach"]: r for r in prior["records"]}
        print(f"  Loaded prior baseline/C_combo for comparison")
    else:
        prior_records = {}
        print(f"  WARNING: prior results not found at {prior_path}")

    records = []

    # Run C_adaptive
    print(f"--- C_adaptive (skip if preselect < {args.threshold}) ---")
    rec = run_adaptive_combo(
        context=context,
        question=question,
        choices=choices,
        gold=gold,
        gemini=gemini,
        rate=args.rate,
        k=args.k,
        skip_threshold_chars=args.threshold,
    )
    rec["ts"] = now_iso()
    rec["case_id"] = row.get("_id", "")
    records.append(rec)

    if rec.get("status") == "ok":
        skip_tag = "[SKIPPED]" if rec.get("compress_skipped") else "[COMPRESSED]"
        print(f"  preselect:  {rec.get('preselect_ms',0):>8.0f} ms  "
              f"(chars {rec.get('preselect_chars','?'):,} / {rec.get('preselect_char_ratio',1)*100:.1f}%)")
        print(f"  decision:   {skip_tag}  {rec.get('skip_reason','')}")
        if not rec.get("compress_skipped"):
            print(f"  compress:   {rec.get('compress_ms',0):>8.0f} ms  "
                  f"ratio={rec.get('compress_ratio_str','?')}")
        print(f"  gemini:     {rec.get('gemini_ms',0):>8.0f} ms  "
              f"tokens={rec.get('gemini_input_tokens','?')}")
        print(f"  TOTAL:    {rec.get('total_ms',0):>8.0f} ms  pred={rec.get('pred','?')} correct={rec.get('correct','?')}")
    else:
        print(f"  ERROR: {rec.get('error', rec.get('status'))}")

    print()

    # Comparison table
    print("=== Comparison vs prior baseline ===")
    print(f"{'Approach':<16} {'PreSel ms':>10} {'Compress ms':>12} {'Gemini ms':>10} {'Total ms':>10} {'Tokens':>8} {'Pred':>4} {'OK?':>4}")
    print("-" * 80)

    for ref_approach in ("baseline", "C_combo"):
        if ref_approach in prior_records:
            r = prior_records[ref_approach]
            if r.get("status") == "ok":
                print(
                    f"  {ref_approach:<14} "
                    f"{r.get('preselect_ms',0):>10.0f} "
                    f"{r.get('compress_ms',0):>12.0f} "
                    f"{r.get('gemini_ms',0):>10.0f} "
                    f"{r.get('total_ms',0):>10.0f} "
                    f"{r.get('gemini_input_tokens', 0) or 0:>8} "
                    f"{r.get('pred','?'):>4} "
                    f"{r.get('correct','?'):>4}"
                )

    rec = records[0]
    if rec.get("status") == "ok":
        print(
            f"  {'C_adaptive':<14} "
            f"{rec.get('preselect_ms',0):>10.0f} "
            f"{rec.get('compress_ms',0):>12.0f} "
            f"{rec.get('gemini_ms',0):>10.0f} "
            f"{rec.get('total_ms',0):>10.0f} "
            f"{rec.get('gemini_input_tokens', 0) or 0:>8} "
            f"{rec.get('pred','?'):>4} "
            f"{rec.get('correct','?'):>4}"
        )

    # Delta analysis
    if "baseline" in prior_records and prior_records["baseline"].get("status") == "ok":
        baseline = prior_records["baseline"]
        print()
        print("=== Delta vs Baseline ===")
        for r in [prior_records.get("C_combo"), records[0]]:
            if r is None or r.get("status") != "ok":
                continue
            delta_ms = r["total_ms"] - baseline["total_ms"]
            baseline_tokens = baseline.get("gemini_input_tokens", 0) or 0
            r_tokens = r.get("gemini_input_tokens", 0) or 0
            token_saving_pct = round(
                (1 - r_tokens / max(baseline_tokens, 1)) * 100, 1
            )
            acc_delta = (r.get("correct", 0) - baseline.get("correct", 0)) * 100
            print(
                f"  {r['approach']:<14}  "
                f"delta_total={delta_ms:>+8.0f} ms  "
                f"token_saving={token_saving_pct:>+6.1f}%  "
                f"acc_delta={acc_delta:>+4.0f}pp"
            )

    # Improvement verdict
    print()
    print("=== Verdict ===")
    if "C_combo" in prior_records and records[0].get("status") == "ok":
        c_combo = prior_records["C_combo"]
        c_adapt = records[0]
        combo_ms = c_combo["total_ms"]
        adapt_ms = c_adapt["total_ms"]
        ms_saved = combo_ms - adapt_ms
        speedup = combo_ms / max(adapt_ms, 1)
        same_accuracy = c_combo.get("correct") == c_adapt.get("correct")
        verdict = "IMPROVE" if (ms_saved > 0 and same_accuracy) else "NO_IMPROVEMENT"
        print(f"  C_combo  total:  {combo_ms:.0f} ms")
        print(f"  C_adaptive total: {adapt_ms:.0f} ms")
        print(f"  Saved:            {ms_saved:.0f} ms ({speedup:.2f}x speedup)")
        print(f"  Accuracy match:   {same_accuracy}")
        print(f"  DECISION:         {verdict}")
        print(f"  Reason:           skip_compress={c_adapt.get('compress_skipped')}, "
              f"preselect_chars={c_adapt.get('preselect_chars')}, threshold={args.threshold}")
        records[0]["verdict"] = verdict
        records[0]["speedup_vs_combo"] = round(speedup, 3)
        records[0]["ms_saved_vs_combo"] = round(ms_saved, 1)

    # Write outputs
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl.exists():
        args.out_jsonl.unlink()
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "ts": now_iso(),
        "case_id": row.get("_id", ""),
        "domain": row.get("domain", ""),
        "context_chars": len(context),
        "gold": gold,
        "args": {
            "rate": args.rate,
            "k": args.k,
            "threshold": args.threshold,
        },
        "records": records,
    }
    with args.out_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nWrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
