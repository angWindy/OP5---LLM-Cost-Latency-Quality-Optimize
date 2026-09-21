#!/usr/bin/env python3
"""
Optimization experiment P5: Tune compressor rate parameter.

Hypothesis:
  rate=0.5 was default. Try rate=0.3 (more compression) and rate=0.7 (less).
  Higher rate = more tokens kept = better recall but slower Gemini.
  Lower rate = fewer tokens = faster Gemini but risk recall.

Method:
  Run Approach C (preselect + compress) at rate=0.3, 0.5, 0.7 on same case.
  Measure: compress_ms, gemini_ms, total_ms, accuracy, tokens.

Usage:
  conda activate vsf
  python scripts/phase-01/opt_p5_rate_tuning.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT_JSONL = REPO_ROOT / "results" / "phase-01-opt-p5-rate-tuning.jsonl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results" / "phase-01-opt-p5-rate-tuning.json"
SLEEP_BETWEEN = 1.5

from compare_3_approaches import (
    load_shortest_case,
    preselect_top_k, compress_llmlingua,
    build_qa_prompt, call_gemini, normalize_pred, now_iso,
)


def run_combo(context, question, choices, gold, gemini, k, rate):
    """Run approach C at given rate."""
    t0 = time.perf_counter()
    ps = preselect_top_k(context, question, k=k)
    ctx = ps["selected_text"]
    comp_text, comp_ms, meta = compress_llmlingua(ctx, question, rate=rate)
    prompt = build_qa_prompt(comp_text, question, choices)
    gr = call_gemini(gemini, prompt)
    pred = normalize_pred(gr["text"])
    return {
        "approach": f"C_rate{rate}",
        "rate": rate,
        "k": k,
        "preselect_ms": ps["total_ms"],
        "preselect_chars": ps["selected_chars"],
        "compress_ms": round(comp_ms, 1),
        "compress_ratio_str": meta.get("ratio_str", ""),
        "compress_origin_tokens": meta.get("origin_tokens"),
        "compress_compressed_tokens": meta.get("compressed_tokens"),
        "gemini_ms": gr["latency_ms"],
        "gemini_input_tokens": gr.get("input_tokens"),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "pred": pred,
        "correct": 1 if pred == gold else 0,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

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

    print("=== Opt P5: Rate Tuning ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars")
    print(f"  gold:     {gold}")
    print(f"  k:        {args.k}")
    print(f"  rates:    [0.3, 0.5, 0.7]")
    print()

    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return 1
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite"))

    # Load prior baseline
    prior_path = REPO_ROOT / "results" / "phase-01-3-approach-1case.json"
    if prior_path.exists():
        with prior_path.open() as fh:
            prior = json.load(fh)
        prior_records = {r["approach"]: r for r in prior["records"]}
        baseline = prior_records.get("baseline", {})
    else:
        prior_records = {}
        baseline = {}

    print(f"  Reference: baseline total={baseline.get('total_ms','?')} ms")
    print()

    RATES = [0.3, 0.5, 0.7]
    records = []

    for rate in RATES:
        print(f"--- rate={rate} ---")
        rec = run_combo(context, question, choices, gold, gemini, args.k, rate)
        rec["ts"] = now_iso()
        rec["case_id"] = row.get("_id", "")
        records.append(rec)

        if rec["status"] == "ok":
            print(f"  presel={rec['preselect_ms']:>7.0f} ms  "
                  f"chars={rec['preselect_chars']:>6,}  "
                  f"compress={rec['compress_ms']:>6.0f} ms  "
                  f"ratio={rec['compress_ratio_str']}  "
                  f"gemini={rec['gemini_ms']:>6.0f} ms  "
                  f"total={rec['total_ms']:>8.0f} ms  "
                  f"pred={rec['pred']} correct={rec['correct']}")
        else:
            print(f"  ERROR: {rec.get('error','?')}")
        time.sleep(SLEEP_BETWEEN)
        print()

    # Table
    print("=== Summary sorted by total_ms ===")
    print(f"{'rate':>5} {'comp_ms':>8} {'gemin_ms':>9} {'total_ms':>10} {'tok_in':>7} {'pred':>4} {'acc':>4}")
    print("-" * 56)
    for rec in sorted(records, key=lambda r: r["total_ms"]):
        if rec["status"] == "ok":
            print(
                f"  {rec['rate']:>5.2f} "
                f"{rec['compress_ms']:>8.0f} "
                f"{rec['gemini_ms']:>9.0f} "
                f"{rec['total_ms']:>10.0f} "
                f"{rec.get('gemini_input_tokens',0) or 0:>7} "
                f"{rec['pred']:>4} "
                f"{rec['correct']:>4}"
            )

    # Delta vs rate=0.5
    r5 = next((r for r in records if abs(r["rate"] - 0.5) < 0.01), None)
    if r5:
        print()
        print("=== Delta vs rate=0.5 ===")
        baseline_total = baseline.get("total_ms", 0)
        for rec in records:
            if rec["status"] != "ok":
                continue
            delta_vs_5 = rec["total_ms"] - r5["total_ms"]
            delta_vs_base = rec["total_ms"] - baseline_total
            acc_delta = (rec.get("correct", 0) - r5.get("correct", 0)) * 100
            print(
                f"  rate={rec['rate']:.2f}  delta_vs_0.5={delta_vs_5:>+7.0f} ms  "
                f"delta_vs_base={delta_vs_base:>+8.0f} ms  acc_delta_vs_0.5={acc_delta:+5.0f}pp"
            )

    # Verdict
    print()
    print("=== Verdict ===")
    correct_recs = [r for r in records if r.get("correct") == 1]
    if correct_recs:
        best = min(correct_recs, key=lambda r: r["total_ms"])
        worst = max(correct_recs, key=lambda r: r["total_ms"])
        ms_saved = worst["total_ms"] - best["total_ms"]
        best_rate = best["rate"]

        print(f"  Best correct: rate={best_rate} at {best['total_ms']:.0f} ms")
        print(f"  Worst correct: rate={worst['rate']} at {worst['total_ms']:.0f} ms")
        print(f"  Saved by tuning: {ms_saved:.0f} ms")

        if ms_saved > 500:
            print(f"  DECISION: IMPROVE -- change rate from 0.5 to {best_rate}")
            verdict = "IMPROVE"
        else:
            print(f"  DECISION: NO_IMPROVEMENT -- rate=0.5 already near-optimal")
            verdict = "NO_IMPROVEMENT"
    else:
        verdict = "ERROR_NO_CORRECT"
        best_rate = 0.5

    print()

    # Save
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
        "args": {"k": args.k, "rates": RATES},
        "records": records,
        "verdict": verdict,
        "best_rate": best_rate,
    }
    with args.out_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
