#!/usr/bin/env python3
"""
Optimization experiment P2: Tune top-k preselect parameter.

Hypothesis:
  Smaller k (10 vs 20) reduces compressor input size -> faster compress.
  Trade-off: may lose recall if answer-relevant sentence is excluded.

Method:
  Run Approach C (preselect + compress) with k=5, 10, 15, 20 on same case.
  Measure: preselect_chars, compress_ms, total_ms, accuracy.
  Also run approach B (preselect only) at each k to isolate effect.

Usage:
  conda activate vsf
  python scripts/phase-01/opt_p2_k_tuning.py [--rate 0.5]
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

DEFAULT_OUT_JSONL = REPO_ROOT / "results" / "phase-01-opt-p2-k-tuning.jsonl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results" / "phase-01-opt-p2-k-tuning.json"
SLEEP_BETWEEN = 1.5

from compare_3_approaches import (
    load_shortest_case,
    preselect_top_k, compress_llmlingua,
    build_qa_prompt, call_gemini, normalize_pred, now_iso,
)


def run_preselect_only(context, question, choices, gold, gemini, k):
    """Run approach B at given k."""
    t0 = time.perf_counter()
    ps = preselect_top_k(context, question, k=k)
    ctx = ps["selected_text"]
    prompt = build_qa_prompt(ctx, question, choices)
    gr = call_gemini(gemini, prompt)
    pred = normalize_pred(gr["text"])
    return {
        "approach": f"B_k{k}",
        "k": k,
        "preselect_ms": ps["total_ms"],
        "preselect_chars": ps["selected_chars"],
        "compress_ms": 0,
        "gemini_ms": gr["latency_ms"],
        "gemini_input_tokens": gr.get("input_tokens"),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "pred": pred,
        "correct": 1 if pred == gold else 0,
        "status": "ok",
    }


def run_combo(context, question, choices, gold, gemini, k, rate):
    """Run approach C at given k."""
    t0 = time.perf_counter()
    ps = preselect_top_k(context, question, k=k)
    ctx = ps["selected_text"]
    comp_text, comp_ms, meta = compress_llmlingua(ctx, question, rate=rate)
    prompt = build_qa_prompt(comp_text, question, choices)
    gr = call_gemini(gemini, prompt)
    pred = normalize_pred(gr["text"])
    return {
        "approach": f"C_k{k}",
        "k": k,
        "rate": rate,
        "preselect_ms": ps["total_ms"],
        "preselect_chars": ps["selected_chars"],
        "compress_ms": round(comp_ms, 1),
        "gemini_ms": gr["latency_ms"],
        "gemini_input_tokens": gr.get("input_tokens"),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "pred": pred,
        "correct": 1 if pred == gold else 0,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.5)
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

    print("=== Opt P2: Top-k Tuning ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars")
    print(f"  gold:     {gold}")
    print(f"  rate:     {args.rate}")
    print(f"  k values: [5, 10, 15, 20]")
    print()

    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return 1
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite"))

    K_VALUES = [5, 10, 15, 20]
    records = []

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

    print(f"  Reference: baseline total={baseline.get('total_ms','?')} ms, "
          f"C_combo k=20 total={prior_records.get('C_combo',{}).get('total_ms','?')} ms")
    print()

    # Run B and C for each k
    for k in K_VALUES:
        # Approach B: preselect only
        rec_b = run_preselect_only(
            context, question, choices, gold, gemini, k
        )
        rec_b["ts"] = now_iso()
        rec_b["case_id"] = row.get("_id", "")
        records.append(rec_b)
        print(
            f"  [B_k{k} (skip)]  "
            f"presel={rec_b['preselect_ms']:>7.0f} ms  "
            f"chars={rec_b['preselect_chars']:>6,}  "
            f"compress=     0 ms  "
            f"gemini={rec_b['gemini_ms']:>6.0f} ms  "
            f"total={rec_b['total_ms']:>8.0f} ms  "
            f"pred={rec_b['pred']} correct={rec_b['correct']}"
        )
        time.sleep(SLEEP_BETWEEN)

        # Approach C: preselect + compress
        rec_c = run_combo(
            context, question, choices, gold, gemini, k, rate=args.rate
        )
        rec_c["ts"] = now_iso()
        rec_c["case_id"] = row.get("_id", "")
        records.append(rec_c)
        print(
            f"  [C_k{k}]        "
            f"presel={rec_c['preselect_ms']:>7.0f} ms  "
            f"chars={rec_c['preselect_chars']:>6,}  "
            f"compress={rec_c['compress_ms']:>6.0f} ms  "
            f"gemini={rec_c['gemini_ms']:>6.0f} ms  "
            f"total={rec_c['total_ms']:>8.0f} ms  "
            f"pred={rec_c['pred']} correct={rec_c['correct']}"
        )
        time.sleep(SLEEP_BETWEEN)
        print()

    # Table: all configs sorted by total_ms
    print("=== All configs sorted by total_ms ===")
    print(f"{'Approach':<12} {'k':>3} {'PreSel ms':>10} {'Comp ms':>8} {'Gemini ms':>10} {'Total ms':>10} {'Chars':>7} {'Tokens':>7} {'Pred':>4} {'OK?':>4}")
    print("-" * 90)
    sorted_recs = sorted(records, key=lambda r: r["total_ms"])
    for rec in sorted_recs:
        k = rec.get("k", "?")
        approach = rec.get("approach", "?")
        print(
            f"  {approach:<12} "
            f"{k:>3} "
            f"{rec.get('preselect_ms',0):>10.0f} "
            f"{rec.get('compress_ms',0):>8.0f} "
            f"{rec.get('gemini_ms',0):>10.0f} "
            f"{rec.get('total_ms',0):>10.0f} "
            f"{rec.get('preselect_chars',0):>7,} "
            f"{rec.get('gemini_input_tokens',0) or 0:>7} "
            f"{rec.get('pred','?'):>4} "
            f"{rec.get('correct','?'):>4}"
        )

    # Best per approach type
    print()
    print("=== Best per approach type ===")
    for prefix in ["B", "C"]:
        subset = [r for r in records if r["approach"].startswith(prefix)]
        if not subset:
            continue
        best = min(subset, key=lambda r: r["total_ms"])
        worst = max(subset, key=lambda r: r["total_ms"])
        k_best = best["k"]
        k_worst = worst["k"]
        ms_diff = worst["total_ms"] - best["total_ms"]
        acc_diff = best["correct"] - worst["correct"]
        print(f"  {prefix}: best k={k_best} ({best['total_ms']:.0f} ms), "
              f"worst k={k_worst} ({worst['total_ms']:.0f} ms), "
              f"delta={ms_diff:.0f} ms, acc_delta={acc_diff}")

    # Delta vs baseline and vs C_k20
    print()
    print("=== Delta vs Baseline ===")
    baseline_total = baseline.get("total_ms", 0)
    c_k20 = next((r for r in records if r["approach"] == "C_k20"), None)
    c_k20_total = c_k20["total_ms"] if c_k20 else 0
    for rec in sorted_recs:
        delta_vs_base = rec["total_ms"] - baseline_total
        delta_vs_k20 = rec["total_ms"] - c_k20_total if rec["approach"].startswith("C") else None
        acc_delta = (rec.get("correct", 0) - baseline.get("correct", 0)) * 100 if baseline else 0
        delta_str = f"vs_base={delta_vs_base:>+7.0f} ms"
        if delta_vs_k20 is not None:
            delta_str += f", vs_C_k20={delta_vs_k20:>+6.0f} ms"
        verdict = "OK" if rec["correct"] == 1 else "FAIL"
        print(f"  {rec['approach']:<12} {delta_str}  acc_delta={acc_delta:>+5.0f}pp {verdict}")

    # Verdict
    print()
    print("=== Verdict ===")
    # Find best C (fastest with accuracy=1)
    correct_c = [r for r in records if r["approach"].startswith("C") and r["correct"] == 1]
    if correct_c:
        best_c = min(correct_c, key=lambda r: r["total_ms"])
        worst_c = max(correct_c, key=lambda r: r["total_ms"])
        ms_saved = worst_c["total_ms"] - best_c["total_ms"]
        k_improvement = best_c["k"]
        print(f"  Best correct C: {best_c['approach']} at {best_c['total_ms']:.0f} ms (k={k_improvement})")
        print(f"  Worst correct C: {worst_c['approach']} at {worst_c['total_ms']:.0f} ms (k={worst_c['k']})")
        print(f"  Best k saves: {ms_saved:.0f} ms vs worst k")
        if ms_saved > 500:
            print(f"  DECISION: IMPROVE -- change k from 20 to {k_improvement}")
            verdict = "IMPROVE"
            best_k = k_improvement
        else:
            print(f"  DECISION: NO_IMPROVEMENT -- k tuning benefit < 500ms")
            verdict = "NO_IMPROVEMENT"
            best_k = 20
    else:
        verdict = "ERROR_NO_CORRECT"
        best_k = 20

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
        "args": {"rate": args.rate, "k_values": K_VALUES},
        "records": records,
        "verdict": verdict,
        "best_k": best_k,
    }
    with args.out_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
