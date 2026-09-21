#!/usr/bin/env python3
"""
Optimization experiment P1: Tune SentenceTransformer batch size.

Hypothesis:
  Higher batch_size better utilizes CPU SIMD. Try 32, 64, 128, 256.
  Goal: maximize sentences_per_sec.

Method:
  Run preselect on the same case (Legal 70k chars, ~381 sentences)
  at 4 batch sizes. Measure embed_ms and sentences_per_sec.
  The best batch size is committed to compare_3_approaches.py.

Usage:
  conda activate vsf
  python scripts/phase-01/opt_p1_batch_size.py
"""
from __future__ import annotations

import argparse
import json
import re
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

DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-opt-p1-batch-size.json"

# Reuse from compare_3_approaches
from compare_3_approaches import (
    load_shortest_case, split_sentences, _get_sentence_model, now_iso,
)


def bench_batch_size(context: str, batch_size: int) -> dict:
    """
    Measure embed time at a given batch_size.
    Returns dict with timings.
    """
    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    model = _get_sentence_model()

    # Warmup: encode 1 batch to ensure model is in cache
    _ = model.encode(sentences[:1], show_progress_bar=False)

    # Measure
    t0 = time.perf_counter()
    embeddings = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=False
    )
    embed_ms = (time.perf_counter() - t0) * 1000.0

    n = len(sentences)
    sents_per_sec = n / max(embed_ms / 1000.0, 0.001)
    chars_per_sec = sum(len(s) for s in sentences) / max(embed_ms / 1000.0, 0.001)

    return {
        "status": "ok",
        "batch_size": batch_size,
        "n_sentences": n,
        "embed_ms": round(embed_ms, 1),
        "sentences_per_sec": round(sents_per_sec, 1),
        "chars_per_sec": round(chars_per_sec, 0),
        "embedding_shape": list(embeddings.shape) if hasattr(embeddings, "shape") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[32, 64, 128, 256],
        help="Batch sizes to test",
    )
    args = parser.parse_args()

    row = load_shortest_case()
    context = row.get("context", "")

    print("=== Opt P1: SentenceTransformer Batch Size ===")
    print(f"  case:    {row.get('_id','?')}")
    print(f"  context: {len(context):,} chars")
    print(f"  sizes:   {args.sizes}")
    print()

    # Pre-load model once
    print("Loading SentenceTransformer (cold)...")
    t0 = time.perf_counter()
    _get_sentence_model()
    cold_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Cold load: {cold_load_ms:.0f} ms")
    print()

    results = []
    for bs in args.sizes:
        # 2 runs each; use 2nd as warm-cache reading
        for run_i in range(2):
            r = bench_batch_size(context, batch_size=bs)
            r["run"] = run_i + 1
            r["cold_load_ms_global"] = round(cold_load_ms, 1)
            results.append(r)

            if r["status"] == "ok":
                tag = "(warm)" if run_i == 1 else "(cold)"
                print(f"  batch={bs:>4} run={run_i+1} {tag:<8} "
                      f"embed={r['embed_ms']:>7.0f} ms "
                      f"({r['sentences_per_sec']:>5.1f} sents/s)")
            else:
                print(f"  batch={bs:>4} run={run_i+1} ERROR: {r.get('error')}")

    print()

    # Aggregate: take run=2 (warm) for each batch size
    print("=== Best batch size (warm cache) ===")
    by_bs_warm = {}
    for r in results:
        if r["status"] == "ok" and r["run"] == 2:
            by_bs_warm[r["batch_size"]] = r

    baseline_bs = 64
    baseline_warm = by_bs_warm.get(baseline_bs)
    print(f"{'batch':>8} {'embed_ms':>10} {'sents/s':>10} {'speedup':>10} {'delta_ms':>10}")
    print("-" * 56)

    best_bs = baseline_bs
    best_throughput = 0
    for bs in sorted(by_bs_warm.keys()):
        r = by_bs_warm[bs]
        speedup = baseline_warm["embed_ms"] / max(r["embed_ms"], 0.1) if baseline_warm else 1.0
        delta = r["embed_ms"] - baseline_warm["embed_ms"] if baseline_warm else 0
        print(
            f"  {bs:>6} "
            f"{r['embed_ms']:>10.0f} "
            f"{r['sentences_per_sec']:>10.1f} "
            f"{speedup:>10.2f}x "
            f"{delta:>+10.0f}"
        )
        if r["sentences_per_sec"] > best_throughput:
            best_throughput = r["sentences_per_sec"]
            best_bs = bs

    print()

    # Decision
    if baseline_warm:
        baseline_sents_per_sec = baseline_warm["sentences_per_sec"]
        best_throughput_val = by_bs_warm[best_bs]["sentences_per_sec"]
        improvement_pct = (
            (best_throughput_val - baseline_sents_per_sec) / baseline_sents_per_sec * 100
        )
        # Threshold: only commit if >= 10% improvement
        VERDICT_THRESHOLD_PCT = 10.0
        verdict = "IMPROVE" if improvement_pct >= VERDICT_THRESHOLD_PCT else "NO_IMPROVEMENT"
        print(f"=== Verdict ===")
        print(f"  Baseline (batch=64):     {baseline_sents_per_sec:.1f} sents/s")
        print(f"  Best    (batch={best_bs}):       {best_throughput_val:.1f} sents/s")
        print(f"  Improvement:             {improvement_pct:+.1f}%")
        print(f"  Threshold:               >= {VERDICT_THRESHOLD_PCT}%")
        print(f"  DECISION:                {verdict}")
        if verdict == "IMPROVE":
            print(f"  ACTION:                  change batch_size 64 -> {best_bs}")
        else:
            print(f"  ACTION:                  keep batch_size=64")
    else:
        verdict = "ERROR"
        improvement_pct = 0
        best_bs = 64

    print()

    # Save
    out_data = {
        "ts": now_iso(),
        "case_id": row.get("_id", ""),
        "context_chars": len(context),
        "cold_load_ms": round(cold_load_ms, 1),
        "results": results,
        "by_bs_warm": by_bs_warm,
        "best_batch_size": best_bs,
        "improvement_pct_vs_64": round(improvement_pct, 1) if baseline_warm else None,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
