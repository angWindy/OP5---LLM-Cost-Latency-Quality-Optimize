#!/usr/bin/env python3
"""
Phase 1 — Compressor ablation NHANH: 3 configs trên 1 case.

Thay vì full grid (3×3×3=27 runs), chỉ test 3 configs để xác định:
  1. iter_size=1024 vs 200: speed gain
  2. rate=0.3 vs 0.5: token savings

Script này test 3 configs lần lượt (chạy nhanh hơn ~10x so với full grid).

Usage:
    conda activate vsf
    python scripts/phase-01/compressor_ablation_1case.py
"""
from __future__ import annotations

import argparse
import json
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
from smoke_both_paths import _get_llmlingua2, COMPRESSOR_MODEL


DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-compressor-ablation-1case.json"


# 3 configs: [iter_size, rate, ft_name]
CONFIGS = [
    # Config 1: paper default (baseline speed reference)
    {"iter_size": 200,  "rate": 0.5, "ft": ["!", ".", "?", "\n"], "ft_name": "ft_basic"},
    # Config 2: larger chunk = faster
    {"iter_size": 1024, "rate": 0.5, "ft": ["!", ".", "?", "\n"], "ft_name": "ft_basic"},
    # Config 3: aggressive compression
    {"iter_size": 1024, "rate": 0.3, "ft": ["!", ".", "?", "\n"], "ft_name": "ft_basic"},
]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compress_with(pc, context, question, cfg):
    t0 = time.perf_counter()
    result = pc.compress_prompt(
        context,
        question=question,
        rate=cfg["rate"],
        force_tokens=cfg["ft"],
        drop_consecutive=True,
        return_word_label=False,
        iterative_size=cfg["iter_size"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    origin = result.get("origin_tokens") or 0
    compressed = result.get("compressed_tokens") or 0
    return {
        "config": {**cfg, "ft": cfg["ft"]},  # drop ft list from config for readability
        "origin_tokens": origin,
        "compressed_tokens": compressed,
        "ratio": result.get("ratio", ""),
        "compressor_ms": round(elapsed_ms, 2),
        "compressed_chars": len(result.get("compressed_prompt", "")),
        "token_saved_pct": round((1 - compressed / max(origin, 1)) * 100, 2) if origin else None,
        "chars_per_sec": round(len(context) / (elapsed_ms / 1000), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"ERROR: {args.test_file} not found.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.test_file.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    row = rows[args.case_index]
    context = row.get("context", "")[: args.max_context_chars]
    question = row.get("question", "")
    case_id = row.get("_id", "?")

    print(f"=== Compressor ablation: 3 configs (fast) ===")
    print(f"  case: _id={case_id[:8]}  domain={row.get('domain', '?')}")
    print(f"  context_chars: {len(context):,}")
    print(f"  configs: {[c['ft_name'] for c in CONFIGS]}")
    print()

    # Load compressor once
    print("Loading compressor (one-time)...")
    t = time.perf_counter()
    pc = _get_llmlingua2()
    print(f"  loaded in {(time.perf_counter()-t)*1000:.0f}ms")
    print()

    results = []
    for i, cfg in enumerate(CONFIGS, 1):
        tag = f"iter={cfg['iter_size']:>4}  rate={cfg['rate']}  {cfg['ft_name']}"
        print(f"[{i}/3] Running {tag}...", end=" ", flush=True)
        try:
            r = compress_with(pc, context, question, cfg)
            results.append(r)
            print(f"DONE  "
                  f"{r['origin_tokens']:>6}->{r['compressed_tokens']:>6} tokens "
                  f"({r['token_saved_pct']:>+5.1f}%)  "
                  f"{r['compressor_ms']:>6.0f}ms  "
                  f"{r['chars_per_sec']:>6.0f} cps")
        except Exception as exc:
            results.append({"config": cfg, "error": str(exc)})
            print(f"ERROR: {exc}")

    # Analysis
    print()
    print("=== Analysis ===")
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No successful runs.")
        return 3

    # Compare iter_size speed at same rate=0.5
    same_rate = [r for r in valid if r["config"]["rate"] == 0.5]
    if len(same_rate) >= 2:
        same_rate.sort(key=lambda r: r["config"]["iter_size"])
        r_slow, r_fast = same_rate[0], same_rate[1]
        speedup = r_slow["compressor_ms"] / r_fast["compressor_ms"]
        print(f"  iter_size speedup (200->1024 @ rate=0.5): {speedup:.1f}x faster")
        print(f"    200: {r_slow['compressor_ms']:.0f}ms  |  1024: {r_fast['compressor_ms']:.0f}ms")

    # Compare rate compression at same iter_size=1024
    same_iter = [r for r in valid if r["config"]["iter_size"] == 1024]
    if len(same_iter) >= 2:
        same_iter.sort(key=lambda r: r["config"]["rate"])
        r_loose, r_aggr = same_iter[0], same_iter[-1]
        print(f"  rate token savings (0.5->0.3 @ iter=1024):")
        print(f"    rate=0.5: {r_loose['token_saved_pct']:+.1f}% saved  ({r_loose['compressed_tokens']} tokens)")
        print(f"    rate=0.3: {r_aggr['token_saved_pct']:+.1f}% saved  ({r_aggr['compressed_tokens']} tokens)")

    # Best by token savings
    best_save = max(valid, key=lambda r: r["token_saved_pct"])
    # Best by speed
    best_speed = min(valid, key=lambda r: r["compressor_ms"])

    print()
    print("  Best token savings : "
          f"iter={best_save['config']['iter_size']} rate={best_save['config']['rate']} "
          f"-> {best_save['token_saved_pct']:+.1f}% in {best_save['compressor_ms']:.0f}ms")
    print("  Fastest           : "
          f"iter={best_speed['config']['iter_size']} rate={best_speed['config']['rate']} "
          f"-> {best_speed['compressor_ms']:.0f}ms  ({best_speed['token_saved_pct']:+.1f}% saved)")

    # Net time recommendation
    # From diag_llm_call.py: baseline Gemini = 1579ms for 18500 tokens
    # Gemini latency ~ proportional to input_tokens (65% of 1579ms is server inference)
    # Net = compressor_ms + est_gemini_saved
    print()
    print("=== Net time vs baseline (Gemini only) ===")
    baseline_gemini_ms = 1579  # from diag
    for r in valid:
        # Est Gemini time for compressed tokens
        est_gemini = (r['compressed_tokens'] / r['origin_tokens']) * baseline_gemini_ms
        net = r['compressor_ms'] + est_gemini
        delta = net - baseline_gemini_ms
        print(f"  iter={r['config']['iter_size']:>4} rate={r['config']['rate']}  "
              f"net={net:>6.0f}ms  (baseline={baseline_gemini_ms}ms  delta={delta:>+6.0f}ms)  "
              f"{'FASTER' if delta < 0 else 'SLOWER'}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump({
            "ts": now_iso(),
            "model": COMPRESSOR_MODEL,
            "case": {
                "_id": case_id,
                "domain": row.get("domain"),
                "context_chars": len(context),
            },
            "configs_tested": [c["ft_name"] for c in CONFIGS],
            "results": results,
        }, fh, ensure_ascii=False, indent=2)

    print()
    print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
