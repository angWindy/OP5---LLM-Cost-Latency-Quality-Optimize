#!/usr/bin/env python3
"""
Tune LLMLingua-2 knobs on 1 row (Legal 70k).

Grid: 3 rates x 2 force_tokens = 6 configs.
For each config, measure:
  - ratio_str (from model, not char-based)
  - origin_tokens / compressed_tokens
  - compressor_latency_ms
  - content_surrogate: how many answer-relevant keywords survive

Select best config: highest compression ratio, latency < 10s (CPU).

Usage:
    conda activate vsf
    python scripts/phase-01/tune_knobs.py [--out results/phase-01-tune-knobs.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_both_paths import compress_pip_direct, chunk_context_for_info_only

DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-tune-knobs.json"

RATES = [0.2, 0.33, 0.5]
FORCE_TOKEN_SETS = {
    "basic": ["!", ".", "?", "\n"],
    "extended": ["!", ".", "?", "\n", ":", ";", ",", "(", ")", "[", "]"],
}
LEGAL_SURROGATE_KEYWORDS = [
    "disaster", "legal", "rights", "international", "preliminary",
    "challenges", "human", "commission", "rapporteur",
]


def content_surrogate(compressed: str, keywords: list[str]) -> dict:
    text_lower = compressed.lower()
    present = [kw for kw in keywords if kw.lower() in text_lower]
    return {
        "keywords_present": len(present),
        "total_keywords": len(keywords),
        "fraction": round(len(present) / max(len(keywords), 1), 3),
        "present_keywords": present,
    }


def run_config(context, question, gold, rate, force_tokens, row_id, config_name):
    try:
        compressed, elapsed_ms, meta = compress_pip_direct(
            context, question, rate=rate, force_tokens=force_tokens
        )
        ratio_str = meta.get("ratio_str", "N/A")
        try:
            ratio_val = float(ratio_str.replace("x", ""))
        except (ValueError, AttributeError):
            ratio_val = None
        surrogate = content_surrogate(compressed, LEGAL_SURROGATE_KEYWORDS)
        return {
            "config_name": config_name,
            "rate": rate,
            "force_tokens": force_tokens,
            "origin_tokens": meta.get("origin_tokens"),
            "compressed_tokens": meta.get("compressed_tokens"),
            "ratio_str": ratio_str,
            "ratio_val": ratio_val,
            "origin_chars": meta.get("origin_chars"),
            "compressed_chars": meta.get("compressed_chars"),
            "compressor_ms": round(elapsed_ms, 1),
            "surrogate": surrogate,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "config_name": config_name,
            "rate": rate,
            "force_tokens": force_tokens,
            "status": "error",
            "error": str(exc),
        }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    test_path = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    row = rows[0]
    context = row["context"]
    question = row.get("question", "")
    gold = row.get("answer", "")
    row_id = row.get("_id", "?")

    print("=== Tune knobs: LLMLingua-2 on Legal row ===")
    print(f"  row:     {row_id}")
    print(f"  context: {len(context):,} chars")
    print(f"  gold:    {gold}")
    print(f"  rates:   {RATES}")
    print(f"  ftoken_sets: {list(FORCE_TOKEN_SETS.keys())}")
    print()

    results = []
    for rate in RATES:
        for ft_name, ft_list in FORCE_TOKEN_SETS.items():
            cfg_name = f"rate{rate}_ft_{ft_name}"
            print(f"  Running {cfg_name} ...", end=" ", flush=True)
            result = run_config(context, question, gold, rate, ft_list, row_id, cfg_name)
            results.append(result)
            if result["status"] == "ok":
                print(
                    f"ratio={result['ratio_str']} "
                    f"tokens={result['origin_tokens']}->{result['compressed_tokens']} "
                    f"ms={result['compressor_ms']:.0f} "
                    f"kw={result['surrogate']['keywords_present']}/{result['surrogate']['total_keywords']}"
                )
            else:
                print(f"ERROR: {result.get('error', '?')}")

    print()
    valid = [r for r in results if r["status"] == "ok"]
    valid.sort(key=lambda r: (r.get("ratio_val") or 0, -r.get("compressor_ms", 9999)))
    best = valid[0] if valid else None

    if best:
        print("=== Best config ===")
        print(f"  {best['config_name']}: ratio={best['ratio_str']} "
              f"tokens={best['origin_tokens']}->{best['compressed_tokens']} "
              f"latency={best['compressor_ms']:.0f}ms "
              f"kw={best['surrogate']['keywords_present']}/{best['surrogate']['total_keywords']}")
        print()
        print("=== All results ===")
        for r in sorted(results, key=lambda x: x.get("ratio_val") or 0, reverse=True):
            if r["status"] == "ok":
                print(
                    f"  {r['config_name']:30s} "
                    f"ratio={r['ratio_str']:8s} "
                    f"tokens={r.get('origin_tokens','?'):>6}->{r.get('compressed_tokens','?'):>6} "
                    f"ms={r['compressor_ms']:>7.0f} "
                    f"kw={r['surrogate']['keywords_present']}/{r['surrogate']['total_keywords']}"
                )
            else:
                print(f"  {r['config_name']:30s} ERROR: {r.get('error','?')}")
    else:
        print("No valid results.")

    out_data = {
        "best_config": best,
        "all_results": results,
        "meta": {
            "row_id": row_id,
            "context_chars": len(context),
            "gold": gold,
            "rates": RATES,
            "force_token_sets": list(FORCE_TOKEN_SETS.keys()),
            "latency_threshold_s": 10,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
