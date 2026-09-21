#!/usr/bin/env python3
"""
Compute aggregated summary from results/phase-01-llmlingua-5cases.jsonl.

Reads existing JSONL (per-config records), groups by case_id and config, and
emits the same shape as eval_llmlingua_5.py's summary block:
  - by_config (avg metrics + accuracy)
  - delta_vs_baseline (% token saved, % time saved, delta accuracy pp)

Usage:
    python scripts/phase-01/compute_summary.py [--in PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "results" / "phase-01-llmlingua-5cases.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-llmlingua-5cases-summary.json"

CONFIGS = ("baseline", "compressed_pip", "compressed_lc")


def stat(xs: list) -> float | None:
    return round(statistics.mean(xs), 1) if xs else None


def aggregate(records: list[dict]) -> dict:
    """Group records by (case_id, config). Compute per-config averages."""
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r.get("status") != "ok":
            continue
        by_case[r["case_id"]][r["config"]] = r

    summary = {"by_config": {}, "delta_vs_baseline": {}, "per_case": {}}

    for config in CONFIGS:
        tokens, total_lat, gem_lat, comp_lat, accs, ratios = [], [], [], [], [], []
        n = 0
        for case_id, cfg_dict in by_case.items():
            r = cfg_dict.get(config)
            if not r:
                continue
            if r.get("input_tokens") is not None:
                tokens.append(r["input_tokens"])
            if r.get("total_latency_ms") is not None:
                total_lat.append(r["total_latency_ms"])
            if r.get("gemini_latency_ms") is not None:
                gem_lat.append(r["gemini_latency_ms"])
            if r.get("compressor_latency_ms") is not None:
                comp_lat.append(r["compressor_latency_ms"])
            if r.get("accuracy") is not None:
                accs.append(r["accuracy"])
            if r.get("compression_ratio") is not None:
                ratios.append(r["compression_ratio"])
            n += 1

        summary["by_config"][config] = {
            "n_cases": n,
            "avg_input_tokens": stat(tokens),
            "avg_gemini_latency_ms": stat(gem_lat),
            "avg_compressor_latency_ms": stat(comp_lat),
            "avg_total_latency_ms": stat(total_lat),
            "accuracy": stat(accs),
            "avg_compression_ratio": stat(ratios),
        }

    # Per-case table (compact)
    for case_id, cfg_dict in sorted(by_case.items()):
        row = {}
        for config in CONFIGS:
            r = cfg_dict.get(config)
            if not r:
                row[config] = None
                continue
            row[config] = {
                "tokens": r.get("input_tokens"),
                "comp_ratio": r.get("compression_ratio"),
                "comp_ms": r.get("compressor_latency_ms"),
                "gem_ms": r.get("gemini_latency_ms"),
                "total_ms": r.get("total_latency_ms"),
                "accuracy": r.get("accuracy"),
                "pred": r.get("pred"),
                "gold": r.get("gold"),
            }
        summary["per_case"][case_id] = row

    # Delta vs baseline
    b = summary["by_config"]["baseline"]
    for config in ("compressed_pip", "compressed_lc"):
        c = summary["by_config"][config]
        token_saved_pct = None
        if b["avg_input_tokens"] and c["avg_input_tokens"]:
            token_saved_pct = round(
                (b["avg_input_tokens"] - c["avg_input_tokens"]) / b["avg_input_tokens"] * 100, 1
            )
        time_saved_pct = None
        if b["avg_total_latency_ms"] and c["avg_total_latency_ms"]:
            time_saved_pct = round(
                (b["avg_total_latency_ms"] - c["avg_total_latency_ms"]) / b["avg_total_latency_ms"] * 100, 1
            )
        accuracy_delta_pp = None
        if c["accuracy"] is not None and b["accuracy"] is not None:
            accuracy_delta_pp = round((c["accuracy"] - b["accuracy"]) * 100, 1)
        summary["delta_vs_baseline"][config] = {
            "token_saved_pct": token_saved_pct,
            "time_saved_pct": time_saved_pct,
            "accuracy_delta_pp": accuracy_delta_pp,
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"ERROR: {args.inp} not found.", file=sys.stderr)
        return 2

    records = [json.loads(l) for l in args.inp.read_text().splitlines() if l.strip()]
    summary = aggregate(records)
    summary["meta"] = {
        "source_jsonl": str(args.inp),
        "n_records": len(records),
        "n_ok": sum(1 for r in records if r.get("status") == "ok"),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
