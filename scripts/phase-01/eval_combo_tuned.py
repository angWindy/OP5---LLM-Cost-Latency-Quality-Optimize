#!/usr/bin/env python3
"""
Phase 1 v4 — Combo Tuning: trade token reduction for accuracy.

Tests multiple (k, rate) combos on the same 11 cases that ran in v3.
Goal: find a sweet spot where accuracy >= baseline (45.5%) with <96% token reduction.

Configs (all = preselect + LLMLingua-2 compress, only k and rate vary):
  C0: k=20, rate=0.5  (v3 baseline -- accuracy 45.5%, 811 tok)
  C1: k=50, rate=0.7  (more content kept -- expected more tokens, better accuracy)
  C2: k=30, rate=0.6  (middle ground)
  C3: k=50, rate=0.5  (more preselect, same compress)

Output:
  results/phase-01-combo-tuned.jsonl
  results/phase-01-combo-tuned-summary.json

Usage:
  conda activate vsf
  python scripts/phase-01/eval_combo_tuned.py
"""
from __future__ import annotations

import argparse
import json
import os
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

from eval_mistral_combo_3 import (
    call_mistral, build_prompt, normalize_pred,
    compress_llmlingua2, preselect_bm25,
    get_llmlingua2, get_mistral_client,
    FORCE_TOKENS, MISTRAL_MODEL, SLEEP, now_iso,
)

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "dev_first95.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-combo-tuned.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-combo-tuned-summary.json"

# Configs to test: (name, preselect_k, compress_rate)
TUNING_CONFIGS = [
    ("C0_k20_r05", 20, 0.5),  # v3 baseline
    ("C1_k50_r07", 50, 0.7),  # primary recommendation
    ("C2_k30_r06", 30, 0.6),  # middle ground
    ("C3_k50_r05", 50, 0.5),  # more preselect, same compress
]


def load_n_cases(n: int, max_ctx: int) -> list[dict]:
    """Same as eval_combo_n15.load_n_cases but exposed inline."""
    rows: list[dict] = []

    def _safe_read(p: Path) -> list[dict]:
        if not p.exists():
            return []
        out = []
        with p.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return out

    rows.extend(_safe_read(DEFAULT_TEST))
    rows.extend(_safe_read(DEFAULT_DEV))

    seen = set()
    unique = []
    for r in rows:
        rid = r.get("_id")
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(r)
    rows = unique
    rows.sort(key=lambda r: len(r.get("context", "")))
    selected = [r for r in rows if len(r.get("context", "")) <= max_ctx][:n]
    for i, r in enumerate(selected):
        r["_case_idx"] = i
    return selected


def run_one_config(case_id: str, row: dict, client, pc, cfg_name: str, k: int, rate: float, out_fh) -> dict:
    """Run a single (k, rate) config on one case."""
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k_: row.get(f"choice_{k_}", "") for k_ in "ABCD"}
    domain = row.get("domain", "?")

    print(f"[{case_id}] {cfg_name} (k={k}, rate={rate})")
    ps = preselect_bm25(context, question, k=k)
    if ps["status"] != "ok":
        rec = {
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": cfg_name, "k": k, "rate": rate, "status": "preselect_error",
            "error": ps.get("error", "")[:200], "gold": gold, "domain": domain,
            "context_chars": len(context),
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        return rec

    cc = compress_llmlingua2(pc, ps["selected_text"], rate=rate)
    if cc["status"] != "ok":
        rec = {
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": cfg_name, "k": k, "rate": rate, "status": "compress_error",
            "error": cc.get("error", "")[:200], "gold": gold, "domain": domain,
            "context_chars": len(context),
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        return rec

    prompt = build_prompt(cc["compressed_text"], question, choices)
    rg = call_mistral(client, prompt)
    pred = normalize_pred(rg["text"])
    total_ms = ps["preselect_ms"] + cc["compress_ms"] + rg["latency_ms"]
    rec = {
        "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
        "config": cfg_name, "k": k, "rate": rate, "model": MISTRAL_MODEL,
        "status": rg["status"],
        "gold": gold, "pred": pred, "correct": 1 if pred == gold else 0,
        "domain": domain, "context_chars": len(context),
        "preselect_ms": ps["preselect_ms"],
        "preselect_chars": ps["preselect_chars"],
        "compress_ms": cc["compress_ms"],
        "input_tokens": rg.get("input_tokens"), "output_tokens": rg.get("output_tokens"),
        "mistral_ms": rg["latency_ms"], "total_ms": total_ms,
        "origin_tokens": cc.get("origin_tokens"),
        "compressed_tokens": cc.get("compressed_tokens"),
        "ratio_str": cc.get("ratio_str", ""),
        "raw_text": rg["text"][:200],
    }
    if rg["status"] == "error":
        rec["error"] = rg.get("error", "")[:200]
    print(f"  {cfg_name}: pre={ps['preselect_ms']:.0f}+comp={cc['compress_ms']:.0f}+mis={rg['latency_ms']:.0f}={total_ms:.0f}ms "
          f"ratio={cc.get('ratio_str','?')} tok={rg.get('input_tokens')} pred={pred!r} {'OK' if pred==gold else 'X'}")
    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()
    time.sleep(SLEEP)
    return rec


def aggregate(per_case_cfg: dict, n_cases: int) -> dict:
    """per_case_cfg: {(case_id, cfg_name): rec} -> aggregated summary."""
    import statistics
    by_cfg = {}
    for cfg_name, _, _ in TUNING_CONFIGS:
        by_cfg[cfg_name] = {
            "toks": [], "preselect_chars": [], "comps": [],
            "mistral_ms": [], "totals": [], "accs": [],
            "origin_toks": [], "compressed_toks": [],
        }
    for (_, cfg_name), r in per_case_cfg.items():
        if r.get("status") != "ok":
            continue
        d = by_cfg[cfg_name]
        d["toks"].append(r.get("input_tokens", 0))
        d["preselect_chars"].append(r.get("preselect_chars", 0))
        d["comps"].append(r.get("compress_ms", 0))
        d["mistral_ms"].append(r.get("mistral_ms", 0))
        d["totals"].append(r.get("total_ms", 0))
        d["accs"].append(r.get("correct", 0))
        if r.get("origin_tokens"):
            d["origin_toks"].append(r["origin_tokens"])
            d["compressed_toks"].append(r["compressed_tokens"])

    summary = {"by_config": {}, "comparison": []}
    for cfg_name, k, rate in TUNING_CONFIGS:
        d = by_cfg[cfg_name]
        n = len(d["accs"])

        def m(xs): return round(statistics.mean(xs), 1) if xs else None

        summary["by_config"][cfg_name] = {
            "k": k, "rate": rate, "n_cases": n,
            "avg_input_tokens": m(d["toks"]),
            "avg_preselect_chars": m(d["preselect_chars"]),
            "avg_compress_ms": m(d["comps"]),
            "avg_mistral_ms": m(d["mistral_ms"]),
            "avg_total_ms": m(d["totals"]),
            "accuracy": m(d["accs"]),
            "accuracy_count": sum(d["accs"]),
            "avg_origin_tokens": m(d["origin_toks"]),
            "avg_compressed_tokens": m(d["compressed_toks"]),
        }

    # Add token reduction vs A_baseline (from combo-n15.jsonl n=11 summary)
    base_avg_tok = 16770  # from n=11 summary
    base_avg_ms = 1502
    summary["comparison"] = []
    for cfg_name, k, rate in TUNING_CONFIGS:
        c = summary["by_config"][cfg_name]
        c_tok = c["avg_input_tokens"]
        tok_red = round((1 - (c_tok / base_avg_tok)) * 100, 1) if c_tok else None
        lat_ratio = round(c["avg_total_ms"] / base_avg_ms, 2) if c["avg_total_ms"] else None
        summary["comparison"].append({
            "config": cfg_name, "k": k, "rate": rate,
            "accuracy_count": f"{c['accuracy_count']}/{n_cases}",
            "accuracy": c["accuracy"],
            "avg_input_tokens": c_tok,
            "token_reduction_pct": tok_red,
            "avg_total_ms": c["avg_total_ms"],
            "latency_vs_baseline": lat_ratio,
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=11, help="Cases to test (default 11, matching v3)")
    parser.add_argument("--max-context-chars", type=int, default=400000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one config (e.g., 'C1_k50_r07')")
    args = parser.parse_args()

    selected = load_n_cases(args.n, args.max_context_chars)
    if not selected:
        print("ERROR: no cases", file=sys.stderr)
        return 2

    configs_to_run = TUNING_CONFIGS
    if args.only:
        configs_to_run = [c for c in TUNING_CONFIGS if c[0] == args.only]
        if not configs_to_run:
            print(f"ERROR: unknown config {args.only}. Options: {[c[0] for c in TUNING_CONFIGS]}", file=sys.stderr)
            return 2

    print(f"=== Combo Tuning ===")
    print(f"  model: {MISTRAL_MODEL}")
    print(f"  cases: {len(selected)}")
    print(f"  configs to run: {[c[0] for c in configs_to_run]}")
    for r in selected:
        print(f"    L5-{r['_case_idx']:02d}-{r['_id'][:6]}: ctx={len(r.get('context','')):,}c gold={r.get('answer','').strip().upper()}")

    if args.dry_run:
        return 0

    client = get_mistral_client()
    print("\nLoading compressor...")
    pc = get_llmlingua2()
    print("Compressor loaded.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    per_case_cfg = {}
    with args.out.open("w", encoding="utf-8") as fh:
        for row in selected:
            cid = f"L5-{row['_case_idx']:02d}-{row['_id'][:6]}"
            for cfg_name, k, rate in configs_to_run:
                rec = run_one_config(cid, row, client, pc, cfg_name, k, rate, fh)
                per_case_cfg[(cid, cfg_name)] = rec

    summary = aggregate(per_case_cfg, len(selected))
    summary["meta"] = {
        "n_cases": len(selected),
        "configs": [{"name": c[0], "k": c[1], "rate": c[2]} for c in TUNING_CONFIGS],
        "baseline_token_count": 16770,
        "baseline_latency_ms": 1502,
    }
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Print summary
    print("\n=== Combo Tuning Summary ===")
    print(f"{'config':12s} {'k':>3} {'rate':>4} {'acc':>9} {'in_tok':>7} {'tok_red%':>9} "
          f"{'tot_ms':>7} {'lat_x':>5}")
    print("-" * 70)
    for c in summary["comparison"]:
        acc = c["accuracy"]
        acc_s = f"{acc*100:.0f}%" if acc is not None else "?"
        acc_c = c["accuracy_count"]
        print(f"{c['config']:12s} {c['k']:>3} {c['rate']:>4.1f} {acc_c:>9} {acc_s:>5} "
              f"{c['avg_input_tokens'] or 0:>7.0f} {(c['token_reduction_pct'] or 0):>+9.1f} "
              f"{c['avg_total_ms'] or 0:>7.0f} {c['latency_vs_baseline'] or 0:>5.2f}")

    print(f"\nWrote: {args.out}")
    print(f"       {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
