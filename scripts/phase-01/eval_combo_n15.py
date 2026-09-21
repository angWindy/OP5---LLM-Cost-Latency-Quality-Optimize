#!/usr/bin/env python3
"""
Phase 1 v3 — Scale combo C (Preselect + Compress) to n=15 cases.

Goal: get statistical signal on correctness (Mistral 8B baseline = ~20% on hard
ZeroSCROLLS cases; LongBench-v2 even lower at ~10-25%). n=15 gives std-error ~10%.

Dataset switch (2026-09-21): zai-org/LongBench-v2 → tau/zero_scrolls.

Cases: 5 from zero_scrolls_test5.jsonl (smallest) + 10 from zero_scrolls_dev95.jsonl
(context < 400k chars to stay within Mistral 32k context window).

Configs (same as combo 3-case):
  A: Mistral full context  (baseline)
  B: Mistral + LLMLingua-2 compress only (rate=0.5)
  C: Mistral + preselect (BM25 top-20) + LLMLingua-2 compress

Output:
  results/phase-01-combo-n15.jsonl
  results/phase-01-combo-n15-summary.json

Usage:
  conda activate vsf
  python scripts/phase-01/eval_combo_n15.py [--n 15] [--max-context-chars 400000]
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

# Reuse helper functions from combo_3
from eval_mistral_combo_3 import (
    call_mistral,
    build_prompt,
    normalize_pred,
    compress_llmlingua2,
    preselect_bm25,
    get_llmlingua2,
    get_mistral_client,
    FORCE_TOKENS,
    PRESELECT_K,
    COMPRESS_RATE,
    MISTRAL_MODEL,
    SLEEP,
    now_iso,
)

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "dev_first95.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-combo-n15.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-combo-n15-summary.json"


def load_n_cases(n: int, max_ctx: int) -> list[dict]:
    """Pick n cases: first (5 from test) + (n-5) from dev, all with context < max_ctx.

    Skips malformed JSONL lines gracefully (dev_first95.jsonl has 3 broken rows from upstream).
    """
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
                    continue  # skip broken
        return out

    rows.extend(_safe_read(DEFAULT_TEST))
    rows.extend(_safe_read(DEFAULT_DEV))

    # Dedupe by _id
    seen = set()
    unique = []
    for r in rows:
        rid = r.get("_id")
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(r)
    rows = unique

    # Filter context size + sort smallest first
    rows.sort(key=lambda r: len(r.get("context", "")))
    selected = [r for r in rows if len(r.get("context", "")) <= max_ctx][:n]

    # Assign stable case IDs (L5-NN-xxxxxx)
    for i, r in enumerate(selected):
        r["_case_idx"] = i

    return selected


def run_one_case(case_id: str, row: dict, client, pc, out_fh) -> list[dict]:
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    domain = row.get("domain", "?")
    out: list[dict] = []

    # ---- A: baseline ----
    print(f"\n[{case_id}] A_baseline (full {len(context):,}c)")
    prompt_a = build_prompt(context, question, choices)
    ra = call_mistral(client, prompt_a)
    pred_a = normalize_pred(ra["text"])
    out.append({
        "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
        "config": "A_baseline_mistral", "model": MISTRAL_MODEL, "status": ra["status"],
        "gold": gold, "pred": pred_a, "correct": 1 if pred_a == gold else 0,
        "domain": domain, "context_chars": len(context),
        "preselect_ms": 0, "compress_ms": 0,
        "input_tokens": ra.get("input_tokens"), "output_tokens": ra.get("output_tokens"),
        "mistral_ms": ra["latency_ms"], "total_ms": ra["latency_ms"],
        "raw_text": ra["text"][:200],
    })
    print(f"  A: tokens={ra.get('input_tokens')} lat={ra['latency_ms']:.0f}ms pred={pred_a!r} {'OK' if pred_a==gold else 'X'}")
    if ra["status"] == "error":
        print(f"  ERR: {ra.get('error','')[:100]}")
    time.sleep(SLEEP)

    # ---- B: compress only ----
    print(f"[{case_id}] B_compress_only (LLMLingua-2 rate={COMPRESS_RATE})")
    cb = compress_llmlingua2(pc, context, rate=COMPRESS_RATE)
    if cb["status"] == "ok":
        prompt_b = build_prompt(cb["compressed_text"], question, choices)
        rb = call_mistral(client, prompt_b)
        pred_b = normalize_pred(rb["text"])
        out.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "B_compress_only", "model": MISTRAL_MODEL, "status": rb["status"],
            "gold": gold, "pred": pred_b, "correct": 1 if pred_b == gold else 0,
            "domain": domain, "context_chars": len(context),
            "preselect_ms": 0, "compress_ms": cb["compress_ms"],
            "input_tokens": rb.get("input_tokens"), "output_tokens": rb.get("output_tokens"),
            "mistral_ms": rb["latency_ms"],
            "total_ms": cb["compress_ms"] + rb["latency_ms"],
            "origin_tokens": cb.get("origin_tokens"),
            "compressed_tokens": cb.get("compressed_tokens"),
            "ratio_str": cb.get("ratio_str", ""),
            "raw_text": rb["text"][:200],
        })
        print(f"  B: comp={cb['compress_ms']:.0f}ms ratio={cb.get('ratio_str','?')} mistral={rb['latency_ms']:.0f}ms total={cb['compress_ms']+rb['latency_ms']:.0f}ms pred={pred_b!r}")
    else:
        out.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "B_compress_only", "status": "compress_error", "error": cb.get("error", "")[:200],
        })
        print(f"  B: COMPRESS ERR {cb.get('error','')[:100]}")
    time.sleep(SLEEP)

    # ---- C: preselect + compress ----
    print(f"[{case_id}] C_preselect+compress (BM25 k={PRESELECT_K} -> LLMLingua-2 rate={COMPRESS_RATE})")
    ps = preselect_bm25(context, question, k=PRESELECT_K)
    if ps["status"] == "ok":
        cc = compress_llmlingua2(pc, ps["selected_text"], rate=COMPRESS_RATE)
        if cc["status"] == "ok":
            prompt_c = build_prompt(cc["compressed_text"], question, choices)
            rc = call_mistral(client, prompt_c)
            pred_c = normalize_pred(rc["text"])
            total_c = ps["preselect_ms"] + cc["compress_ms"] + rc["latency_ms"]
            out.append({
                "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                "config": "C_preselect_compress", "model": MISTRAL_MODEL, "status": rc["status"],
                "gold": gold, "pred": pred_c, "correct": 1 if pred_c == gold else 0,
                "domain": domain, "context_chars": len(context),
                "preselect_ms": ps["preselect_ms"],
                "preselect_n_sentences": ps["n_sentences"],
                "preselect_n_kept": ps["n_kept"],
                "preselect_chars": ps["preselect_chars"],
                "compress_ms": cc["compress_ms"],
                "input_tokens": rc.get("input_tokens"), "output_tokens": rc.get("output_tokens"),
                "mistral_ms": rc["latency_ms"], "total_ms": total_c,
                "origin_tokens": cc.get("origin_tokens"),
                "compressed_tokens": cc.get("compressed_tokens"),
                "ratio_str": cc.get("ratio_str", ""),
                "raw_text": rc["text"][:200],
            })
            print(f"  C: presel={ps['preselect_ms']:.0f}ms comp={cc['compress_ms']:.0f}ms ratio={cc.get('ratio_str','?')} mistral={rc['latency_ms']:.0f}ms TOTAL={total_c:.0f}ms pred={pred_c!r}")
        else:
            out.append({
                "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                "config": "C_preselect_compress", "status": "compress_error",
                "error": cc.get("error", "")[:200],
            })
            print(f"  C: COMPRESS ERR {cc.get('error','')[:100]}")
    else:
        out.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "C_preselect_compress", "status": "preselect_error",
            "error": ps.get("error", "")[:200],
        })
        print(f"  C: PRESELECT ERR {ps.get('error','')[:100]}")

    for r in out:
        out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_fh.flush()
    return out


def aggregate(per_case: list[tuple[str, list[dict]]]) -> dict:
    """Compute per-config stats with mean/stdev for n cases."""
    import statistics
    summary = {
        "ts": now_iso(),
        "model": MISTRAL_MODEL,
        "by_config": {},
        "delta_vs_baseline": {},
        "per_case": [],
    }
    CONFIGS = ("A_baseline_mistral", "B_compress_only", "C_preselect_compress")
    by_cfg = {c: {"toks": [], "comps": [], "mistral_ms": [], "totals": [], "accs": [],
                   "origin_toks": [], "compressed_toks": []} for c in CONFIGS}

    for cid, recs in per_case:
        per_case_entry = {"case_id": cid}
        for r in recs:
            cfg = r.get("config")
            if r.get("status") != "ok":
                per_case_entry[cfg] = {"status": r.get("status"), "error": r.get("error", "")[:80]}
                continue
            toks = r.get("input_tokens") or 0
            comp = r.get("compress_ms", 0)
            mis = r.get("mistral_ms", 0)
            tot = r.get("total_ms", 0)
            acc = r.get("correct", 0)
            by_cfg[cfg]["toks"].append(toks)
            by_cfg[cfg]["comps"].append(comp)
            by_cfg[cfg]["mistral_ms"].append(mis)
            by_cfg[cfg]["totals"].append(tot)
            by_cfg[cfg]["accs"].append(acc)
            if r.get("origin_tokens") and r.get("compressed_tokens"):
                by_cfg[cfg]["origin_toks"].append(r["origin_tokens"])
                by_cfg[cfg]["compressed_toks"].append(r["compressed_tokens"])
            per_case_entry[cfg] = {
                "input_tokens": toks, "compress_ms": comp,
                "mistral_ms": round(mis, 1), "total_ms": round(tot, 1),
                "correct": acc, "pred": r.get("pred"),
                "origin_tokens": r.get("origin_tokens"),
                "compressed_tokens": r.get("compressed_tokens"),
            }
        summary["per_case"].append(per_case_entry)

    for cfg in CONFIGS:
        d = by_cfg[cfg]
        n = len(d["accs"])
        def m(xs): return round(statistics.mean(xs), 1) if xs else None
        def s(xs): return round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0
        summary["by_config"][cfg] = {
            "n_cases": n,
            "avg_input_tokens": m(d["toks"]),
            "stdev_input_tokens": s(d["toks"]),
            "avg_compress_ms": m(d["comps"]),
            "avg_mistral_ms": m(d["mistral_ms"]),
            "avg_total_ms": m(d["totals"]),
            "stdev_total_ms": s(d["totals"]),
            "accuracy": m(d["accs"]),
            "accuracy_count": sum(d["accs"]),
            "avg_origin_tokens": m(d["origin_toks"]),
            "avg_compressed_tokens": m(d["compressed_toks"]),
        }

    # Delta vs baseline
    b = summary["by_config"]["A_baseline_mistral"]
    for cfg in ("B_compress_only", "C_preselect_compress"):
        c = summary["by_config"][cfg]
        tok_saved = None
        if b["avg_input_tokens"] and c["avg_input_tokens"]:
            tok_saved = round((b["avg_input_tokens"] - c["avg_input_tokens"]) / b["avg_input_tokens"] * 100, 1)
        time_saved = None
        if b["avg_total_ms"] and c["avg_total_ms"]:
            time_saved = round((b["avg_total_ms"] - c["avg_total_ms"]) / b["avg_total_ms"] * 100, 1)
        acc_delta = None
        if c["accuracy"] is not None and b["accuracy"] is not None:
            acc_delta = round((c["accuracy"] - b["accuracy"]) * 100, 1)
        speedup = None
        if b["avg_total_ms"] and c["avg_total_ms"]:
            speedup = round(b["avg_total_ms"] / c["avg_total_ms"], 2)
        summary["delta_vs_baseline"][cfg] = {
            "token_saved_pct": tok_saved,
            "time_saved_pct": time_saved,
            "accuracy_delta_pp": acc_delta,
            "speedup": speedup,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="Total cases (default 15)")
    parser.add_argument("--max-context-chars", type=int, default=400000,
                        help="Skip cases with context > this (Mistral 32k ctx limit)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dry-run", action="store_true", help="Just print plan, don't run")
    args = parser.parse_args()

    selected = load_n_cases(args.n, args.max_context_chars)
    if not selected:
        print("ERROR: no cases selected", file=sys.stderr)
        return 2

    print(f"=== Combo n={args.n} -- 3 configs ===")
    print(f"  model: {MISTRAL_MODEL}")
    print(f"  preselect_k={PRESELECT_K}, compress_rate={COMPRESS_RATE}")
    print(f"  max_context_chars={args.max_context_chars}")
    print(f"  cases: {len(selected)} selected")
    for r in selected:
        print(f"    L5-{r['_case_idx']:02d}-{r['_id'][:6]}: ctx={len(r.get('context','')):,}c gold={r.get('answer','').strip().upper()} domain={r.get('domain','?')}")

    if args.dry_run:
        return 0

    client = get_mistral_client()
    print("\nLoading compressor...")
    pc = get_llmlingua2()
    print("Compressor loaded.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    per_case = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"L5-{i:02d}-{row['_id'][:6]}"
            res = run_one_case(case_id, row, client, pc, fh)
            per_case.append((case_id, res))
            if i < len(selected) - 1:
                time.sleep(SLEEP)

    summary = aggregate(per_case)
    summary["meta"] = {
        "n_cases": len(selected),
        "max_context_chars": args.max_context_chars,
        "preselect_k": PRESELECT_K,
        "compress_rate": COMPRESS_RATE,
        "force_tokens": FORCE_TOKENS,
    }

    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Print table
    print("\n=== Combo n={} Summary ===".format(len(selected)))
    print(f"{'config':25s} {'n':>2} {'tok':>6} {'tot_ms':>8} {'acc':>5} "
          f"{'tok_save%':>10} {'time_save%':>11} {'speedup':>8}")
    print("-" * 90)
    b = summary["by_config"]["A_baseline_mistral"]
    print(f"{'A_baseline_mistral':25s} {b['n_cases']:>2} {b['avg_input_tokens'] or 0:>6.0f} "
          f"{b['avg_total_ms'] or 0:>8.0f} {b['accuracy'] or 0:>5.1f} {'(baseline)':>10}")
    for cfg in ("B_compress_only", "C_preselect_compress"):
        c = summary["by_config"][cfg]
        d = summary["delta_vs_baseline"][cfg]
        print(f"{cfg:25s} {c['n_cases']:>2} {c['avg_input_tokens'] or 0:>6.0f} "
              f"{c['avg_total_ms'] or 0:>8.0f} {c['accuracy'] or 0:>5.1f} "
              f"{(d['token_saved_pct'] or 0):>+10.1f} {(d['time_saved_pct'] or 0):>+11.1f} "
              f"{(d['speedup'] or 0):>8.2f}")

    print(f"\nWrote: {args.out}")
    print(f"       {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
