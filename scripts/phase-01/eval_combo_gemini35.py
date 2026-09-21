#!/usr/bin/env python3
"""
Phase 1 v5 — Gemini-3.5-flash accuracy experiment.

Goal: 80% accuracy on n=11 requires 9/11 correct.
Tests both:
  A_gemini     = baseline (full context, no compression)
  C0_gemini    = preselect (k=20) + compress (rate=0.5)

Both on the same 11 cases for apples-to-apples comparison.

Usage:
  conda activate vsf
  python scripts/phase-01/eval_combo_gemini35.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from eval_mistral_combo_3 import (
    normalize_pred,
    compress_llmlingua2, preselect_bm25,
    get_llmlingua2, now_iso,
)

import google.generativeai as genai

GEMINI_MODEL = os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash")
SLEEP = 2          # Gemini has higher rate limit than Mistral free tier
PRESELECT_K = 20
COMPRESS_RATE = 0.5
N_CASES = 11

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "dev_first95.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase-01-gemini-lite-combo.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "phase-01-gemini-lite-combo-summary.json"


def load_cases(n: int, max_ctx: int) -> list[dict]:
    """Same case selection as combo-tuned (sorted by context len, ≤max_ctx)."""
    rows = []
    for p in [DEFAULT_TEST, DEFAULT_DEV]:
        if not p.exists():
            continue
        with p.open() as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue

    seen = set()
    unique = []
    for r in rows:
        rid = r.get("_id")
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(r)
    unique.sort(key=lambda r: len(r.get("context", "")))
    selected = [r for r in unique if len(r.get("context", "")) <= max_ctx][:n]
    for i, r in enumerate(selected):
        r["_case_idx"] = i
    return selected


def call_gemini(model, prompt: str, max_tokens: int = 512) -> dict:
    """Single Gemini API call."""
    t0 = time.perf_counter()
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        return {
            "status": "ok",
            "text": text,
            "latency_ms": elapsed_ms,
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "text": "",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": str(exc)[:200],
        }


def build_prompt(context: str, question: str, choices: dict) -> str:
    options = "\n".join(f"  {k}. {v}" for k, v in choices.items())
    return (
        f"DUA VAO NGU CANH SAU, TRA LOI CAU HOI.\n\n"
        f"NGU CANH:\n{context}\n\n"
        f"CAU HOI: {question}\n\n"
        f"LUA CHON:\n{options}\n\n"
        f"Chi tra loi mot chu cai A, B, C, D. Khong giai thich."
    )


def run_A(row: dict, model, cid: str) -> dict:
    """Config A: Gemini baseline (full context, no compression)."""
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k_: row.get(f"choice_{k_}", "") for k_ in "ABCD"}
    domain = row.get("domain", "?")

    prompt = build_prompt(context, question, choices)
    rg = call_gemini(model, prompt)
    pred = normalize_pred(rg["text"])

    return {
        "ts": now_iso(), "track": "qa_mc", "case_id": cid,
        "config": "A_gemini_baseline", "model": GEMINI_MODEL,
        "status": rg["status"],
        "gold": gold, "pred": pred, "correct": 1 if pred == gold else 0,
        "domain": domain, "context_chars": len(context),
        "input_tokens": rg.get("input_tokens"), "output_tokens": rg.get("output_tokens"),
        "total_ms": rg["latency_ms"],
        "ratio_str": "1.0x",
        "raw_text": rg["text"][:200],
        "error": rg.get("error", "")[:200] if rg["status"] == "error" else None,
    }


def run_C0(row: dict, model, pc, cid: str) -> dict:
    """Config C0: BM25 preselect (k=20) + LLMLingua-2 compress (rate=0.5)."""
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k_: row.get(f"choice_{k_}", "") for k_ in "ABCD"}
    domain = row.get("domain", "?")

    ps = preselect_bm25(context, question, k=PRESELECT_K)
    if ps["status"] != "ok":
        return {
            "ts": now_iso(), "case_id": cid, "config": "C0_gemini",
            "status": "preselect_error", "error": ps.get("error","")[:200],
            "gold": gold, "domain": domain, "context_chars": len(context),
        }

    cc = compress_llmlingua2(pc, ps["selected_text"], rate=COMPRESS_RATE)
    if cc["status"] != "ok":
        return {
            "ts": now_iso(), "case_id": cid, "config": "C0_gemini",
            "status": "compress_error", "error": cc.get("error","")[:200],
            "gold": gold, "domain": domain, "context_chars": len(context),
        }

    prompt = build_prompt(cc["compressed_text"], question, choices)
    rg = call_gemini(model, prompt)
    pred = normalize_pred(rg["text"])
    total_ms = ps["preselect_ms"] + cc["compress_ms"] + rg["latency_ms"]

    rec = {
        "ts": now_iso(), "track": "qa_mc", "case_id": cid,
        "config": "C0_gemini", "model": GEMINI_MODEL,
        "status": rg["status"],
        "gold": gold, "pred": pred, "correct": 1 if pred == gold else 0,
        "domain": domain, "context_chars": len(context),
        "preselect_ms": ps["preselect_ms"], "preselect_chars": ps["preselect_chars"],
        "compress_ms": cc["compress_ms"],
        "input_tokens": rg.get("input_tokens"), "output_tokens": rg.get("output_tokens"),
        "total_ms": total_ms,
        "origin_tokens": cc.get("origin_tokens"),
        "compressed_tokens": cc.get("compressed_tokens"),
        "ratio_str": cc.get("ratio_str", ""),
        "raw_text": rg["text"][:200],
    }
    if rg["status"] == "error":
        rec["error"] = rg.get("error", "")[:200]
    return rec


def summarize(results: list[dict]) -> dict:
    import statistics
    by_cfg = {}
    for r in results:
        cfg = r.get("config", "?")
        if cfg not in by_cfg:
            by_cfg[cfg] = {"accs": [], "toks": [], "tots": [], "n": 0}
        if r.get("status") == "ok":
            by_cfg[cfg]["accs"].append(r.get("correct", 0))
            by_cfg[cfg]["toks"].append(r.get("input_tokens") or 0)
            by_cfg[cfg]["tots"].append(r.get("total_ms") or 0)
            by_cfg[cfg]["n"] += 1

    summary = {"configs": []}
    for cfg_name in ["A_gemini_baseline", "C0_gemini"]:
        if cfg_name not in by_cfg:
            continue
        d = by_cfg[cfg_name]
        n = d["n"]
        acc = sum(d["accs"])
        avg_tok = statistics.mean(d["toks"]) if d["toks"] else 0
        avg_tot = statistics.mean(d["tots"]) if d["tots"] else 0

        # Token reduction vs A
        a_tok = None
        if cfg_name == "C0_gemini" and "A_gemini_baseline" in by_cfg:
            a_toks = by_cfg["A_gemini_baseline"]["toks"]
            a_tok = statistics.mean(a_toks) if a_toks else 16770
        tok_red = round((1 - avg_tok / (a_tok or avg_tok)) * 100, 1) if avg_tok else None

        summary["configs"].append({
            "config": cfg_name,
            "n_cases": n,
            "accuracy": f"{acc}/{n} = {acc/n:.1%}" if n else "N/A",
            "accuracy_count": acc,
            "avg_input_tokens": round(avg_tok, 1),
            "token_reduction_pct": tok_red,
            "avg_total_ms": round(avg_tot, 1),
            "target_80_pct": f"{int(0.8 * n)}/{n}" if n else "N/A",
        })

    # Per-case table
    by_case = {}
    for r in results:
        cid = r.get("case_id", "")
        cfg = r.get("config", "")
        if cid not in by_case:
            by_case[cid] = {}
        # Always record, even if error
        by_case[cid][cfg] = r

    summary["per_case"] = []
    for cid in sorted(by_case):
        row = by_case[cid]
        a = row.get("A_gemini_baseline", {})
        c = row.get("C0_gemini", {})
        summary["per_case"].append({
            "case_id": cid,
            "gold": a.get("gold") or c.get("gold", "?"),
            "A_pred": a.get("pred", "?"),
            "A_correct": a.get("correct", 0),
            "C_pred": c.get("pred", "?"),
            "C_correct": c.get("correct", 0),
            "A_tok": a.get("input_tokens"),
            "C_tok": c.get("input_tokens"),
        })

    summary["comparison"] = {
        "mistral8b_C0": "5/11 = 45.5% (from phase-01-combo-tuned.jsonl)",
        "mistral8b_A": "5/11 = 45.5% (from phase-01-combo-n15.jsonl)",
    }
    return summary


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_CASES)
    parser.add_argument("--max-context-chars", type=int, default=400000)
    parser.add_argument("--configs", type=str, default="A,C",
                        help="Comma-separated: A (baseline), C (preselect+compress)")
    args = parser.parse_args()

    configs_to_run = []
    if "A" in args.configs:
        configs_to_run.append("A")
    if "C" in args.configs:
        configs_to_run.append("C")

    selected = load_cases(args.n, args.max_context_chars)
    if not selected:
        print("ERROR: no cases loaded", file=sys.stderr)
        return 2

    print(f"=== Gemini-3.5-flash Experiment ===")
    print(f"  model: {GEMINI_MODEL}")
    print(f"  configs: {[('A' if c=='A' else 'C0') for c in configs_to_run]}")
    print(f"  cases: {len(selected)}")
    for r in selected:
        print(f"    L5-{r['_case_idx']:02d}-{r['_id'][:6]}: ctx={len(r.get('context','')):,}c "
              f"gold={r.get('answer','').strip().upper()}")

    # Setup Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        return 1
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"\nGemini model loaded: {GEMINI_MODEL}")

    # Load compressor (only needed for C config)
    pc = None
    if "C" in configs_to_run:
        print("Loading LLMLingua-2 compressor...")
        pc = get_llmlingua2()
        print("Compressor loaded.\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    results = []
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in selected:
            context = row.get("context", "")
            question = row.get("question", "")
            gold = row.get("answer", "").strip().upper()
            choices = {k_: row.get(f"choice_{k_}", "") for k_ in "ABCD"}
            cid = f"L5-{row['_case_idx']:02d}-{row['_id'][:6]}"
            domain = row.get("domain", "?")

            for cfg_type in configs_to_run:
                if cfg_type == "A":
                    print(f"[{cid}] A_gemini_baseline (full context, no compression)")
                    rec = run_A(row, model, cid)
                    rec["domain"] = domain
                    if rec["status"] == "ok":
                        print(f"  A_gemini_baseline: gem={rec['total_ms']:.0f}ms "
                              f"tok={rec.get('input_tokens')} pred={rec['pred']!r} "
                              f"{'OK' if rec['correct'] else 'X'}")
                    else:
                        print(f"  A_gemini_baseline: ERROR {rec.get('error','?')}")
                else:
                    print(f"[{cid}] C0_gemini (k={PRESELECT_K}, rate={COMPRESS_RATE})")
                    rec = run_C0(row, model, pc, cid)
                    rec["domain"] = domain
                    if rec.get("status") == "ok":
                        gem_ms = rec.get('total_ms', 0) - rec.get('preselect_ms', 0) - rec.get('compress_ms', 0)
                        print(f"  C0_gemini: pre={rec.get('preselect_ms', 0):.0f}+"
                              f"comp={rec.get('compress_ms', 0):.0f}+"
                              f"gem={gem_ms:.0f}={rec.get('total_ms', 0):.0f}ms "
                              f"ratio={rec.get('ratio_str','?')} tok={rec.get('input_tokens')} "
                              f"pred={rec['pred']!r} {'OK' if rec['correct'] else 'X'}")
                    else:
                        print(f"  C0_gemini: ERROR {rec.get('error','?')}")

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                results.append(rec)
                time.sleep(SLEEP)

    summary = summarize(results)
    summary["meta"] = {
        "model": GEMINI_MODEL,
        "n_cases": len(selected),
        "configs_run": configs_to_run,
        "preselect_k": PRESELECT_K,
        "compress_rate": COMPRESS_RATE,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Print table
    print(f"\n=== Gemini Summary ===")
    print(f"{'config':22s} {'n':>2} {'acc':>9} {'avg_tok':>8} {'tok_red%':>9} {'avg_ms':>7}")
    print("-" * 60)
    for c in summary["configs"]:
        print(f"{c['config']:22s} {c['n_cases']:>2} {c['accuracy']:>9} "
              f"{c['avg_input_tokens']:>8.0f} {(c['token_reduction_pct'] or 0):>+9.1f} "
              f"{c['avg_total_ms']:>7.0f}")

    print(f"\n=== Per-case ===")
    print(f"{'case':18s} {'gold':5s} {'A_pred':7s} {'A_ok':>5s} {'C_pred':7s} {'C_ok':>5s} {'A_tok':>6s} {'C_tok':>6s}")
    print("-" * 70)
    for row in summary["per_case"]:
        a_ok = "Y" if row["A_correct"] else "N"
        c_ok = "Y" if row["C_correct"] else "N"
        print(f"{row['case_id'][:18]:18s} {row['gold']:5s} {str(row['A_pred']):>7s} "
              f"{a_ok:>5s} {str(row['C_pred']):>7s} {c_ok:>5s} "
              f"{str(row['A_tok'] or '?'):>6s} {str(row['C_tok'] or '?'):>6s}")

    print(f"\nWrote: {OUT_PATH}")
    print(f"       {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
