#!/usr/bin/env python3
"""
Phase 1 v2 - LLMLingua-2 eval on 5 cases, 4 configs.

Uses CORRECT API: PromptCompressor(use_llmlingua2=True), no manual chunking,
force_tokens from tune_knobs best config (rate=0.5, ft_basic).

Configs:
  A_baseline_gemini  — no compression, Gemini only
  B_pip_llmlingua2   — LLMLingua-2 pip-direct (rate=0.5, ft_basic)
  C_lc_llmlingua2    — same as B but via LangChain wrapper
  D_longllmlingua    — LongLLMLingua question-aware (rank_method=longllmlingua)

Metrics (per case): input_tokens, compressor_ms, gemini_ms, total_ms, accuracy.
Delta vs baseline: %token_saved, %time_saved, accuracy_delta_pp.

Output:
  results/phase-01-llmlingua-corrected-5cases.jsonl
  results/phase-01-llmlingua-corrected-summary.json

Usage:
    conda activate vsf
    python scripts/phase-01/eval_llmlingua_v2.py [--n 5] [--max-context-chars 200000]
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from smoke_both_paths import (
    compress_pip_direct,
    compress_langchain,
    COMPRESSOR_MODEL,
)

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-llmlingua-corrected-5cases.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-llmlingua-corrected-summary.json"
GEMINI_MODEL = "gemini-3.5-flash-lite"
SLEEP = 0.5
TUNED_RATE = 0.5
TUNED_FORCE_TOKENS = ["!", ".", "?", "\n"]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_gemini_model():
    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY chua duoc set trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", GEMINI_MODEL))


def call_gemini(model, prompt, max_tokens=256):
    t0 = time.perf_counter()
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def build_prompt(context, question, choices):
    choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
    return (
        "Doc doan van ban sau va tra loi cau hoi. CHI tra ve mot chu cai (A, B, C hoac D) "
        "dung nhat. Khong giai thich.\n\n"
        f"CAU HOI: {question}\n\n"
        f"LUA CHON:\n{choices_str}\n\n"
        f"DOAN VAN BAN:\n{context}\n\n"
        "DAP AN (1 chu cai):"
    )


def normalize_pred(text):
    s = text.strip().upper()
    for ch in s:
        if ch in "ABCD":
            return ch
    return ""


def run_one_case(case_id, row, model, out_fh):
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    domain = row.get("domain", "?")
    results = {}

    # --- A: baseline ---
    prompt_a = build_prompt(context, question, choices)
    ra = call_gemini(model, prompt_a)
    pred_a = normalize_pred(ra["text"])
    toks_a = ra["input_tokens"] or (len(prompt_a) // 4)
    results["A_baseline_gemini"] = {
        "input_tokens": toks_a,
        "compressor_ms": 0.0,
        "gemini_ms": ra["latency_ms"],
        "total_ms": ra["latency_ms"],
        "accuracy": 1 if pred_a == gold else 0,
        "pred": pred_a,
        "raw_text": ra["text"][:300],
        "compression_ratio": 1.0,
        "origin_tokens": toks_a,
        "compressed_tokens": toks_a,
    }
    print(f"  A_baseline: tokens={toks_a} lat={ra['latency_ms']:.0f}ms pred={pred_a!r} {'OK' if pred_a==gold else 'X'}")
    time.sleep(SLEEP)

    # --- B: pip-direct LLMLingua-2 ---
    try:
        comp_b, cmp_ms_b, meta_b = compress_pip_direct(
            context, question, rate=TUNED_RATE, force_tokens=TUNED_FORCE_TOKENS
        )
        prompt_b = build_prompt(comp_b, question, choices)
        rb = call_gemini(model, prompt_b)
        pred_b = normalize_pred(rb["text"])
        toks_b = rb["input_tokens"] or (len(prompt_b) // 4)
        ot_b = meta_b.get("origin_tokens", toks_b)
        ct_b = meta_b.get("compressed_tokens", toks_b)
        results["B_pip_llmlingua2"] = {
            "input_tokens": toks_b,
            "compressor_ms": cmp_ms_b,
            "gemini_ms": rb["latency_ms"],
            "total_ms": cmp_ms_b + rb["latency_ms"],
            "accuracy": 1 if pred_b == gold else 0,
            "pred": pred_b,
            "raw_text": rb["text"][:300],
            "compression_ratio": round(ct_b / max(ot_b, 1), 4),
            "origin_tokens": ot_b,
            "compressed_tokens": ct_b,
        }
        print(f"  B_pip_ll2: ratio={meta_b.get('ratio_str','?')} tokens={toks_b} "
              f"comp={cmp_ms_b:.0f}+gem={rb['latency_ms']:.0f}ms pred={pred_b!r} {'OK' if pred_b==gold else 'X'}")
    except Exception as exc:
        print(f"  B_pip_ll2 FAILED: {exc}")
        results["B_pip_llmlingua2"] = {"error": str(exc)}
    time.sleep(SLEEP)

    # --- C: LangChain wrapper ---
    try:
        comp_c, cmp_ms_c, meta_c = compress_langchain(
            context, question, rate=TUNED_RATE, force_tokens=TUNED_FORCE_TOKENS
        )
        prompt_c = build_prompt(comp_c, question, choices)
        rc = call_gemini(model, prompt_c)
        pred_c = normalize_pred(rc["text"])
        toks_c = rc["input_tokens"] or (len(prompt_c) // 4)
        ot_c = meta_c.get("origin_tokens", toks_c)
        ct_c = meta_c.get("compressed_tokens", toks_c)
        results["C_lc_llmlingua2"] = {
            "input_tokens": toks_c,
            "compressor_ms": cmp_ms_c,
            "gemini_ms": rc["latency_ms"],
            "total_ms": cmp_ms_c + rc["latency_ms"],
            "accuracy": 1 if pred_c == gold else 0,
            "pred": pred_c,
            "raw_text": rc["text"][:300],
            "compression_ratio": round(ct_c / max(ot_c, 1), 4),
            "origin_tokens": ot_c,
            "compressed_tokens": ct_c,
        }
        print(f"  C_lc_ll2:  ratio={meta_c.get('ratio_str','?')} tokens={toks_c} "
              f"comp={cmp_ms_c:.0f}+gem={rc['latency_ms']:.0f}ms pred={pred_c!r} {'OK' if pred_c==gold else 'X'}")
    except Exception as exc:
        print(f"  C_lc_ll2 FAILED: {exc}")
        results["C_lc_llmlingua2"] = {"error": str(exc)}
    time.sleep(SLEEP)

    # --- D: LongLLMLingua (question-aware) ---
    try:
        from llmlingua import PromptCompressor
        ll = PromptCompressor(model_name=COMPRESSOR_MODEL, use_llmlingua2=True, device_map="cpu")
        t0d = time.perf_counter()
        rd = ll.compress_prompt(
            context,
            question=question,
            rate=0.55,
            rank_method="longllmlingua",
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
        )
        cmp_ms_d = (time.perf_counter() - t0d) * 1000.0
        comp_d = rd["compressed_prompt"]
        prompt_d = build_prompt(comp_d, question, choices)
        rg = call_gemini(model, prompt_d)
        pred_d = normalize_pred(rg["text"])
        toks_d = rg["input_tokens"] or (len(prompt_d) // 4)
        ot_d = rd.get("origin_tokens", toks_d)
        ct_d = rd.get("compressed_tokens", toks_d)
        results["D_longllmlingua"] = {
            "input_tokens": toks_d,
            "compressor_ms": cmp_ms_d,
            "gemini_ms": rg["latency_ms"],
            "total_ms": cmp_ms_d + rg["latency_ms"],
            "accuracy": 1 if pred_d == gold else 0,
            "pred": pred_d,
            "raw_text": rg["text"][:300],
            "compression_ratio": round(ct_d / max(ot_d, 1), 4),
            "origin_tokens": ot_d,
            "compressed_tokens": ct_d,
            "ratio_str": rd.get("ratio"),
        }
        print(f"  D_longLL:  ratio={rd.get('ratio','?')} tokens={toks_d} "
              f"comp={cmp_ms_d:.0f}+gem={rg['latency_ms']:.0f}ms pred={pred_d!r} {'OK' if pred_d==gold else 'X'}")
    except Exception as exc:
        print(f"  D_longLL FAILED: {exc}")
        results["D_longllmlingua"] = {"error": str(exc)}
    time.sleep(SLEEP)

    # Write records
    for cfg, m in results.items():
        if "error" in m and "input_tokens" not in m:
            rec = {"ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                   "config": cfg, "status": "error", "error": m["error"],
                   "gold": gold, "domain": domain, "context_chars": len(context)}
        else:
            rec = {"ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                   "config": cfg, "status": "ok", "model": GEMINI_MODEL,
                   "gold": gold, "domain": domain,
                   "context_chars": len(context),
                   "input_tokens": m["input_tokens"],
                   "compressor_ms": m["compressor_ms"],
                   "gemini_ms": m["gemini_ms"],
                   "total_ms": m["total_ms"],
                   "accuracy": m["accuracy"],
                   "pred": m["pred"],
                   "compression_ratio": m.get("compression_ratio"),
                   "origin_tokens": m.get("origin_tokens"),
                   "compressed_tokens": m.get("compressed_tokens")}
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()
    return results


def aggregate(per_case):
    summary = {"by_config": {}, "delta_vs_baseline": {}, "per_case": {}}
    CONFIGS = ("A_baseline_gemini", "B_pip_llmlingua2", "C_lc_llmlingua2", "D_longllmlingua")
    for cfg in CONFIGS:
        toks, comps, gems, totals, accs = [], [], [], [], []
        or_toks, co_toks = [], []
        n = 0
        for _, rdict in per_case:
            r = rdict.get(cfg, {})
            if "error" in r and "input_tokens" not in r:
                continue
            toks.append(r["input_tokens"])
            comps.append(r["compressor_ms"])
            gems.append(r["gemini_ms"])
            totals.append(r["total_ms"])
            accs.append(r["accuracy"])
            if r.get("origin_tokens") and r.get("compressed_tokens"):
                or_toks.append(r["origin_tokens"])
                co_toks.append(r["compressed_tokens"])
            n += 1
        def stat(xs):
            return round(statistics.mean(xs), 1) if xs else None
        summary["by_config"][cfg] = {
            "n_cases": n,
            "avg_input_tokens": stat(toks),
            "avg_compressor_ms": stat(comps),
            "avg_gemini_ms": stat(gems),
            "avg_total_ms": stat(totals),
            "accuracy": stat(accs),
            "avg_origin_tokens": stat(or_toks),
            "avg_compressed_tokens": stat(co_toks),
        }
    b = summary["by_config"]["A_baseline_gemini"]
    for cfg in ("B_pip_llmlingua2", "C_lc_llmlingua2", "D_longllmlingua"):
        c = summary["by_config"].get(cfg, {})
        tok_saved = None
        if b.get("avg_input_tokens") and c.get("avg_input_tokens"):
            tok_saved = round((b["avg_input_tokens"] - c["avg_input_tokens"]) / b["avg_input_tokens"] * 100, 1)
        time_saved = None
        if b.get("avg_total_ms") and c.get("avg_total_ms"):
            time_saved = round((b["avg_total_ms"] - c["avg_total_ms"]) / b["avg_total_ms"] * 100, 1)
        acc_delta = None
        if c.get("accuracy") is not None and b.get("accuracy") is not None:
            acc_delta = round((c["accuracy"] - b["accuracy"]) * 100, 1)
        summary["delta_vs_baseline"][cfg] = {
            "token_saved_pct": tok_saved,
            "time_saved_pct": time_saved,
            "accuracy_delta_pp": acc_delta,
        }
    for case_id, rdict in per_case:
        summary["per_case"][case_id] = {
            cfg: {"accuracy": r.get("accuracy"), "pred": r.get("pred"),
                  "input_tokens": r.get("input_tokens"), "total_ms": r.get("total_ms"),
                  "origin_tokens": r.get("origin_tokens"),
                  "compressed_tokens": r.get("compressed_tokens")}
            for cfg, r in rdict.items()
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=200000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"ERROR: {args.test_file} not found.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.test_file.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    selected, skipped = [], []
    for r in rows:
        if len(selected) >= args.n:
            break
        if len(r.get("context", "")) > args.max_context_chars:
            skipped.append(r["_id"])
            continue
        selected.append(r)

    print("=== Eval v2: 4 configs on 5 cases ===")
    print(f"  tuned: rate={TUNED_RATE}, ft={TUNED_FORCE_TOKENS}")
    print(f"  cases: {len(selected)} selected, {len(skipped)} skipped")

    model = get_gemini_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    per_case = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"L5-{i:02d}-{row.get('_id', '?')[:6]}"
            print(f"\n[{case_id}] domain={row.get('domain','?')} gold={row.get('answer','')}")
            m = run_one_case(case_id, row, model, fh)
            per_case.append((case_id, m))

    summary = aggregate(per_case)
    summary["meta"] = {
        "model": GEMINI_MODEL,
        "tuned_rate": TUNED_RATE,
        "tuned_force_tokens": TUNED_FORCE_TOKENS,
        "n_selected": len(selected),
        "n_skipped_large": len(skipped),
        "skipped_ids": skipped,
        "ts": now_iso(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"{'Config':30s} {'n':>2} {'tok_in':>8} {'tot_ms':>9} {'acc':>5} {'tok_save%':>10} {'time_save%':>11}")
    print("-" * 80)
    b = summary["by_config"]["A_baseline_gemini"]
    print(f"{'A_baseline_gemini':30s} {b['n_cases']:>2} {b['avg_input_tokens'] or 0:>8.0f} "
          f"{(b['avg_total_ms'] or 0):>9.0f} {b['accuracy'] or 0:>5.1f} {'(baseline)':>10}")
    for cfg in ("B_pip_llmlingua2", "C_lc_llmlingua2", "D_longllmlingua"):
        c = summary["by_config"].get(cfg, {})
        d = summary["delta_vs_baseline"].get(cfg, {})
        n = c.get("n_cases", 0)
        tok = c.get("avg_input_tokens")
        tot = c.get("avg_total_ms")
        acc = c.get("accuracy")
        ts = d.get("token_saved_pct")
        tms = d.get("time_saved_pct")
        print(f"{cfg:30s} {n:>2} {(tok or 0):>8.0f} {(tot or 0):>9.0f} {(acc or 0):>5.1f} "
              f"{(ts or 0):>+10.1f} {(tms or 0):>+11.1f}")
    print(f"\nDone. {args.out}")
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
