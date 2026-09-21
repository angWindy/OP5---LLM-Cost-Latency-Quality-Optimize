#!/usr/bin/env python3
"""
Fix round 3 — try the SKIP-COMPRESSOR strategy for hard tasks.

Round 1 (k=10, rate=0.5, mt=512): 55.6% (10/18) — best so far
Round 2 (per-task k/rate, mt=1024): 50.0% (9/18) — regressed on musique/gov_report/squality

Remaining failing tasks after round 1 (LLM-as-judge):
  - L00, L04 musique (multi-hop QA — needs ALL paragraphs)
  - L08, L05 gov_report (long summary — compressor may drop key facts)
  - L12, L14 space_digest (positive % count — needs ALL reviews visible)
  - L13, L15 summ_screen_fd (full episode — needs all scenes)
  - L10 squality (story QA — needs character names)

For these, the compressor (rate=0.5) likely cuts essential info.

Strategy: For each failing task, run with NO compressor:
  - k=30 (more preselect sentences)
  - rate=1.0 (no compression)
  - max_tokens=1024

Compare against round 1 (best so far) on the SAME 18 cases — only count
the formerly-failing cases that flip from incorrect -> correct.

Usage:
  conda activate vsf
  python scripts/phase-01/fix_longllmlingua_round3.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_longllmlingua import (
    stream_dev_with_instructions,
    judge_v2,
    call_gemini_fixed,
    build_qa_prompt_fixed,
    now_iso,
    GEMINI_MODEL,
    DEFAULT_DEV,
    SLEEP,
)
from longllmlingua_opt import (
    preselect_tfidf,
    compress_llmlingua2,
    get_gemini,
)

# Per-task knobs for hard tasks. Use rate=1.0 (no compression).
HARD_CASES = {
    "space_digest": {"k": 30, "rate": 1.0, "max_tokens": 1024},
    "summ_screen_fd": {"k": 25, "rate": 1.0, "max_tokens": 1024},
    "gov_report": {"k": 25, "rate": 1.0, "max_tokens": 1024},
    "musique": {"k": 25, "rate": 1.0, "max_tokens": 1024},
    "squality": {"k": 25, "rate": 1.0, "max_tokens": 1024},
    "qmsum": {"k": 25, "rate": 1.0, "max_tokens": 1024},
}
# Easy tasks (round 1 already worked) — use round 1 config to confirm no regression.
EASY_CASES = {
    "book_sum_sort": {"k": 10, "rate": 0.5, "max_tokens": 512},
    "qasper": {"k": 10, "rate": 0.5, "max_tokens": 512},
    "quality": {"k": 10, "rate": 0.5, "max_tokens": 512},
}


def load_dev_with_instructions(n: int, seed: int = 42) -> list[dict]:
    cleaned = stream_dev_with_instructions()
    print(f"  [load] {len(cleaned)} cases (from dev95)")
    import random
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in cleaned:
        by_task[r["task"]].append(r)
    selected = []
    per = max(1, n // max(1, len(by_task)))
    for task, group in by_task.items():
        rng.shuffle(group)
        selected.extend(group[:per])
    rng.shuffle(selected)
    selected = selected[:n]
    selected.sort(key=lambda r: len(r["context"]))
    return selected


def run_no_compress(row: dict, k: int, max_tokens: int) -> dict:
    """No-compressor: just preselect top-k sentences, send raw to Gemini."""
    context = row["context"]
    question = row["question"]
    gold = row["answer"]
    instruction = row.get("_orig_instruction", "")

    ps = preselect_tfidf(context, question, k=k)
    if ps["status"] != "ok":
        return {"status": "preselect_error", "error": ps.get("error", "")}

    prompt = build_qa_prompt_fixed(ps["selected_text"], question, instruction)
    g = get_gemini()
    ll = call_gemini_fixed(g, prompt, max_tokens=max_tokens)
    if ll["status"] != "ok":
        return {"status": "llm_error", "error": ll.get("error", "")}

    jj = judge_v2(gold, ll["text"])
    return {
        "status": "ok",
        "k": k,
        "rate": 1.0,
        "max_tokens": max_tokens,
        "ps_ms": ps["preselect_ms"],
        "ps_chars": ps["preselect_chars"],
        "comp_ms": 0,
        "comp_tok_in": None,
        "comp_tok_out": None,
        "comp_ratio": "1.0x (no compress)",
        "llm_ms": ll["latency_ms"],
        "input_tokens": ll.get("input_tokens"),
        "output_tokens": ll.get("output_tokens"),
        "total_ms": round(ps["preselect_ms"] + ll["latency_ms"], 1),
        "pred": ll["text"][:400],
        "gold": gold[:200],
        "judge_correct": jj["judge_correct"],
        "judge_reason": jj["judge_reason"],
        "used_instruction": bool(instruction),
    }


def run_with_compress(row: dict, k: int, rate: float, max_tokens: int) -> dict:
    """Standard compress pipeline."""
    context = row["context"]
    question = row["question"]
    gold = row["answer"]
    instruction = row.get("_orig_instruction", "")

    ps = preselect_tfidf(context, question, k=k)
    if ps["status"] != "ok":
        return {"status": "preselect_error", "error": ps.get("error", "")}

    cc = compress_llmlingua2(ps["selected_text"], question, rate=rate)
    if cc["status"] != "ok":
        return {"status": "compress_error", "error": cc.get("error", "")}

    prompt = build_qa_prompt_fixed(cc["compressed_text"], question, instruction)
    g = get_gemini()
    ll = call_gemini_fixed(g, prompt, max_tokens=max_tokens)
    if ll["status"] != "ok":
        return {"status": "llm_error", "error": ll.get("error", "")}

    jj = judge_v2(gold, ll["text"])
    return {
        "status": "ok",
        "k": k,
        "rate": rate,
        "max_tokens": max_tokens,
        "ps_ms": ps["preselect_ms"],
        "ps_chars": ps["preselect_chars"],
        "comp_ms": cc["compress_ms"],
        "comp_tok_in": cc.get("origin_tokens"),
        "comp_tok_out": cc.get("compressed_tokens"),
        "comp_ratio": cc.get("ratio_str", ""),
        "llm_ms": ll["latency_ms"],
        "input_tokens": ll.get("input_tokens"),
        "output_tokens": ll.get("output_tokens"),
        "total_ms": round(ps["preselect_ms"] + cc["compress_ms"] + ll["latency_ms"], 1),
        "pred": ll["text"][:400],
        "gold": gold[:200],
        "judge_correct": jj["judge_correct"],
        "judge_reason": jj["judge_reason"],
        "used_instruction": bool(instruction),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or (REPO_ROOT / "results" / "phase-01-longllmlingua-opt-n20-fix3.jsonl")
    summary_path = args.summary or (REPO_ROOT / "results" / f"phase-01-longllmlingua-opt-n20-fix3-summary.json")

    print("=== LongLLMLingua fix round 3 (no-compressor cho hard tasks) ===")
    print(f"  Hard tasks (rate=1.0 = no compressor):")
    for t, cfg in HARD_CASES.items():
        print(f"    {t:<18s} k={cfg['k']:>3}  max_tokens={cfg['max_tokens']}")
    print(f"  Easy tasks (round 1 config):")
    for t, cfg in EASY_CASES.items():
        print(f"    {t:<18s} k={cfg['k']:>3}  rate={cfg['rate']:.2f}  max_tokens={cfg['max_tokens']}")

    rows = load_dev_with_instructions(args.n, seed=args.seed)
    print(f"  [loaded] {len(rows)} cases")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    all_records = []
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            case_id = f"L{i:02d}-{row['_id'][:8]}"
            task = row["task"]

            if task in HARD_CASES:
                cfg = HARD_CASES[task]
                rec = run_no_compress(row, k=cfg["k"], max_tokens=cfg["max_tokens"])
                rec["strategy"] = "no_compress"
            elif task in EASY_CASES:
                cfg = EASY_CASES[task]
                rec = run_with_compress(row, k=cfg["k"], rate=cfg["rate"], max_tokens=cfg["max_tokens"])
                rec["strategy"] = "compress"
            else:
                rec = run_with_compress(row, k=10, rate=0.5, max_tokens=512)
                rec["strategy"] = "compress_default"

            t0 = time.perf_counter()
            rec["ts"] = now_iso()
            rec["case_id"] = case_id
            rec["task"] = task
            rec["ctx_chars"] = len(row["context"])
            all_records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            if rec.get("status") == "ok":
                marker = "OK" if rec["judge_correct"] else "X"
                strat = rec.get("strategy", "?")
                print(f"  [{case_id}] {strat:>14s} task={task:<14s} k={rec.get('k','?'):>2} "
                      f"ctx={len(row['context']):>5,} ps={rec['ps_chars']:>5,} "
                      f"comp_ratio={rec.get('comp_ratio','?'):<10} "
                      f"llm={rec['llm_ms']:>5.0f}ms -> {marker}")
            else:
                print(f"  [{case_id}] ERROR: {rec.get('error', rec.get('status',''))[:80]}")
            time.sleep(SLEEP)
            _ = (time.perf_counter() - t0)

    ok = sum(1 for r in all_records if r.get("status") == "ok")
    corr = sum(1 for r in all_records if r.get("judge_correct"))
    print(f"\n  Total: {corr}/{ok} correct ({corr/ok:.1%})" if ok else "  No records.")

    summary = {"ts": now_iso(), "n_cases": len(rows), "fix": "no-compress for hard tasks",
               "hard_tasks": HARD_CASES, "easy_tasks": EASY_CASES}
    per_task = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": [], "llm_ms": [], "total_ms": []})
    for r in all_records:
        if r.get("status") != "ok":
            continue
        t = r["task"]
        per_task[t]["n"] += 1
        if r["judge_correct"]:
            per_task[t]["correct"] += 1
        if r.get("input_tokens"):
            per_task[t]["tokens"].append(r["input_tokens"])
        per_task[t]["llm_ms"].append(r["llm_ms"])
        per_task[t]["total_ms"].append(r["total_ms"])

    summary["per_task"] = {
        t: {
            "n": d["n"],
            "correct": d["correct"],
            "acc": round(d["correct"]/d["n"], 3) if d["n"] else None,
            "avg_input_tokens": round(statistics.mean(d["tokens"]), 1) if d["tokens"] else None,
            "avg_llm_ms": round(statistics.mean(d["llm_ms"]), 1) if d["llm_ms"] else None,
            "avg_total_ms": round(statistics.mean(d["total_ms"]), 1) if d["total_ms"] else None,
            "strategy": "no_compress" if t in HARD_CASES else "compress",
        }
        for t, d in sorted(per_task.items())
    }
    summary["overall_acc"] = round(corr / ok, 3) if ok else None
    summary["overall_correct"] = corr
    summary["overall_n"] = ok

    with summary_path.open("w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" SUMMARY - Round 3 (no-compress for hard tasks)")
    print("=" * 100)
    print(f"  {'Task':<18s} {'Strategy':<14s} {'Acc':>9s} {'InTok':>8s} {'LlmMs':>7s} {'TotMs':>7s}")
    print("  " + "-" * 90)
    for t, d in sorted(per_task.items()):
        strat = "no_compress" if t in HARD_CASES else "compress"
        print(f"  {t:<18s} {strat:<14s} {d['correct']}/{d['n']:<6}  "
              f"{round(statistics.mean(d['tokens']),0) if d['tokens'] else 0:>8.0f} "
              f"{round(statistics.mean(d['llm_ms']),0) if d['llm_ms'] else 0:>7.0f} "
              f"{round(statistics.mean(d['total_ms']),0) if d['total_ms'] else 0:>7.0f}")

    print(f"\nWrote: {out_path}")
    print(f"       {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
