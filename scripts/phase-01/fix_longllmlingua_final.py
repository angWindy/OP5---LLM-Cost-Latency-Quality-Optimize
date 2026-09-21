#!/usr/bin/env python3
"""
Fix final — hybrid best-of-all-rounds config.

Round comparison (LLM-as-judge):
  R1 (k=10, r=0.5, mt=512): 55.6% (10/18) — best overall
  R2 (per-task k/r, mt=1024): 50.0% (9/18)
  R3 (no-compress): 33.3% (6/18) — worst

Per-task best:
  book_sum_sort -> R1 or R2 (both 100%)
  gov_report   -> R1 (50%)
  musique      -> R1 or R3 (50%)
  qasper      -> any (100%)
  qmsum        -> R2 (100%)
  quality      -> R1 or R2 (both 100%)
  space_digest -> all 0% (hard: need exact %, compressor cuts detail)
  squality     -> R1 or R2 (50%)
  summ_screen_fd -> all 0% (hard: need full episode summary)

Hypothesis: max_tokens=1024 for ALL tasks (instead of 512) might close the gap.
Also: the only real failure modes are:
  1. Task where exact string match is needed (space_digest: "70%")
  2. Task needing full context (summ_screen_fd: multi-scene summary)
  3. Tasks where LLM paraphrases instead of extracts

Strategy: run final hybrid with:
  - k=10, rate=0.5 (round 1 config, best)
  - max_tokens=1024 (round 2 improvement, should help truncation)
  - Same 18 cases (seed=42)

Output:
  results/phase-01-longllmlingua-opt-n20-final.jsonl

Usage:
  conda activate vsf
  python scripts/phase-01/fix_longllmlingua_final.py
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

FINAL_K = 10
FINAL_RATE = 0.5
FINAL_MAX_TOKENS = 1024


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


def run_final(row: dict) -> dict:
    context = row["context"]
    question = row["question"]
    gold = row["answer"]
    instruction = row.get("_orig_instruction", "")

    ps = preselect_tfidf(context, question, k=FINAL_K)
    if ps["status"] != "ok":
        return {"status": "preselect_error", "error": ps.get("error", "")}

    cc = compress_llmlingua2(ps["selected_text"], question, rate=FINAL_RATE)
    if cc["status"] != "ok":
        return {"status": "compress_error", "error": cc.get("error", "")}

    prompt = build_qa_prompt_fixed(cc["compressed_text"], question, instruction)
    g = get_gemini()
    ll = call_gemini_fixed(g, prompt, max_tokens=FINAL_MAX_TOKENS)
    if ll["status"] != "ok":
        return {"status": "llm_error", "error": ll.get("error", "")}

    jj = judge_v2(gold, ll["text"])
    return {
        "status": "ok",
        "k": FINAL_K,
        "rate": FINAL_RATE,
        "max_tokens": FINAL_MAX_TOKENS,
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
        "pred": ll["text"][:500],
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
    args = parser.parse_args()

    out_path = args.out or (REPO_ROOT / "results" / "phase-01-longllmlingua-opt-n20-final.jsonl")

    print("=== LongLLMLingua FINAL (R1 base + mt=1024) ===")
    print(f"  k={FINAL_K}, rate={FINAL_RATE}, max_tokens={FINAL_MAX_TOKENS}")
    print(f"  seed={args.seed}, n={args.n}")

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
            rec = run_final(row)
            rec["ts"] = now_iso()
            rec["case_id"] = case_id
            rec["task"] = task
            rec["ctx_chars"] = len(row["context"])
            all_records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            if rec.get("status") == "ok":
                marker = "OK" if rec["judge_correct"] else "X"
                print(f"  [{case_id}] task={task:<14s} "
                      f"ctx={len(row['context']):>5,} ps={rec['ps_chars']:>5,} "
                      f"comp_ratio={rec.get('comp_ratio','?'):<6} "
                      f"in_tok={rec.get('input_tokens','?'):>4} "
                      f"llm={rec['llm_ms']:>5.0f}ms gold={rec['gold'][:20]:<20s} -> {marker}")
            else:
                print(f"  [{case_id}] ERROR: {rec.get('error', rec.get('status',''))[:80]}")
            time.sleep(SLEEP)

    ok = sum(1 for r in all_records if r.get("status") == "ok")
    corr = sum(1 for r in all_records if r.get("judge_correct"))
    print(f"\n  Total: {corr}/{ok} correct ({corr/ok:.1%})" if ok else "  No records.")

    per_task = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": [], "llm_ms": []})
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

    print(f"\n  Per-task (heuristic judge):")
    print(f"  {'Task':<18s} {'Acc':>9s} {'AvgTok':>8s} {'AvgLlmMs':>9s}")
    print("  " + "-" * 50)
    for t in sorted(per_task):
        d = per_task[t]
        print(f"  {t:<18s} {d['correct']}/{d['n']:<6}  "
              f"{round(statistics.mean(d['tokens']),0) if d['tokens'] else 0:>8.0f} "
              f"{round(statistics.mean(d['llm_ms']),0):>9.0f}")

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
