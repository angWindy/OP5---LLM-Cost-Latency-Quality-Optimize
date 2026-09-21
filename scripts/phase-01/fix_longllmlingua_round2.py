#!/usr/bin/env python3
"""
Fix round 2 — target the remaining 8 failures from n=18.

Failures grouped by root cause:

  A. Compressor too aggressive (need more context):
     - L12, L14 (space_digest) — need ALL reviews to compute %,
       k=10 sentences only covers ~half.  -> try k=30, rate=0.7
     - L00 (musique multi-hop), L10 (squality),
       L16 (qmsum) — compressor cut key sentences.
       -> try k=20 (more context)

  B. Output truncation:
     - L08 (gov_report), L13/L15 (summ_screen_fd),
       L14 (space_digest) — output >512 tokens
       -> max_tokens 512 -> 1024

  C. Format mismatch:
     - L13/L15 (summ_screen_fd) — model summarized a scene
       instead of the whole episode. With more context (k=20)
       it should pick up the broader arc.

Strategy: re-run only the failing cases with TWO new configs:
  F1: k=15, rate=0.5, max_tokens=1024 (default tune)
  F2: k=30, rate=0.7, max_tokens=1024 (high-recall)

Then LLM-as-judge again.

Output:
  results/phase-01-longllmlingua-opt-n20-fix2.jsonl
  results/phase-01-longllmlingua-opt-n20-fix2-summary.json

Usage:
  conda activate vsf
  python scripts/phase-01/fix_longllmlingua_round2.py
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
    MAX_TOKENS_LLM,
)
from longllmlingua_opt import (
    preselect_tfidf,
    compress_llmlingua2,
    get_gemini,
)

# Cases that failed after round 1, by task — drives per-task knob choice
FIX_CASES = {
    "space_digest": {"k": 30, "rate": 0.7, "max_tokens": 1024},   # A+B: keep all reviews, allow longer output
    "summ_screen_fd": {"k": 20, "rate": 0.5, "max_tokens": 1024}, # B+C: more context, longer output
    "gov_report": {"k": 20, "rate": 0.6, "max_tokens": 1024},     # B: longer output, more context
    "musique": {"k": 20, "rate": 0.5, "max_tokens": 1024},        # A: multi-hop needs more context
    "squality": {"k": 20, "rate": 0.5, "max_tokens": 1024},       # A: free-form needs more context
    "qmsum": {"k": 20, "rate": 0.6, "max_tokens": 1024},          # A+B: long meeting summary
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


def run_one(row: dict, k: int, rate: float, max_tokens: int) -> dict:
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

    out_path = args.out or (REPO_ROOT / "results" / "phase-01-longllmlingua-opt-n20-fix2.jsonl")
    summary_path = args.summary or (REPO_ROOT / "results" / "phase-01-longllmlingua-opt-n20-fix2-summary.json")

    print("=== LongLLMLingua fix round 2 (per-task knobs) ===")
    print(f"  Per-task config (k, rate, max_tokens):")
    for t, cfg in FIX_CASES.items():
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
            # Use per-task config if defined, else use defaults (k=15, rate=0.5, max_tokens=1024)
            cfg = FIX_CASES.get(task, {"k": 15, "rate": 0.5, "max_tokens": 1024})
            t0 = time.perf_counter()
            rec = run_one(row, k=cfg["k"], rate=cfg["rate"], max_tokens=cfg["max_tokens"])
            rec["ts"] = now_iso()
            rec["case_id"] = case_id
            rec["task"] = task
            rec["ctx_chars"] = len(row["context"])
            all_records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            if rec.get("status") == "ok":
                marker = "OK" if rec["judge_correct"] else "X"
                print(f"  [{case_id}] task={task:<14s} k={cfg['k']:>2} rate={cfg['rate']:.2f} "
                      f"ctx={len(row['context']):>5,} ps={rec['ps_chars']:>5,} "
                      f"comp={rec['comp_ms']:>5.0f}ms llm={rec['llm_ms']:>5.0f}ms "
                      f"gold={rec['gold'][:20]:<20s} -> {marker}")
            else:
                print(f"  [{case_id}] ERROR: {rec.get('error', rec.get('status',''))[:80]}")
            time.sleep(SLEEP)
            _ = (time.perf_counter() - t0)

    ok = sum(1 for r in all_records if r.get("status") == "ok")
    corr = sum(1 for r in all_records if r.get("judge_correct"))
    print(f"\n  Total: {corr}/{ok} correct ({corr/ok:.1%})" if ok else "  No records.")

    summary = {"ts": now_iso(), "n_cases": len(rows), "fix": "per-task knobs round 2", "configs": FIX_CASES}
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
            "knob": FIX_CASES.get(t, {"k": 15, "rate": 0.5, "max_tokens": 1024}),
        }
        for t, d in sorted(per_task.items())
    }
    summary["overall_acc"] = round(corr / ok, 3) if ok else None
    summary["overall_correct"] = corr
    summary["overall_n"] = ok

    with summary_path.open("w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" SUMMARY - Round 2 (per-task knobs)")
    print("=" * 100)
    print(f"  {'Task':<18s} {'Acc':>9s} {'InTok':>8s} {'LlmMs':>7s} {'TotMs':>7s}  Knob (k, rate, mt)")
    print("  " + "-" * 90)
    for t, d in sorted(per_task.items()):
        knob = FIX_CASES.get(t, {"k": 15, "rate": 0.5, "max_tokens": 1024})
        print(f"  {t:<18s} {d['correct']}/{d['n']:<6}  "
              f"{round(statistics.mean(d['tokens']),0) if d['tokens'] else 0:>8.0f} "
              f"{round(statistics.mean(d['llm_ms']),0) if d['llm_ms'] else 0:>7.0f} "
              f"{round(statistics.mean(d['total_ms']),0) if d['total_ms'] else 0:>7.0f}  "
              f"k={knob['k']:>2} rate={knob['rate']:.2f} mt={knob['max_tokens']}")

    print(f"\nWrote: {out_path}")
    print(f"       {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
