#!/usr/bin/env python3
"""
LLM-as-judge on Phase-02 baseline 200 (English prompt).

Reads:
  results/phase-02-baseline-200-en.jsonl  (Gemini baseline predictions)
  data/processed/zero_scrolls_200.jsonl    (ZeroSCROLLS contexts + gold)

For each of 200 cases, calls LLMJudge chain (NIM -> OpenRouter -> Gemini)
with (context, question, gold, pred) and writes JudgeResult fields back
into the baseline JSONL (in-place replace by case_id).

Output:
  results/phase-02-baseline-200-en-judged.jsonl  - full records w/ judge fields
  results/phase-02-baseline-200-en-judged-summary.json
  results/phase-02-baseline-200-en-judge-ops.jsonl - ops log (retries, skips)

Usage:
  conda activate vsf
  python scripts/phase-02/judge_200.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT / "src"))

from op5.llm import LLMJudge


BASELINE_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en.jsonl"
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
OUT_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en-judged.jsonl"
OPS_LOG = REPO_ROOT / "results" / "phase-02-baseline-200-en-judge-ops.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en-judged-summary.json"


def load_jsonl(path: Path) -> list[dict]:
    decoder = json.JSONDecoder()
    buf = path.read_text(encoding="utf-8")
    rows, pos = [], 0
    while pos < len(buf):
        while pos < len(buf) and buf[pos] in " \t\n\r":
            pos += 1
        if pos >= len(buf):
            break
        try:
            obj, end = decoder.raw_decode(buf, pos)
            rows.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return rows


def build_context_map(sample_rows: list[dict]) -> dict[str, dict]:
    """case_id -> {context, question, gold}"""
    m: dict[str, dict] = {}
    for r in sample_rows:
        cid = r.get("id", "")
        inp = r.get("input", "")
        ds = int(r.get("document_start_index", 0))
        de = int(r.get("document_end_index", 0))
        qs = int(r.get("query_start_index", 0))
        qe = int(r.get("query_end_index", 0))
        m[cid] = {
            "context": inp[ds:de],
            "question": inp[qs:qe],
            "gold": r.get("output", "") or "",
        }
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge on Phase-02 baseline 200")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--ops", type=Path, default=OPS_LOG)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep seconds between judge calls (rate limit cushion)")
    args = parser.parse_args()

    print("=== Phase 02 — LLM-as-judge on baseline 200 ===\n")

    # 1. Load
    print(f"[1/3] Loading baseline + sample...")
    baseline = load_jsonl(args.baseline)
    sample_rows = load_jsonl(args.sample)
    ctx_map = build_context_map(sample_rows)
    print(f"  baseline: {len(baseline)} rows")
    print(f"  sample:   {len(sample_rows)} rows")
    print(f"  context map: {len(ctx_map)} entries")

    # Match baseline to sample
    judge_cases = []
    skipped = []
    for rec in baseline:
        cid = rec.get("case_id", "")
        info = ctx_map.get(cid)
        if not info:
            skipped.append((cid, "no context match"))
            continue
        if rec.get("status") != "ok":
            skipped.append((cid, f"baseline status={rec.get('status')}"))
            continue
        judge_cases.append({
            "case_id": cid,
            "task": rec.get("task"),
            "question": info["question"],
            "gold": info["gold"],
            "pred": rec.get("text", ""),
            "context": info["context"],
            "truncated": rec.get("truncated", False),
            "approx_input_tokens": rec.get("approx_input_tokens", 0),
        })
    print(f"  to judge: {len(judge_cases)} | skipped: {len(skipped)}")
    if skipped[:3]:
        for cid, reason in skipped[:3]:
            print(f"    skip {cid}: {reason}")

    # 2. Init judge
    print(f"\n[2/3] Initializing LLMJudge (chain: nim -> openrouter -> gemini)...")
    judge = LLMJudge(log_path=args.ops)

    # 3. Judge loop
    print(f"\n[3/3] Judging {len(judge_cases)} cases...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    verdicts: dict[str, int] = defaultdict(int)
    per_task_verdict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_status: dict[str, int] = defaultdict(int)

    total = len(judge_cases)
    out_records: list[dict] = []

    for i, c in enumerate(judge_cases):
        # Find the baseline record to preserve original fields
        baseline_rec = next((b for b in baseline if b.get("case_id") == c["case_id"]), {})

        try:
            result = judge.judge(
                question=c["question"],
                gold=c["gold"],
                pred=c["pred"],
                context=c["context"],
            )
        except Exception as exc:  # safety net
            result = None
            print(f"  [{i+1}/{total}] case_id={c['case_id']} EXCEPTION: {exc}")

        if result is None:
            by_status["error"] += 1
            verdict = "error"
            confidence = 0.0
            reason = "judge returned None"
            model_used = "none"
            latency_ms = 0.0
            correct = None
        else:
            verdict = result.verdict
            confidence = result.confidence
            reason = result.reason
            model_used = result.model_used
            latency_ms = result.latency_ms
            correct = result.correct
            by_status[verdict] += 1

        verdicts[verdict] += 1
        per_task_verdict[c["task"]][verdict] += 1

        merged = dict(baseline_rec)
        merged.update({
            "judge_verdict": verdict,
            "judge_correct": correct,
            "judge_reason": reason[:200],
            "judge_confidence": confidence,
            "judge_model": model_used,
            "judge_latency_ms": round(latency_ms, 1),
            "judge_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        out_records.append(merged)

        marker = {"correct": "+", "incorrect": "-", "ambiguous": "~",
                  "error": "E", "timeout": "T"}.get(verdict, "?")
        if i % 10 == 0 or verdict not in ("correct", "ambiguous", "incorrect"):
            print(f"  [{i+1:>3}/{total}] {marker} {c['task']:<14} "
                  f"cid={c['case_id']:<30} v={verdict:<10} conf={confidence:.2f} "
                  f"model={model_used:<30} ms={latency_ms:>6.0f}")

        # Append incrementally (so we don't lose progress on interrupt)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(merged, ensure_ascii=False) + "\n")

        time.sleep(args.sleep)

    # 4. Summary
    judged = total - by_status.get("error", 0)
    correct_n = verdicts.get("correct", 0)
    incorrect_n = verdicts.get("incorrect", 0)
    ambiguous_n = verdicts.get("ambiguous", 0)
    accuracy = correct_n / judged if judged else 0.0
    # Standard "strict" accuracy = correct / total
    strict_accuracy = correct_n / total if total else 0.0

    summary = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": str(args.baseline),
        "n_total": total,
        "n_judged": judged,
        "n_skipped": len(skipped),
        "verdicts": dict(verdicts),
        "accuracy_strict": round(strict_accuracy, 4),
        "accuracy_excl_ambig": round(correct_n / (correct_n + incorrect_n), 4) if (correct_n + incorrect_n) else 0.0,
        "per_task": {
            t: dict(v) for t, v in sorted(per_task_verdict.items())
        },
        "judge_chain": [p.get("label", "?") for p in judge._chain],
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== Summary ===")
    print(f"  Total:      {total}")
    print(f"  Judged:     {judged}")
    print(f"  Verdicts:   {dict(verdicts)}")
    print(f"  Strict accuracy (correct/total):       {strict_accuracy*100:.1f}%")
    if (correct_n + incorrect_n):
        print(f"  Excl-ambiguous accuracy:              {summary['accuracy_excl_ambig']*100:.1f}%")
    print(f"  Summary: {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
