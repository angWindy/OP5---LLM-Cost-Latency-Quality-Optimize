#!/usr/bin/env python3
"""
LLM-as-judge on a JSONL file, with configurable profile routing.

Usage:
    # Auto-chain (NIM -> OpenRouter -> Gemini)
    python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl

    # Single profile (fastest)
    python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl --profile nim

    # Custom chain
    python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl \
        --profiles nim openrouter

    # Custom profile files
    python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl \
        --profile-files ./my-judge-profile.yaml

    # Use specific field names
    python scripts/phase-01/judge_llm_profile.py results/my_eval.jsonl \
        --gold-field answer --pred-field prediction --question-field question

Input JSONL schema (required fields):
    gold  : ground truth answer  (or: answer, expected, reference)
    pred  : model prediction    (or: prediction, output, generated)

Optional fields:
    question : the evaluation question
    context  : supporting context

Output JSONL (one line per input):
    Same as input + judge_verdict, judge_correct, judge_reason,
    judge_confidence, judge_model_used, judge_latency_ms

After processing, prints summary table.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from op5.llm import LLMJudge

_GOLD_NAMES     = ("gold","answer","expected","reference","ground_truth","gold_answer")
_PRED_NAMES    = ("pred","prediction","output","generated","model_output","system_output")
_QUESTION_NAMES = ("question","q","query","prompt")
_CONTEXT_NAMES  = ("context","ctx","passage","document")

def _get(row, *names):
    for n in names:
        if n in row: return str(row[n])
    return ""

def _run(input_path, args):
    judge = LLMJudge(
        profile=args.profile,
        profiles=args.profiles,
        profile_files=args.profile_files,
        log_path=args.log,
    )
    stats    = {"correct":0,"incorrect":0,"ambiguous":0,"error":0}
    total_ms = 0.0
    out_path = Path(args.output) if args.output else input_path.with_suffix(".judged.jsonl")
    out_lines = []
    rows = [json.loads(l) for l in input_path.open(encoding="utf-8")]

    print(f"Input : {input_path}  ({len(rows)} rows)")
    print(f"Output: {out_path}")
    chain_desc = args.profile or " -> ".join(args.profiles) or "auto-chain"
    print(f"Profile: {chain_desc}")
    print()

    for i, row in enumerate(rows, 1):
        gold     = _get(row, *_GOLD_NAMES)
        pred     = _get(row, *_PRED_NAMES)
        question = _get(row, *_QUESTION_NAMES)
        context  = _get(row, *_CONTEXT_NAMES)
        if not gold or not pred:
            print(f"  Row {i}: SKIP — missing gold/pred fields"); continue
        r = judge.judge(question=question, gold=gold, pred=pred, context=context)
        stats[r.verdict] = stats.get(r.verdict, 0) + 1
        total_ms += r.latency_ms
        out_row = dict(row)
        out_row.update(
            judge_verdict=r.verdict, judge_correct=r.correct,
            judge_reason=r.reason,   judge_confidence=r.confidence,
            judge_model_used=r.model_used, judge_latency_ms=round(r.latency_ms,1))
        out_lines.append(json.dumps(out_row, ensure_ascii=False))
        print(f"  [{i:>3}] {r.verdict:<12} conf={r.confidence:.1f}  "
              f"{r.latency_ms:>6.0f}ms  {r.model_used.split('/')[-1]:<25}  "
              f"pred={pred[:40]!r}  gold={gold[:40]!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    total = sum(stats.values())
    print()
    print("=" * 60)
    print(f"  Total     : {total}")
    print(f"  correct   : {stats['correct']}  ({stats['correct']/total*100:.0f}%)")
    print(f"  incorrect : {stats['incorrect']}  ({stats['incorrect']/total*100:.0f}%)")
    print(f"  ambiguous : {stats['ambiguous']}  ({stats['ambiguous']/total*100:.0f}%)")
    print(f"  error     : {stats['error']}  ({stats['error']/total*100:.0f}%)")
    print(f"  avg latency: {total_ms/total:.0f}ms  total: {total_ms/1000:.1f}s")
    print(f"  Wrote: {out_path}")

def main():
    p = argparse.ArgumentParser(description="LLM-as-judge on JSONL with profile routing")
    p.add_argument("input", type=Path, help="Input JSONL file")
    p.add_argument("--output","-o", type=str, default=None, help="Output JSONL path")
    p.add_argument("--profile", type=str, default=None, help="Single profile name")
    p.add_argument("--profiles", type=str, nargs="+", default=None, help="Profile chain")
    p.add_argument("--profile-files", type=Path, nargs="+", default=None, help="Custom YAML files")
    p.add_argument("--log", type=str, default=None, help="Ops log path (JSONL)")
    args = p.parse_args()
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr); sys.exit(1)
    _run(args.input, args)

if __name__ == "__main__": main()
