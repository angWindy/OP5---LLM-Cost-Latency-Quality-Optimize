"""
run_baseline.py — LongBench baseline run (no compression, predict-only by default).

By default this script ONLY produces predictions (no LLM-as-judge).
To score the output, run judge_jsonl.py afterwards:

  conda activate vsf
  export GOOGLE_API_KEYS="key1,key2,key3,key4"   # multi-key rotation (or GOOGLE_API_KEY for single)
  python scripts/phase-02/run_baseline.py --out-tag baseline_n196
  python scripts/phase-02/judge_jsonl.py \
      --in results/phase-02-run-baseline_n196.jsonl \
      --out results/phase-02-run-baseline_n196-judged.jsonl

If you want to run judge inline (debug only, default is OFF), pass --judge.

API key rotation: set GOOGLE_API_KEYS="k1,k2" (comma-separated) in env.
On 429/503/timeout, the client rotates to the next key and retries immediately.
See src/op5/llm/gemini_client.py for rotation logic.

Usage:
  # Predict only (default):
  python scripts/phase-02/run_baseline.py --out-tag baseline_n196
  python scripts/phase-02/run_baseline.py --model gemini-3.5-flash-lite --sleep 3 --out-tag baseline_n196

  # Predict + judge inline (debug only):
  python scripts/phase-02/run_baseline.py --out-tag baseline_n196 --judge

Outputs:
  results/phase-02-run-{tag}.jsonl          one record per case
  results/phase-02-run-{tag}-summary.json   latency stats
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")
from _prompts import format_eval_prompt as _format_eval_prompt

import requests


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_JUDGE_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "longbench_200_stratified.jsonl"
RESULTS_DIR = REPO_ROOT / "results" / "runs"


# ---------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------
def judge(question: str, gold: str, pred: str, model: str) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return {"judge_correct": None, "judge_reason": "no_api_key",
                "judge_model": model, "judge_latency_ms": 0}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    judge_prompt = (
        "You are an expert evaluator for a question-answering task.\n\n"
        "Given the original question, the reference answer, and the model's response,\n"
        "determine whether the response is CORRECT, INCORRECT, or UNCERTAIN.\n\n"
        "Rules:\n"
        "  - CORRECT: the response contains or substantively matches the reference answer.\n"
        "  - INCORRECT: the response contradicts or misses the reference answer.\n"
        "  - UNCERTAIN: you cannot determine correctness from the information given.\n\n"
        "Output exactly one line: RESULT: CORRECT | INCORRECT | UNCERTAIN\n"
        "Then one line explaining your reasoning (1-2 sentences).\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {gold}\n\n"
        f"Model Response: {pred}\n"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(DEEPSEEK_BASE_URL, json=payload,
                             headers=headers, timeout=60)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return {"judge_correct": None,
                    "judge_reason": f"HTTP {resp.status_code}",
                    "judge_model": model, "judge_latency_ms": elapsed}
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("RESULT: CORRECT"):
            correct = True
        elif content.startswith("RESULT: INCORRECT"):
            correct = False
        else:
            correct = None
        reason = content.split("\n", 1)[-1].strip() if "\n" in content else content
        return {"judge_correct": correct, "judge_reason": reason,
                "judge_model": model, "judge_latency_ms": elapsed}
    except Exception as exc:
        return {"judge_correct": None, "judge_reason": str(exc),
                "judge_model": model,
                "judge_latency_ms": (time.perf_counter() - t0) * 1000}


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "p95": round(sorted(vals)[max(0, int(len(vals) * 0.95) - 1)], 1),
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--judge", dest="run_judge", action="store_true",
                   help="Also run LLM-as-judge inline after predicting (default: predict only). "
                        "For normal scoring workflow, run judge_jsonl.py instead.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Judge model when --judge is set (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--sleep", type=float, default=3.0,
                   help="Sleep between Gemini calls (seconds)")
    p.add_argument("--out-tag", default="baseline",
                   help="Output suffix (default: baseline)")
    args = p.parse_args()

    out_path = RESULTS_DIR / f"phase-02-run-{args.out_tag}.jsonl"
    summary_path = RESULTS_DIR / f"phase-02-run-{args.out_tag}-summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Init Gemini client (key rotation via GOOGLE_API_KEYS env or single GOOGLE_API_KEY)
    from op5.llm import GeminiClient
    try:
        gemini = GeminiClient()
        n_keys = gemini.n_keys
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"  LongBench FULL run - BASELINE (no compression)  |  N=196 cases")
    print(f"  Gemini : {args.model}  ({n_keys} key(s) in pool)")
    print(f"  Judge  : {args.judge_model if args.run_judge else 'DISABLED (use --judge to enable)'}")
    print(f"  Output : {out_path.relative_to(REPO_ROOT)}")
    print("=" * 70)

    if not SAMPLE_PATH.exists():
        print(f"FATAL: sample not found: {SAMPLE_PATH}")
        return 1
    sample = [json.loads(l) for l in SAMPLE_PATH.open() if l.strip()]
    print(f"Loaded {len(sample)} cases from {SAMPLE_PATH.name}")

    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("pred"):
                    # Predict-only mode: skip if record already has a valid prediction
                    # Judge mode: additionally require judge_correct is not None
                    if args.run_judge and r.get("judge_correct") is None:
                        pass  # need to re-judge
                    else:
                        results.append(r)
                        done_ids.add(r["case_id"])
        if done_ids:
            print(f"Resuming with {len(done_ids)} complete records")

    print(f"\n[1/2] Gemini (no compression)...")
    for i, case in enumerate(sample):
        case_id = case.get("id", case.get("case_id", f"case_{i}"))
        task = case.get("task", "unknown")
        question = case.get("question", case.get("input", "")).strip()
        context = case.get("context", "")
        gold_ans = case.get("answer")
        if isinstance(gold_ans, list):
            gold = (gold_ans[0] if gold_ans else "").strip()
        elif gold_ans:
            gold = gold_ans.strip()
        else:
            answers = case.get("answers", [])
            gold = (answers[0] if answers else "").strip()

        if not question or not context:
            print(f"  [{i+1:>3}/{len(sample)}] {case_id[:30]} SKIP (missing q/context)")
            continue
        if case_id in done_ids:
            print(f"  [{i+1:>3}/{len(sample)}] {case_id[:30]} - already done, skip")
            continue

        prompt = _format_eval_prompt(context, question)
        llm = gemini.generate(prompt, model=args.model)
        if llm["status"] != "ok":
            err_key = llm.get("key_id", "?")
            print(f"  [{i+1:>3}/{len(sample)}] {case_id[:30]} LLM ERR "
                  f"[{err_key}]: {llm.get('error', 'unknown')[:80]}")
            continue

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_id": case_id,
            "task": task,
            "question": question[:200],
            "gold": gold[:200],
            "pred": llm["text"][:500],
            "status": "ok",
            "mode": "baseline",
            "compress_ms": 0.0,
            "origin_tokens": None,
            "compressed_tokens": None,
            "compress_ratio": None,
            "llm_model": args.model,
            "llm_latency_ms": float(llm["latency_ms"]),
            "input_tokens": llm.get("input_tokens"),
            "output_tokens": llm.get("output_tokens"),
            "gemini_key_id": llm.get("key_id"),
            "gemini_attempts": llm.get("attempts"),
            "judge_correct": None,
            "judge_reason": "",
            "judge_model": "",
            "judge_latency_ms": 0.0,
        }
        results.append(record)

        with out_path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [{i+1:>3}/{len(sample)}] {case_id[:30]} "
              f"key={llm.get('key_id')} "
              f"in_tok={llm.get('input_tokens')} out_tok={llm.get('output_tokens')} "
              f"gemini_ms={llm['latency_ms']:.0f}")
        time.sleep(args.sleep)

    if not args.run_judge:
        print(f"\n[2/2] Judge: SKIPPED (predict-only mode)")
        print(f"         To score results, run:")
        print(f"         python scripts/phase-02/judge_jsonl.py \\")
        print(f"             --in results/phase-02-run-{args.out_tag}.jsonl \\")
        print(f"             --out results/phase-02-run-{args.out_tag}-judged.jsonl")
    else:
        print(f"\n[2/2] Judging with {args.judge_model}...")
        to_judge = [r for r in results if r.get("judge_correct") is None]
        print(f"  {len(to_judge)} cases need judging")
        for rec in to_judge:
            jj = judge(rec["question"], rec["gold"], rec["pred"], args.judge_model)
            rec["judge_correct"] = jj["judge_correct"]
            rec["judge_reason"] = jj.get("judge_reason", "")[:200]
            rec["judge_model"] = jj["judge_model"]
            rec["judge_latency_ms"] = jj.get("judge_latency_ms", 0)

            lines = out_path.read_text().splitlines()
            for idx, line in enumerate(lines):
                if json.loads(line)["case_id"] == rec["case_id"]:
                    lines[idx] = json.dumps(rec, ensure_ascii=False)
                    break
            out_path.write_text("\n".join(lines) + "\n")

            jc = rec["judge_correct"]
            jc_s = "T" if jc else ("?" if jc is None else "F")
            print(f"  {rec['case_id'][:30]} judge={jc_s} "
                  f"judge_ms={jj.get('judge_latency_ms', 0):.0f}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY - {args.out_tag}")
    print(f"{'=' * 70}")
    judged = [r for r in results if r.get("judge_correct") is not None]
    correct = sum(1 for r in judged if r["judge_correct"])
    accuracy = correct / len(judged) if judged else 0
    gem_lats = [r["llm_latency_ms"] / 1000 for r in results]
    judge_lats = [r["judge_latency_ms"] / 1000 for r in judged]
    in_toks = [r["input_tokens"] or 0 for r in results]
    out_toks = [r["output_tokens"] or 0 for r in results]

    summary = {
        "tag": args.out_tag,
        "mode": "baseline",
        "n_total": len(sample),
        "n_completed": len(results),
        "n_judged": len(judged),
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "model": args.model,
        "judge_model": args.judge_model if args.run_judge else None,
        "gemini_key_pool": gemini.stats_summary(),
        "latency": {
            "gemini_s": _stats(gem_lats),
            "judge_s": _stats(judge_lats),
        },
        "tokens": {
            "input_median": int(statistics.median(in_toks)) if in_toks else 0,
            "input_p95": int(sorted(in_toks)[max(0, int(len(in_toks) * 0.95) - 1)]) if in_toks else 0,
            "input_max": max(in_toks) if in_toks else 0,
            "output_median": int(statistics.median(out_toks)) if out_toks else 0,
        },
        "ts_range": [results[0]["ts"] if results else None,
                     results[-1]["ts"] if results else None],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"  Completed   : {len(results)}/{len(sample)}")
    print(f"  Judged      : {len(judged)}")
    print(f"  Accuracy    : {accuracy:.1%} ({correct}/{len(judged)})")
    print(f"  Gemini med  : {statistics.median(gem_lats):.1f}s")
    if gem_lats:
        print(f"  Gemini p95  : {sorted(gem_lats)[max(0, int(len(gem_lats) * 0.95) - 1)]:.1f}s")
    print(f"  Key usage   : {gemini.stats_summary()}")
    print(f"  Input tok med/p95/max : "
          f"{summary['tokens']['input_median']}/"
          f"{summary['tokens']['input_p95']}/"
          f"{summary['tokens']['input_max']}")
    print(f"  Output tok med : {summary['tokens']['output_median']}")
    print(f"  Summary    : {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
