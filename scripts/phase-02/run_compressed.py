"""
run_compressed.py — LongBench run with LongLLMLingua compression (predict-only by default).

By default this script ONLY produces predictions (no LLM-as-judge).
To score the output, run judge_jsonl.py afterwards:

  conda activate vsf
  python scripts/phase-02/run_compressed.py --rate 0.4
  python scripts/phase-02/judge_jsonl.py \
      --in results/phase-02-run-rate40.jsonl \
      --out results/phase-02-run-rate40-judged.jsonl

If you want to run judge inline (debug only, default is OFF), pass --judge.

Auto-detects CUDA. Falls back to CPU if no GPU is available.
Use --device to force a specific device.

Usage:
  # Predict only (default):
  python scripts/phase-02/run_compressed.py --rate 0.4
  python scripts/phase-02/run_compressed.py --rate 0.5 --sleep 3
  python scripts/phase-02/run_compressed.py --rate 0.6 --device cuda     # force GPU
  python scripts/phase-02/run_compressed.py --rate 0.7 --device cpu      # force CPU

  # Predict + judge inline (debug only):
  python scripts/phase-02/run_compressed.py --rate 0.4 --judge

Outputs:
  results/phase-02-run-rate{XX}.jsonl
  results/phase-02-run-rate{XX}-summary.json
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
# Device auto-detect
# ---------------------------------------------------------------
def detect_device(requested: str = "auto") -> str:
    """Resolve device string. 'auto' prefers cuda if available, else cpu."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA available - {name}")
            return "cuda"
        print("[device] CUDA not available - falling back to CPU")
        return "cpu"
    except ImportError:
        print("[device] torch not installed - falling back to CPU")
        return "cpu"


# ---------------------------------------------------------------
# LongLLMLingua loader (lazy, device-aware)
# ---------------------------------------------------------------
_pc = None
_pc_device = None


def get_compressor(device: str):
    """Lazy-init the PromptCompressor on the chosen device."""
    global _pc, _pc_device
    if _pc is not None and _pc_device == device:
        return _pc
    import torch
    # CPU threading: leave at defaults, but cap to a sane number
    if device == "cpu" and not os.environ.get("OMP_NUM_THREADS"):
        try:
            import psutil
            torch.set_num_threads(psutil.cpu_count(logical=False) or 8)
        except Exception:
            torch.set_num_threads(8)
    from llmlingua import PromptCompressor
    print(f"[device] Loading llmlingua-2 on {device} ...")
    _pc = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map=device,
    )
    _pc_device = device
    print(f"[device] Compressor ready on {device}")
    return _pc


def compress(pc, context: str, question: str, rate: float) -> dict:
    t0 = time.perf_counter()
    try:
        result = pc.compress_prompt(
            context,
            question=question,
            rate=rate,
            target_token=-1,
            token_budget_ratio=1.4,
            iterative_size=512,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        ms = (time.perf_counter() - t0) * 1000
        compressed = result.get("compressed_prompt", "")
        origin = int(result.get("origin_tokens", 0))
        comp_tok = int(result.get("compressed_tokens", 0))
        ratio_raw = result.get("ratio", 0)
        if isinstance(ratio_raw, str):
            ratio = float(ratio_raw.replace("x", "")) if ratio_raw else 0
        else:
            ratio = float(ratio_raw) if ratio_raw else 0
        if ratio == 0 and origin > 0 and comp_tok > 0:
            ratio = origin / comp_tok
        return {
            "status": "ok",
            "compressed_text": compressed,
            "origin_tokens": origin,
            "compressed_tokens": comp_tok,
            "ratio": ratio,
            "compress_ms": ms,
        }
    except Exception as exc:
        return {
            "status": "error", "error": str(exc),
            "compress_ms": (time.perf_counter() - t0) * 1000,
        }


# ---------------------------------------------------------------
# Gemini call (with retry/backoff)
# ---------------------------------------------------------------
def call_gemini(prompt: str, model: str, *, max_retries: int = 4,
                sleep_between: float = 5.0) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}

    url = GEMINI_URL.format(model=model) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    backoff = [10, 20, 40, 80, 160]
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                usage = data.get("usageMetadata", {})
                return {
                    "status": "ok",
                    "text": text,
                    "latency_ms": elapsed,
                    "input_tokens": usage.get("promptTokenCount"),
                    "output_tokens": usage.get("candidatesTokenCount"),
                }
            elif resp.status_code in (429, 503):
                wait = backoff[min(attempt, len(backoff) - 1)]
                print(f"    HTTP {resp.status_code} - sleeping {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "latency_ms": elapsed,
                }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            wait = backoff[min(attempt, len(backoff) - 1)]
            print(f"    {type(exc).__name__} - retry in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
    return {"status": "error", "error": "Max retries exceeded", "latency_ms": 0}


# ---------------------------------------------------------------
# Judge
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
    p.add_argument("--rate", type=float, required=True,
                   help="LongLLMLingua keep-ratio (e.g. 0.4 / 0.5 / 0.6 / 0.7)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device (default: auto - prefers cuda)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--judge", dest="run_judge", action="store_true",
                   help="Also run LLM-as-judge inline after predicting (default: predict only). "
                        "For normal scoring workflow, run judge_jsonl.py instead.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Judge model when --judge is set (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--sleep", type=float, default=3.0,
                   help="Sleep between Gemini calls (seconds)")
    p.add_argument("--input", default=None,
                   help="Path to input JSONL dataset (default: longbench_200_stratified.jsonl)")
    p.add_argument("--out-tag", default=None,
                   help="Override output suffix (default: rate{XX})")
    args = p.parse_args()
    sample_path = Path(args.input) if args.input else SAMPLE_PATH

    rate = args.rate
    if not 0 < rate <= 1.0:
        print(f"FATAL: --rate must be in (0, 1], got {rate}")
        return 1

    device = detect_device(args.device)
    tag = args.out_tag or f"rate{int(rate * 100):02d}"
    out_path = RESULTS_DIR / f"phase-02-run-{tag}.jsonl"
    summary_path = RESULTS_DIR / f"phase-02-run-{tag}-summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  LongBench FULL run - LongLLMLingua rate={rate}  |  N=20 cases")
    print(f"  Device : {device}")
    print(f"  Gemini : {args.model}")
    print(f"  Judge  : {args.judge_model if args.run_judge else 'DISABLED (use --judge to enable)'}")
    print(f"  Output : {out_path.relative_to(REPO_ROOT)}")
    print("=" * 70)

    if not sample_path.exists():
        print(f"FATAL: sample not found: {sample_path}")
        return 1
    sample = [json.loads(l) for l in sample_path.open() if l.strip()]
    print(f"Loaded {len(sample)} cases from {sample_path.name}")

    # Load compressor once
    pc = get_compressor(device)

    # Resume support
    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("status") == "ok"
                        and r.get("pred")
                        and (args.run_judge or r.get("judge_correct") is not None)):
                    results.append(r)
                    done_ids.add(r["case_id"])
        if done_ids:
            print(f"Resuming with {len(done_ids)} complete records")

    print(f"\n[1/2] Compression + Gemini (rate={rate})...")
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
            print(f"  [{i+1:>2}/{len(sample)}] {case_id[:30]} SKIP (missing q/context)")
            continue
        if case_id in done_ids:
            print(f"  [{i+1:>2}/{len(sample)}] {case_id[:30]} - already done, skip")
            continue

        comp = compress(pc, context, question, rate)
        if comp["status"] != "ok":
            print(f"  [{i+1:>2}/{len(sample)}] {case_id[:30]} "
                  f"COMPRESS ERR: {comp.get('error')}")
            continue

        prompt = _format_eval_prompt(comp["compressed_text"], question)
        llm = call_gemini(prompt, args.model, sleep_between=args.sleep)
        if llm["status"] != "ok":
            print(f"  [{i+1:>2}/{len(sample)}] {case_id[:30]} "
                  f"LLM ERR: {llm.get('error')}")
            continue

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_id": case_id,
            "task": task,
            "question": question[:200],
            "gold": gold[:200],
            "pred": llm["text"][:500],
            "status": "ok",
            "mode": "longllmlingua",
            "rate": rate,
            "device": device,
            "compress_ms": round(float(comp["compress_ms"]), 1),
            "origin_tokens": comp["origin_tokens"],
            "compressed_tokens": comp["compressed_tokens"],
            "compress_ratio": round(float(comp["ratio"]), 2),
            "llm_model": args.model,
            "llm_latency_ms": float(llm["latency_ms"]),
            "input_tokens": llm.get("input_tokens"),
            "output_tokens": llm.get("output_tokens"),
            "judge_correct": None,
            "judge_reason": "",
            "judge_model": "",
            "judge_latency_ms": 0.0,
        }
        results.append(record)

        with out_path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [{i+1:>2}/{len(sample)}] {case_id[:30]} "
              f"orig={comp['origin_tokens']} comp={comp['compressed_tokens']} "
              f"ratio={comp['ratio']:.2f}x "
              f"comp_ms={comp['compress_ms']:.0f} "
              f"gem_ms={llm['latency_ms']:.0f}")
        time.sleep(args.sleep)

    if not args.run_judge:
        print(f"\n[2/2] Judge: SKIPPED (predict-only mode)")
        print(f"         To score results, run:")
        print(f"         python scripts/phase-02/judge_jsonl.py \\")
        print(f"             --in {out_path.relative_to(REPO_ROOT)} \\")
        print(f"             --out results/phase-02-run-{tag}-judged.jsonl")
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
    print(f"  SUMMARY - {tag} (rate={rate}, device={device})")
    print(f"{'=' * 70}")
    judged = [r for r in results if r.get("judge_correct") is not None]
    correct = sum(1 for r in judged if r["judge_correct"])
    accuracy = correct / len(judged) if judged else 0
    comp_lats = [r["compress_ms"] / 1000 for r in results]
    gem_lats = [r["llm_latency_ms"] / 1000 for r in results]
    judge_lats = [r["judge_latency_ms"] / 1000 for r in judged]
    in_toks = [r["input_tokens"] or 0 for r in results]
    out_toks = [r["output_tokens"] or 0 for r in results]
    ratios = [r["compress_ratio"] for r in results if r.get("compress_ratio")]

    summary = {
        "tag": tag,
        "mode": "longllmlingua",
        "rate": rate,
        "device": device,
        "n_total": len(sample),
        "n_completed": len(results),
        "n_judged": len(judged),
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "model": args.model,
        "judge_model": args.judge_model if args.run_judge else None,
        "compression": {
            "ratio_median": round(statistics.median(ratios), 2) if ratios else None,
            "ratio_min": round(min(ratios), 2) if ratios else None,
            "ratio_max": round(max(ratios), 2) if ratios else None,
        },
        "latency": {
            "compress_s": _stats(comp_lats),
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
    print(f"  Compress med: {statistics.median(comp_lats):.1f}s")
    print(f"  Gemini med  : {statistics.median(gem_lats):.1f}s")
    if gem_lats:
        print(f"  Gemini p95  : {sorted(gem_lats)[max(0, int(len(gem_lats) * 0.95) - 1)]:.1f}s")
    if ratios:
        print(f"  Ratio med   : {statistics.median(ratios):.2f}x")
    print(f"  Input tok med/p95/max : "
          f"{summary['tokens']['input_median']}/"
          f"{summary['tokens']['input_p95']}/"
          f"{summary['tokens']['input_max']}")
    print(f"  Output tok med : {summary['tokens']['output_median']}")
    print(f"  Summary    : {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
