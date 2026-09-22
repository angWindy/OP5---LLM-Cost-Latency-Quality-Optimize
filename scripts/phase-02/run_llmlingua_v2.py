"""
LLMLingua-2 (task-agnostic compressor) benchmark on ZeroSCROLLS.

Pipeline per case:
  1. Optional TF-IDF preselect (top-k sentences by cosine similarity).
  2. LLMLingua-2 compress at given rate.
  3. Gemini call.
  4. LLM-as-judge evaluation.

This is the v2 of llmlingua_20.py — focused CLI, configurable, no inline defaults
that hide what the script is doing.

CLI flags:
  --n INT            number of cases (default 20)
  --seed INT         sample seed (default 42)
  --k INT            TF-IDF top-k (default 15; 0 = skip preselect)
  --rate FLOAT       LLMLingua rate (default 0.5)
  --iter-size INT    iterative_size (default 512)
  --model STR        Gemini model (default gemini-3.5-flash-lite)
  --judge-profile STR  judge profile (default nim; or openrouter/gemini/auto)
  --sleep FLOAT      seconds between Gemini calls (default 3.0)
  --tasks STR        comma-separated task subset
  --no-preselect     skip TF-IDF (compress raw context)
  --no-compress      skip LLMLingua-2 (just preselect or baseline)
  --out PATH         output JSONL
  --summary-out PATH summary JSON
  --ops-out PATH     ops log for LLMJudge (optional)
  --no-judge         skip judge

Usage:
  conda activate vsf
  python scripts/phase-02/run_llmlingua_v2.py --n 20
  python scripts/phase-02/run_llmlingua_v2.py --n 50 --k 30 --rate 0.6
  python scripts/phase-02/run_llmlingua_v2.py --n 20 --no-compress --judge-profile openrouter
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")
from _prompts import format_eval_prompt as _format_eval_prompt

# Repo paths
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
BASELINE_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-02-llmlingua-v2.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-02-llmlingua-v2-summary.json"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

ZERO_SCROLLS_TASKS = [
    "qasper", "musique", "gov_report", "space_digest",
    "summ_screen_fd", "qmsum", "squality", "quality", "book_sum_sort",
]

PROMPT_TEMPLATE = None  # canonical prompt lives in scripts/_prompts.py


# ---------------------------------------------------------------
# TF-IDF preselect
# ---------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def preselect_tfidf(context: str, question: str, k: int) -> dict:
    t0 = time.perf_counter()
    if k <= 0:
        return {"status": "ok", "selected_text": context, "selected_chars": len(context),
                "embed_ms": 0, "n_selected": 0}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        sents = split_sentences(context)
        if not sents:
            return {"status": "empty", "selected_text": context, "selected_chars": len(context),
                    "embed_ms": (time.perf_counter() - t0) * 1000, "n_selected": 0}
        if len(sents) <= k:
            return {"status": "ok", "selected_text": context, "selected_chars": len(context),
                    "embed_ms": (time.perf_counter() - t0) * 1000, "n_selected": len(sents)}

        corpus = sents + [question]
        vec = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        q_vec = vec[-1]
        s_vecs = vec[:-1]
        sims = cosine_similarity(q_vec, s_vecs)[0]
        top = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        top.sort()
        selected = " ".join(sents[i] for i in top)
        return {
            "status": "ok",
            "selected_text": selected,
            "selected_chars": len(selected),
            "embed_ms": (time.perf_counter() - t0) * 1000,
            "n_selected": len(top),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc),
                "embed_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------
# LLMLingua-2 compress
# ---------------------------------------------------------------
_llmlingua = None


def get_llmlingua():
    global _llmlingua
    if _llmlingua is None:
        from llmlingua import PromptCompressor
        _llmlingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _llmlingua


def compress_llmlingua2(text: str, question: str, rate: float, iter_size: int) -> dict:
    t0 = time.perf_counter()
    try:
        pc = get_llmlingua()
        prompt_list = [
            "Given the following question and context, answer concisely:",
            f"Question: {question}",
            f"Context: {text}",
        ]
        result = pc.compress_prompt(
            prompt_list,
            rate=rate,
            force_tokens=["\n", ".", "!", "?", ",", "the", "a", "is", "was", "to", "of"],
            chunk_end_tokens=["."],
            iterative_size=iter_size,
            return_word_label=False,
            drop_consecutive=False,
        )
        ms = (time.perf_counter() - t0) * 1000
        compressed = result.get("compressed_prompt", "")
        origin = result.get("origin_tokens", 0)
        comp = result.get("compressed_tokens", 0)
        ratio_raw = result.get("ratio", 0)
        if isinstance(ratio_raw, str):
            ratio = float(ratio_raw.replace("x", "")) if ratio_raw else 0
        else:
            ratio = float(ratio_raw) if ratio_raw else 0
        if ratio == 0 and origin > 0 and comp > 0:
            ratio = origin / comp
        return {
            "status": "ok",
            "compressed_text": compressed,
            "origin_tokens": int(origin),
            "compressed_tokens": int(comp),
            "ratio": float(ratio),
            "compress_ms": ms,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc),
                "compress_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------
def call_gemini(prompt: str, model: str, *, max_retries: int = 4) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}
    url = GEMINI_URL.format(model=model) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    backoff = [30, 60, 120, 180]
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            import requests
            resp = requests.post(url, json=payload, timeout=120)
            elapsed = (time.perf_counter() - t0) * 1000
            sc = resp.status_code
            if sc == 200:
                data = resp.json()
                cands = data.get("candidates", [])
                if not cands:
                    return {"status": "blocked", "latency_ms": elapsed,
                            "error": "no candidates (safety block)"}
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                usage = data.get("usageMetadata", {})
                return {
                    "status": "ok", "text": text,
                    "input_tokens": usage.get("promptTokenCount"),
                    "output_tokens": usage.get("candidatesTokenCount"),
                    "latency_ms": elapsed,
                }
            if sc in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    time.sleep(backoff[min(attempt, len(backoff) - 1)])
                    continue
                return {"status": "error", "error": f"HTTP {sc} after retries",
                        "latency_ms": elapsed}
            return {"status": "error", "error": f"HTTP {sc}: {resp.text[:200]}",
                    "latency_ms": elapsed}
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            return {"status": "timeout", "error": "timeout after retries"}
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            return {"status": "error", "error": str(exc)}
    return {"status": "error", "error": "exhausted retries"}


# ---------------------------------------------------------------
# Judge
# ---------------------------------------------------------------
def judge_case(question: str, gold: str, pred: str, profile: str,
               log_path: Path | None) -> dict:
    from op5.llm import LLMJudge
    judge = LLMJudge(profile=profile, log_path=log_path)
    try:
        result = judge.judge(question=question, gold=gold, pred=pred)
        d = result.to_dict() if hasattr(result, "to_dict") else result
        return {
            "judge_correct": bool(d.get("correct")),
            "judge_verdict": d.get("verdict", ""),
            "judge_reason": str(d.get("reason", ""))[:200],
            "judge_model": d.get("model_used", ""),
            "judge_latency_ms": d.get("latency_ms", 0),
        }
    except Exception as exc:
        return {"judge_correct": None, "judge_reason": f"judge error: {exc}"}


# ---------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------
def load_stratified_sample(n: int, seed: int, tasks: list[str] | None) -> list[dict]:
    """Load paired sample from zero_scrolls_200.jsonl + baseline-en.jsonl."""
    sample_by_id = {}
    if SAMPLE_PATH.exists():
        with SAMPLE_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                sample_by_id[r["id"]] = r
    if not sample_by_id:
        raise RuntimeError(f"Sample not found at {SAMPLE_PATH} — run baseline_200.py first")

    baseline_by_id = {}
    if BASELINE_PATH.exists():
        with BASELINE_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                baseline_by_id[r["case_id"]] = r

    all_cases = []
    for cid, row in sample_by_id.items():
        if tasks and row.get("task") not in tasks:
            continue
        baseline = baseline_by_id.get(cid)
        if not baseline:
            continue
        all_cases.append({
            "case_id": cid, "task": row.get("task"),
            "row": row, "baseline": baseline,
        })

    rng = random.Random(seed)
    by_task = defaultdict(list)
    for c in all_cases:
        by_task[c["task"]].append(c)
    quotas = {t: max(1, round(n * len(v) / len(all_cases)))
              for t, v in by_task.items()}
    diff = n - sum(quotas.values())
    if diff > 0:
        for t in sorted(quotas, key=lambda x: -quotas[x]):
            quotas[t] += 1; diff -= 1
            if diff == 0: break
    elif diff < 0:
        for t in sorted(quotas, key=lambda x: quotas[x]):
            quotas[t] -= 1; diff += 1
            if diff == 0: break

    selected = []
    for task, group in by_task.items():
        q = quotas.get(task, 1)
        if len(group) <= q:
            selected.extend(group); continue
        selected.extend(rng.sample(group, q))
    selected.sort(key=lambda c: c["baseline"].get("approx_input_tokens", 0))
    return selected


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=15,
                   help="TF-IDF top-k; 0 = skip preselect")
    p.add_argument("--rate", type=float, default=0.5,
                   help="LLMLingua rate (token keep ratio)")
    p.add_argument("--iter-size", type=int, default=512)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--judge-profile", default="nim")
    p.add_argument("--tasks", default="")
    p.add_argument("--sleep", type=float, default=3.0)
    p.add_argument("--no-preselect", action="store_true",
                   help="Skip TF-IDF preselect (compress raw context)")
    p.add_argument("--no-compress", action="store_true",
                   help="Skip LLMLingua-2 (just preselect)")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--ops-out", type=Path, default=None)
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or None

    mode_parts = []
    if not args.no_preselect and args.k > 0:
        mode_parts.append(f"TF-IDF k={args.k}")
    if not args.no_compress:
        mode_parts.append(f"LLMLingua-2 rate={args.rate}")
    mode_str = " → ".join(mode_parts) if mode_parts else "(no compression)"

    print(f"=== LLMLingua-2 benchmark on {args.n} cases ===")
    print(f"  Pipeline:    {mode_str}")
    print(f"  Model:       {args.model}")
    print(f"  Judge:       {'(skipped)' if args.no_judge else args.judge_profile}")
    print(f"  Tasks:       {tasks or 'all 9'}")
    print()

    print("[1/4] Loading stratified sample...")
    sample = load_stratified_sample(args.n, args.seed, tasks)
    sample = sample[: args.n]
    print(f"  Selected: {len(sample)} cases")
    by_task = defaultdict(int)
    for c in sample:
        by_task[c["task"]] += 1
    for t in sorted(by_task):
        print(f"    {t:<18}: {by_task[t]}")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    print("[2/4] Running pipeline...")
    all_records = []
    statuses = defaultdict(int)

    for i, case in enumerate(sample):
        row = case["row"]
        baseline = case["baseline"]
        case_id = case["case_id"]
        task = case["task"]

        inp = row.get("input", "")
        ds = int(row.get("document_start_index", 0))
        de = int(row.get("document_end_index", 0))
        qs = int(row.get("query_start_index", 0))
        qe = int(row.get("query_end_index", 0))
        context = inp[ds:de]
        question = inp[qs:qe]
        gold = (row.get("output") or baseline.get("gold", "")).strip()

        # Stage 1: preselect
        ps_ms = 0
        if not args.no_preselect and args.k > 0:
            ps = preselect_tfidf(context, question, args.k)
            if ps["status"] == "error":
                print(f"  [{i+1:>3}/{len(sample)}] {case_id} PRESELECT ERR: {ps.get('error')}")
                statuses["preselect_error"] += 1
                continue
            working = ps["selected_text"]
            ps_ms = float(ps.get("embed_ms", 0))
            n_sel = ps.get("n_selected", 0)
        else:
            working = context
            n_sel = 0

        # Stage 2: compress
        if args.no_compress:
            comp = {"status": "ok", "compressed_text": working,
                    "origin_tokens": 0, "compressed_tokens": 0,
                    "ratio": 1.0, "compress_ms": 0}
        else:
            comp = compress_llmlingua2(working, question, args.rate, args.iter_size)
            if comp["status"] != "ok":
                print(f"  [{i+1:>3}/{len(sample)}] {case_id} COMPRESS ERR: {comp.get('error')}")
                statuses["compress_error"] += 1
                continue

        # Stage 3: LLM
        prompt = _format_eval_prompt(comp["compressed_text"], question)
        llm = call_gemini(prompt, args.model)
        if llm["status"] != "ok":
            print(f"  [{i+1:>3}/{len(sample)}] {case_id} LLM ERR: {llm.get('error')}")
            statuses["llm_error"] += 1
            continue

        # Stage 4: judge
        if args.no_judge:
            jj = {"judge_correct": None, "judge_reason": "(skipped)",
                  "judge_model": "", "judge_latency_ms": 0}
        else:
            jj = judge_case(question, gold, llm["text"], args.judge_profile, args.ops_out)

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_id": case_id,
            "task": task,
            "question": question[:200],
            "gold": gold[:200],
            "pred": llm["text"][:500],
            "status": "ok",
            "mode": "tfidf+llmlingua" if (args.k > 0 and not args.no_preselect and not args.no_compress)
                   else ("llmlingua" if not args.no_compress else "tfidf" if args.k > 0 else "baseline"),
            # Pipeline
            "preselect_sentences": int(n_sel),
            "preselect_ms": round(ps_ms, 1),
            "compress_origin_tokens": int(comp.get("origin_tokens", 0)),
            "compress_tokens": int(comp.get("compressed_tokens", 0)),
            "compress_ratio": round(float(comp.get("ratio", 0)), 2),
            "compress_ms": round(float(comp.get("compress_ms", 0)), 1),
            # LLM
            "llm_model": args.model,
            "llm_latency_ms": float(llm["latency_ms"]),
            "input_tokens": int(llm["input_tokens"]) if llm.get("input_tokens") else None,
            "output_tokens": int(llm["output_tokens"]) if llm.get("output_tokens") else None,
            # Baseline comparison
            "baseline_input_tokens": baseline.get("input_tokens"),
            "baseline_latency_ms": baseline.get("latency_ms"),
            "token_saving": (
                (baseline.get("input_tokens", 0) - llm["input_tokens"]) / max(1, baseline.get("input_tokens", 1))
                if baseline.get("input_tokens") and llm.get("input_tokens") else None
            ),
            # Judge
            **jj,
            # Params
            "k": args.k if args.k > 0 else None,
            "rate": args.rate if not args.no_compress else None,
            "iter_size": args.iter_size if not args.no_compress else None,
            "seed": args.seed,
        }
        all_records.append(record)
        statuses["ok"] += 1

        with args.out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        marker = "OK" if jj.get("judge_correct") else ("X" if jj.get("judge_correct") is False else "—")
        comp_tok = int(comp.get("compressed_tokens", 0))
        ratio = float(comp.get("ratio", 0))
        in_tok = record["input_tokens"]
        save = record.get("token_saving", 0) or 0
        total_ms = ps_ms + float(comp.get("compress_ms", 0)) + float(llm["latency_ms"])
        print(f"  [{i+1:>3}/{len(sample)}] {marker} {case_id:<32} task={task:<14} "
              f"comp={comp_tok:>4}tok({ratio:.1f}x) "
              f"in={in_tok} save={save:.1%} ms={total_ms:.0f}")

        time.sleep(args.sleep)

    # Summary
    print(f"\n[3/4] Statuses: {dict(statuses)}")
    ok_records = [r for r in all_records if r.get("status") == "ok"]
    if not ok_records:
        print("  No successful records — skipping summary")
        return 1

    corr = sum(1 for r in ok_records if r.get("judge_correct"))
    in_toks = [r["input_tokens"] for r in ok_records if r.get("input_tokens")]
    out_toks = [r["output_tokens"] for r in ok_records if r.get("output_tokens")]
    total_ms = [r["preselect_ms"] + r["compress_ms"] + r["llm_latency_ms"] for r in ok_records]
    baseline_in = [r["baseline_input_tokens"] for r in ok_records if r.get("baseline_input_tokens")]
    token_sav = [r["token_saving"] for r in ok_records if r.get("token_saving") is not None]
    ratios = [r["compress_ratio"] for r in ok_records if r.get("compress_ratio")]

    def pctile(lst, p):
        if not lst:
            return None
        s = sorted(lst)
        return s[int(len(s) * p)]

    summary = {
        "total": len(sample),
        "ok": len(ok_records),
        "correct": corr,
        "accuracy": round(corr / len(ok_records), 4) if ok_records else 0,
        "statuses": dict(statuses),
        "model": args.model,
        "judge_profile": None if args.no_judge else args.judge_profile,
        "params": {
            "k": args.k,
            "rate": args.rate,
            "iter_size": args.iter_size,
            "no_preselect": args.no_preselect,
            "no_compress": args.no_compress,
            "seed": args.seed,
            "n": args.n,
        },
        "input_tokens": {
            "median": statistics.median(in_toks) if in_toks else None,
            "p95": pctile(in_toks, 0.95),
            "max": max(in_toks) if in_toks else None,
        },
        "compression_ratio_median": round(statistics.median(ratios), 2) if ratios else None,
        "token_saving_median": round(statistics.median(token_sav), 4) if token_sav else None,
        "latency_ms_total_median": round(statistics.median(total_ms)) if total_ms else None,
        "baseline_comparison": {
            "this_input_median": statistics.median(in_toks) if in_toks else None,
            "baseline_input_median": statistics.median(baseline_in) if baseline_in else None,
            "delta_median": (statistics.median(in_toks) - statistics.median(baseline_in))
                            if (in_toks and baseline_in) else None,
        },
    }
    with args.summary_out.open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # Per-task breakdown
    print(f"\n[4/4] Per-task breakdown:")
    print(f"  {'Task':<18} {'Acc':>8} {'CompTokMed':>11} {'RatioMed':>9} {'InTokMed':>9} {'SaveMed':>9}")
    print("  " + "-" * 70)
    per_task = defaultdict(lambda: {"n": 0, "correct": 0, "comp": [], "ratio": [], "in": [], "save": []})
    for r in ok_records:
        t = r["task"]
        per_task[t]["n"] += 1
        if r["judge_correct"]:
            per_task[t]["correct"] += 1
        per_task[t]["comp"].append(r["compress_tokens"])
        if r.get("compress_ratio"):
            per_task[t]["ratio"].append(r["compress_ratio"])
        if r.get("input_tokens"):
            per_task[t]["in"].append(r["input_tokens"])
        if r.get("token_saving") is not None:
            per_task[t]["save"].append(r["token_saving"])
    for t in sorted(per_task):
        d = per_task[t]
        acc = f"{d['correct']}/{d['n']}"
        cmed = round(statistics.median(d["comp"])) if d["comp"] else 0
        rmed = round(statistics.median(d["ratio"]), 2) if d["ratio"] else 0
        imed = round(statistics.median(d["in"])) if d["in"] else 0
        smed = round(statistics.median(d["save"]), 3) if d["save"] else 0
        print(f"  {t:<18} {acc:>8} {cmed:>11} {rmed:>9} {imed:>9} {smed:>9.3f}")

    print(f"\n  OVERALL: {corr}/{len(ok_records)} correct ({corr/len(ok_records):.1%})")
    if token_sav:
        print(f"  Median token saving vs baseline: {summary['token_saving_median']:.1%}")
    if ratios:
        print(f"  Median compression ratio: {summary['compression_ratio_median']:.2f}x")
    print(f"\nWrote: {args.out}")
    print(f"Wrote: {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
