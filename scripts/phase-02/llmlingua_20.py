#!/usr/bin/env python3
"""
LLMLingua compression test on 20 cases from the 200-case baseline.

Pipeline per case:
  1. TF-IDF preselect (top-k sentences by cosine similarity)
  2. LLMLingua-2 compress (rate=0.5, iterative_size=512 for speed)
  3. Gemini 3.5 Flash-Lite call (max_tokens=256)
  4. LLM-as-judge evaluation

20 cases stratified by task (same seed=42 as baseline).

Output:
  results/phase-02-llmlingua-20.jsonl

Usage:
  conda activate vsf
  python scripts/phase-02/llmlingua_20.py

Comparison:
  results/phase-02-baseline-200-en.jsonl — baseline (no compression)
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
from _prompts import format_eval_prompt as _format_eval_prompt, _format_gemini_judge_prompt

# Constants
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
BASELINE_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en.jsonl"
OUTPUT_PATH = REPO_ROOT / "results" / "phase-02-llmlingua-20.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "phase-02-llmlingua-20-summary.json"

# TF-IDF preselect config
PRESELECT_K = 10  # top-k sentences

# LLMLingua-2 config (best from Phase 1)
COMPRESS_RATE = 0.5
COMPRESS_ITERATIVE_SIZE = 512  # higher = faster compression on CPU

# Gemini config
MAX_TOKENS = 256
SLEEP = 1.0  # seconds between calls (avoid rate limit)

# Tasks from ZeroSCROLLS
TASKS = [
    "qasper", "musique", "gov_report", "space_digest",
    "summ_screen_fd", "qmsum", "squality", "quality", "book_sum_sort",
]

# ---------------------------------------------------------------
# TF-IDF Preselect
# ---------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split text into sentences (naive split on '.', '!', '?')."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def preselect_tfidf(context: str, question: str, k: int = 10) -> dict:
    """TF-IDF cosine similarity preselect.

    Returns dict with keys: status, selected_text, selected_chars, embed_ms
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    t0 = time.perf_counter()
    try:
        sentences = split_sentences(context)
        if not sentences:
            return {"status": "empty", "error": "No sentences found"}

        if len(sentences) <= k:
            return {
                "status": "ok",
                "selected_text": context,
                "selected_chars": len(context),
                "embed_ms": (time.perf_counter() - t0) * 1000,
                "n_sentences": len(sentences),
            }

        # Build TF-IDF on sentences + question
        corpus = sentences + [question]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Cosine similarity between question (last) and each sentence
        q_vec = tfidf_matrix[-1]
        s_vecs = tfidf_matrix[:-1]
        sims = cosine_similarity(q_vec, s_vecs)[0]

        # Top-k indices
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        top_idx.sort()  # maintain original order

        selected = " ".join(sentences[i] for i in top_idx)
        embed_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": "ok",
            "selected_text": selected,
            "selected_chars": len(selected),
            "embed_ms": embed_ms,
            "n_sentences": len(sentences),
            "n_selected": len(top_idx),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------
# LLMLingua-2 Compress
# ---------------------------------------------------------------
_llm_lingua = None  # singleton

def get_llm_lingua():
    """Lazy-init LLMLingua-2 compressor."""
    global _llm_lingua
    if _llm_lingua is None:
        from llmlingua import PromptCompressor
        _llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _llm_lingua


def compress_llmlingua2(text: str, question: str, rate: float = 0.5,
                        iterative_size: int = 512) -> dict:
    """Compress text using LLMLingua-2.

    Returns dict with keys: status, compressed_text, origin_tokens,
    compressed_tokens, compress_ms, ratio_str
    """
    t0 = time.perf_counter()
    try:
        lingua = get_llm_lingua()
        # Format: [instruction, context]
        prompt_list = [
            "Given the following question and context, answer concisely:",
            f"Question: {question}",
            f"Context: {text}"
        ]
        result = lingua.compress_prompt(
            prompt_list,
            rate=rate,
            force_tokens=["\n", ".", "!", "?", ",", "the", "a", "is", "was", "to", "of"],
            chunk_end_tokens=["."],
            iterative_size=iterative_size,
            return_word_label=False,
            drop_consecutive=False,
        )

        compress_ms = (time.perf_counter() - t0) * 1000
        compressed = result.get("compressed_prompt", "")
        origin = result.get("origin_tokens", 0)
        comp = result.get("compressed_tokens", 0)
        
        # Handle ratio - could be string "Nx" or float
        ratio_raw = result.get("ratio", "")
        if isinstance(ratio_raw, str):
            ratio = float(ratio_raw.replace("x", "")) if ratio_raw else 0
        else:
            ratio = float(ratio_raw) if ratio_raw else 0
        
        # Calculate ratio if not available
        if ratio == 0 and origin > 0 and comp > 0:
            ratio = origin / comp

        return {
            "status": "ok",
            "compressed_text": compressed,
            "origin_tokens": origin,
            "compressed_tokens": comp,
            "compress_ms": compress_ms,
            "ratio": ratio,
            "ratio_str": str(ratio) + "x" if ratio > 0 else "N/A",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "compress_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------
def call_gemini(prompt: str, *, max_retries: int = 4) -> dict:
    """Single Gemini REST call with retry/backoff."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}

    url = GEMINI_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": MAX_TOKENS},
    }

    backoff_schedule = [30, 60, 120, 180, 300]
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            import requests
            resp = requests.post(url, json=payload, timeout=120)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            sc = resp.status_code

            if sc == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return {
                        "status": "blocked", "http_code": 200,
                        "latency_ms": elapsed_ms,
                        "error": "no candidates (safety block)",
                    }
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                usage = data.get("usageMetadata", {})
                return {
                    "status": "ok",
                    "text": text,
                    "input_tokens": usage.get("promptTokenCount"),
                    "output_tokens": usage.get("candidatesTokenCount"),
                    "latency_ms": elapsed_ms,
                }

            if sc in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                    time.sleep(wait)
                    continue
                return {"status": "error", "error": f"HTTP {sc} after retries"}

            return {"status": "error", "error": f"HTTP {sc}: {resp.text[:200]}", "latency_ms": elapsed_ms}

        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(backoff_schedule[min(attempt, len(backoff_schedule) - 1)])
                continue
            return {"status": "timeout", "error": "timeout after retries"}
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(backoff_schedule[min(attempt, len(backoff_schedule) - 1)])
                continue
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": "exhausted retries"}


# ---------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------
def judge_v3(gold: str, pred: str, question: str = "") -> dict:
    """LLM-as-judge using Gemini directly (no OpenRouter dependency).

    Uses canonical Gemini-direct judge prompt from scripts/_prompts.py.
    """
    judge_prompt = _format_gemini_judge_prompt(question, gold, pred)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"judge_correct": None, "judge_reason": "no API key"}

    url = GEMINI_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": judge_prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100},
    }

    try:
        import requests as _req
        resp = _req.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            return {"judge_correct": None, "judge_reason": f"HTTP {resp.status_code}"}
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        try:
            parsed = json.loads(text)
            return {
                "judge_correct": bool(parsed.get("correct")),
                "judge_reason": str(parsed.get("reason", ""))[:200],
            }
        except Exception:
            return {"judge_correct": "true" in text.lower(), "judge_reason": text[:200]}
    except Exception as exc:
        return {"judge_correct": None, "judge_reason": str(exc)}


# ---------------------------------------------------------------
# Load stratified 20 from baseline results
# ---------------------------------------------------------------
def load_stratified_20(seed: int = 42) -> list[dict]:
    """Load 20 cases from baseline results, stratified by task (same seed).
    
    Returns list of sample rows (with full context) matched by case_id.
    """
    # Load sample rows (full data with context)
    sample_by_id = {}
    if SAMPLE_PATH.exists():
        with SAMPLE_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                sample_by_id[r["id"]] = r

    # Load baseline records (for comparison metrics)
    baseline_by_case_id = {}
    with BASELINE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            baseline_by_case_id[r["case_id"]] = r

    # Build cases with metadata
    all_cases = []
    for case_id, baseline_rec in baseline_by_case_id.items():
        sample_row = sample_by_id.get(case_id)
        if not sample_row:
            continue
        task = sample_row.get("task", baseline_rec.get("task"))
        ds = int(sample_row.get("document_start_index", 0))
        de = int(sample_row.get("document_end_index", 0))
        qs = int(sample_row.get("query_start_index", 0))
        qe = int(sample_row.get("query_end_index", 0))
        ctx_chars = de - ds
        q_chars = qe - qs
        approx_tok = (ctx_chars + q_chars + 200) // 4

        all_cases.append({
            "case_id": case_id,
            "task": task,
            "row": sample_row,
            "baseline": baseline_rec,
            "_approx_input_tokens": approx_tok,
            "_ctx_chars": ctx_chars,
            "_question_chars": q_chars,
        })

    # Stratified: proportional quota, min 1 per task
    rng = random.Random(seed)
    n_total = 20
    by_task = defaultdict(list)
    for c in all_cases:
        by_task[c["task"]].append(c)

    total = len(all_cases)
    quotas = {t: max(1, round(n_total * len(v) / total)) for t, v in by_task.items()}

    # Adjust to exactly 20
    diff = n_total - sum(quotas.values())
    if diff > 0:
        for t in sorted(quotas, key=lambda x: -quotas[x]):
            quotas[t] += 1
            diff -= 1
            if diff == 0:
                break
    elif diff < 0:
        for t in sorted(quotas, key=lambda x: quotas[x]):
            quotas[t] -= 1
            diff += 1
            if diff == 0:
                break

    selected = []
    for task, group in by_task.items():
        q = quotas.get(task, 1)
        if len(group) <= q:
            selected.extend(group)
            continue
        # Quintile bins by length
        group_sorted = sorted(group, key=lambda r: r["_approx_input_tokens"])
        bin_size = len(group_sorted) / 5
        bins = [group_sorted[int(i * bin_size):int((i + 1) * bin_size)] for i in range(5)]
        per_bin = [q // 5] * 5
        for i in range(q % 5):
            per_bin[i] += 1
        for b, k in zip(bins, per_bin):
            if len(b) > k > 0:
                chosen = rng.sample(b, k)
            elif len(b) > 0:
                chosen = b[:k]
            else:
                chosen = []
            selected.extend(chosen)

    selected.sort(key=lambda r: r["_approx_input_tokens"])
    return selected


# ---------------------------------------------------------------
# Prompt template (canonical, lives in scripts/_prompts.py)
# ---------------------------------------------------------------
PROMPT_TEMPLATE = None  # kept for backward-compat reference, not used


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of cases (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=15, help="TF-IDF top-k (recommended: 15)")
    parser.add_argument("--rate", type=float, default=0.5, help="LLMLingua rate (only if --use-llmlingua)")
    parser.add_argument("--iter-size", type=int, default=COMPRESS_ITERATIVE_SIZE,
                        help="LLMLingua iterative_size")
    parser.add_argument("--sleep", type=float, default=3.0, help="Sleep between calls")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--use-llmlingua", action="store_true", default=False,
                        help="Enable LLMLingua-2 compression (slower, often worse accuracy)")
    args = parser.parse_args()

    mode = "TF-IDF + LLMLingua-2" if args.use_llmlingua else "TF-IDF only (RECOMMENDED)"
    print(f"=== {mode} on {args.n} stratified cases ===")
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  Preselect: TF-IDF k={args.k}")
    if args.use_llmlingua:
        print(f"  Compress: LLMLingua-2 rate={args.rate}, iter_size={args.iter_size}")
    print(f"  Seed: {args.seed}")
    print()

    # 1. Load stratified sample
    print("[1/4] Loading stratified sample...")
    sample = load_stratified_20(seed=args.seed)
    if len(sample) < args.n:
        print(f"  WARNING: only {len(sample)} cases available, requested {args.n}")
    sample = sample[:args.n]
    print(f"  Loaded {len(sample)} cases")
    by_task = defaultdict(int)
    for r in sample:
        by_task[r["task"]] += 1
    for t in sorted(by_task):
        print(f"    {t:<18}: {by_task[t]}")

    # 2. Run compression pipeline
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    all_records = []
    statuses = defaultdict(int)
    total = len(sample)

    print(f"\n[2/4] Running pipeline...")
    for i, rec in enumerate(sample):
        case_id = rec["case_id"]
        task = rec["task"]
        row = rec["row"]
        baseline_rec = rec["baseline"]
        gold = baseline_rec.get("gold", "")

        # Extract context and question from sample row
        inp = row.get("input", "")
        ds = int(row.get("document_start_index", 0))
        de = int(row.get("document_end_index", 0))
        qs = int(row.get("query_start_index", 0))
        qe = int(row.get("query_end_index", 0))
        context = inp[ds:de]
        question = inp[qs:qe]

        # Step 1: TF-IDF preselect
        ps = preselect_tfidf(context, question, k=args.k)
        if ps["status"] != "ok":
            print(f"  [{i+1:>2}/{total}] {case_id} PRESELECT ERROR: {ps.get('error')}")
            statuses["preselect_error"] += 1
            continue

        # Step 2: LLMLingua-2 compress (optional)
        cc = {"status": "ok", "compressed_text": ps["selected_text"], "origin_tokens": 0,
              "compressed_tokens": 0, "compress_ms": 0, "ratio": 0}
        if args.use_llmlingua:
            cc = compress_llmlingua2(ps["selected_text"], question, rate=args.rate,
                                      iterative_size=args.iter_size)
            if cc["status"] != "ok":
                print(f"  [{i+1:>2}/{total}] {case_id} COMPRESS ERROR: {cc.get('error')}")
                statuses["compress_error"] += 1
                continue

        # Step 3: Gemini call
        prompt = _format_eval_prompt(cc["compressed_text"], question)
        llm = call_gemini(prompt)
        if llm["status"] != "ok":
            print(f"  [{i+1:>2}/{total}] {case_id} LLM ERROR: {llm.get('error')}")
            statuses["llm_error"] += 1
            continue

        # Step 4: Judge
        jj = judge_v3(gold, llm["text"], question)

        # Build record
        total_ms = float(ps["embed_ms"]) + float(cc["compress_ms"]) + float(llm["latency_ms"])
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_id": case_id,
            "task": task,
            "question": question[:200],
            "gold": gold[:200],
            "pred": llm["text"][:500],
            # Pipeline stages
            "status": "ok",
            "mode": "tfidf+llmlingua" if args.use_llmlingua else "tfidf_only",
            "preselect_chars": int(ps["selected_chars"]),
            "preselect_sentences": int(ps.get("n_selected", ps.get("n_sentences", 0))),
            "preselect_ms": round(float(ps["embed_ms"]), 1),
            "compress_origin_tokens": int(cc.get("origin_tokens", 0)),
            "compress_tokens": int(cc.get("compressed_tokens", 0)),
            "compress_ratio": float(str(cc.get("ratio", 0)).replace("x", "")) if cc.get("ratio") else 0,
            "compress_ms": round(float(cc.get("compress_ms", 0)), 1),
            "total_compress_ms": round(float(ps["embed_ms"]) + float(cc.get("compress_ms", 0)), 1),
            # LLM
            "llm_latency_ms": float(llm["latency_ms"]),
            "total_ms": round(total_ms, 1),
            "input_tokens": int(llm["input_tokens"]) if llm.get("input_tokens") else None,
            "output_tokens": int(llm["output_tokens"]) if llm.get("output_tokens") else None,
            # Baseline comparison
            "baseline_input_tokens": baseline_rec.get("input_tokens"),
            "baseline_latency_ms": baseline_rec.get("latency_ms"),
            "token_saving": (
                (baseline_rec.get("input_tokens", 0) - llm["input_tokens"]) / max(1, baseline_rec.get("input_tokens", 1))
                if baseline_rec.get("input_tokens") else None
            ),
            # Judge
            "judge_correct": jj["judge_correct"],
            "judge_reason": jj.get("judge_reason", ""),
            # Params
            "k": args.k,
            "rate": args.rate,
            "iterative_size": args.iter_size,
        }
        all_records.append(record)
        statuses["ok"] += 1

        # Write incrementally
        with args.out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Log
        marker = "OK" if jj["judge_correct"] else "X"
        ctx_chars = len(context)
        comp_tokens = int(cc.get("compressed_tokens", 0))
        comp_ratio_raw = str(cc.get("ratio", "0x"))
        comp_ratio = float(comp_ratio_raw.replace("x", ""))
        print(f"  [{i+1:>2}/{total}] {marker} {case_id:<30} task={task:<14} "
              f"ctx={ctx_chars:>5,} → presel={ps['selected_chars']:>5,} "
              f"→ comp={comp_tokens:>4}tok({comp_ratio:.1f}x) "
              f"tok_sav={record.get('token_saving', 0):>5.1%} "
              f"total={total_ms:>7.0f}ms "
              f"gold={gold[:20]!r}")

        time.sleep(args.sleep)

    # 3. Summary
    print(f"\n[3/4] Computing summary...")
    ok_records = [r for r in all_records if r.get("status") == "ok"]
    if not ok_records:
        print("  No successful records!")
        return 1

    ok = len(ok_records)
    corr = sum(1 for r in ok_records if r.get("judge_correct"))

    # Aggregate
    in_toks = [r["input_tokens"] for r in ok_records if r.get("input_tokens")]
    out_toks = [r["output_tokens"] for r in ok_records if r.get("output_tokens")]
    comp_ms = [r["compress_ms"] for r in ok_records]
    ps_ms = [r["preselect_ms"] for r in ok_records]
    total_ms = [r["total_ms"] for r in ok_records]
    llm_ms = [r["llm_latency_ms"] for r in ok_records]
    token_sav = [r["token_saving"] for r in ok_records if r.get("token_saving") is not None]

    baseline_toks = [r["baseline_input_tokens"] for r in ok_records if r.get("baseline_input_tokens")]
    baseline_ms = [r["baseline_latency_ms"] for r in ok_records if r.get("baseline_latency_ms")]

    def pctile(lst, p):
        if not lst:
            return None
        s = sorted(lst)
        return s[int(len(s) * p)]

    summary = {
        "total": len(all_records),
        "ok": ok,
        "correct": corr,
        "accuracy": round(corr / ok, 4) if ok else 0,
        "statuses": dict(statuses),
        "input_tokens": {
            "n": len(in_toks),
            "median": statistics.median(in_toks) if in_toks else None,
            "p95": pctile(in_toks, 0.95),
            "max": max(in_toks) if in_toks else None,
        },
        "output_tokens": {
            "n": len(out_toks),
            "median": statistics.median(out_toks) if out_toks else None,
        },
        "latency_ms": {
            "preselect_median": round(statistics.median(ps_ms)) if ps_ms else None,
            "compress_median": round(statistics.median(comp_ms)) if comp_ms else None,
            "llm_median": round(statistics.median(llm_ms)) if llm_ms else None,
            "total_median": round(statistics.median(total_ms)) if total_ms else None,
            "p95": round(pctile(total_ms, 0.95)) if total_ms else None,
        },
        "token_saving": {
            "median": round(statistics.median(token_sav), 4) if token_sav else None,
            "min": round(min(token_sav), 4) if token_sav else None,
            "max": round(max(token_sav), 4) if token_sav else None,
        },
        "compression": {
            "median_ratio": round(statistics.median([r["compress_ratio"] for r in ok_records]), 2)
            if ok_records else None,
        },
        "baseline_comparison": {
            "llmlingua_input_median": statistics.median(in_toks) if in_toks else None,
            "baseline_input_median": statistics.median(baseline_toks) if baseline_toks else None,
            "llmlingua_total_median_ms": round(statistics.median(total_ms)) if total_ms else None,
            "baseline_latency_median_ms": round(statistics.median(baseline_ms)) if baseline_ms else None,
        },
        "params": {
            "k": args.k,
            "rate": args.rate,
            "iterative_size": args.iter_size,
            "model": GEMINI_MODEL,
            "seed": args.seed,
        },
    }

    with SUMMARY_PATH.open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # 4. Per-task breakdown
    print(f"\n[4/4] Per-task breakdown:")
    print(f"  {'Task':<18} {'Acc':>8} {'InTokMed':>9} {'CompMsMed':>10} {'TotalMsMed':>10}")
    print("  " + "-" * 60)
    per_task = defaultdict(lambda: {"n": 0, "correct": 0, "in_toks": [], "comp_ms": [], "total_ms": []})
    for r in ok_records:
        t = r["task"]
        per_task[t]["n"] += 1
        if r["judge_correct"]:
            per_task[t]["correct"] += 1
        if r.get("input_tokens"):
            per_task[t]["in_toks"].append(r["input_tokens"])
        per_task[t]["comp_ms"].append(r["compress_ms"])
        per_task[t]["total_ms"].append(r["total_ms"])

    for t in sorted(per_task):
        d = per_task[t]
        acc_str = f"{d['correct']}/{d['n']}"
        in_med = round(statistics.median(d["in_toks"])) if d["in_toks"] else 0
        comp_med = round(statistics.median(d["comp_ms"])) if d["comp_ms"] else 0
        tot_med = round(statistics.median(d["total_ms"])) if d["total_ms"] else 0
        print(f"  {t:<18} {acc_str:>8} {in_med:>9} {comp_med:>10.0f} {tot_med:>10.0f}")

    print(f"\n  OVERALL: {corr}/{ok} correct ({corr/ok:.1%} accuracy)")
    print(f"  Token saving (median): {summary['token_saving']['median']:.1%}")
    print(f"  Compression ratio (median): {summary['compression']['median_ratio']:.2f}x")
    print(f"  Total latency (median): {summary['latency_ms']['total_median']:.0f}ms")
    print(f"\nWrote: {args.out}")
    print(f"Wrote: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
