#!/usr/bin/env python3
"""
LLMLingua-2 + TF-IDF compression test on stratified cases from the 200-case baseline.

Pipeline per case:
  1. TF-IDF preselect (top-k sentences by cosine similarity)
  2. LLMLingua-2 compress (rate=0.5, iterative_size=1024 for 2x faster CPU)
  3. Gemini Flash-Lite call (max_tokens=256, via shared HTTP session)
  4. LLM-as-judge evaluation via unified LLMJudge (NIM profile, falls back gracefully)

Output JSONL + summary paths default to `--out` location. Sample seed=42 matches
the baseline so accuracy can be compared case-by-case.

Usage:
  conda activate vsf
  # TF-IDF only (fastest, baseline comparison)
  python scripts/phase-02/llmlingua_20.py --n 20 --out results/phase-02-tfidf.jsonl
  # TF-IDF + LLMLingua-2 (token savings target: 50-80%)
  python scripts/phase-02/llmlingua_20.py --n 20 --use-llmlingua \\
      --out results/phase-02-tfidf-lingua.jsonl
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
from _http import get_session
from _prompts import format_eval_prompt as _format_eval_prompt

# Constants
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "zero_scrolls_200.jsonl"
BASELINE_PATH = REPO_ROOT / "results" / "phase-02-baseline-200-en.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-02-llmlingua-20.jsonl"

# LLMLingua-2: iterative_size=1024 doubles CPU compression speed vs 512 with
# negligible quality loss on this corpus.
COMPRESS_ITERATIVE_SIZE = 1024

# Tasks from ZeroSCROLLS
ZERO_SCROLLS_TASKS = [
    "qasper", "musique", "gov_report", "space_digest",
    "summ_screen_fd", "qmsum", "squality", "quality", "book_sum_sort",
]

# Module-level session — reused across all Gemini calls in the process.
_SESSION = get_session()


# ---------------------------------------------------------------
# Sentence splitting + TF-IDF preselect
# ---------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split text into sentences (naive split on '.', '!', '?')."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def preselect_tfidf(context: str, question: str, k: int = 15) -> dict:
    """TF-IDF cosine similarity preselect.

    Returns dict with keys: status, selected_text, selected_chars, embed_ms,
    n_selected, n_sentences.
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
                "n_selected": len(sentences),
            }

        corpus = sentences + [question]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        q_vec = tfidf_matrix[-1]
        s_vecs = tfidf_matrix[:-1]
        sims = cosine_similarity(q_vec, s_vecs)[0]

        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        top_idx.sort()  # preserve original document order
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
# LLMLingua-2 compression
# ---------------------------------------------------------------
_llm_lingua = None


def get_llm_lingua():
    """Lazy-init LLMLingua-2 compressor (singleton)."""
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
                        iterative_size: int = COMPRESS_ITERATIVE_SIZE) -> dict:
    """Compress text using LLMLingua-2.

    Returns dict with keys: status, compressed_text, origin_tokens,
    compressed_tokens, compress_ms, ratio.
    """
    t0 = time.perf_counter()
    try:
        lingua = get_llm_lingua()
        prompt_list = [
            "Given the following question and context, answer concisely:",
            f"Question: {question}",
            f"Context: {text}",
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
        ratio_raw = result.get("ratio", 0)
        if isinstance(ratio_raw, str):
            ratio = float(ratio_raw.replace("x", "")) if ratio_raw else 0.0
        else:
            ratio = float(ratio_raw) if ratio_raw else 0.0
        if ratio == 0 and origin > 0 and comp > 0:
            ratio = origin / comp

        return {
            "status": "ok",
            "compressed_text": compressed,
            "origin_tokens": int(origin),
            "compressed_tokens": int(comp),
            "compress_ms": compress_ms,
            "ratio": ratio,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "compress_ms": (time.perf_counter() - t0) * 1000,
        }


# ---------------------------------------------------------------
# Gemini call (shared session + capped retry/backoff)
# ---------------------------------------------------------------
def call_gemini(prompt: str, *, max_retries: int = 3, max_tokens: int = 256) -> dict:
    """Single Gemini REST call using the shared session.

    Uses _http.call_with_retry which caps backoff at 15s instead of 300s.
    """
    from _http import call_with_retry

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "error", "error": "GOOGLE_API_KEY not set"}

    url = GEMINI_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }

    t0 = time.perf_counter()
    try:
        resp = call_with_retry(
            "POST", url, max_retries=max_retries, timeout=120,
            json=payload, session=_SESSION,
        )
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

        return {
            "status": "error",
            "error": f"HTTP {sc}: {resp.text[:200]}",
            "latency_ms": elapsed_ms,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc),
                "latency_ms": (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------
# LLM-as-Judge (unified, profile-driven, with retry)
# ---------------------------------------------------------------
def judge_case(question: str, gold: str, pred: str, profile: str = "nim") -> dict:
    """LLM-as-judge via unified LLMJudge (NIM profile = fastest, most stable)."""
    from op5.llm import LLMJudge
    judge = LLMJudge(profile=profile)
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
# Stratified sample loader (paired by case_id with baseline)
# ---------------------------------------------------------------
def load_stratified_20(seed: int = 42, n: int = 20) -> list[dict]:
    """Load `n` stratified cases from baseline results, paired with full sample row.

    Proportional quota per task (min 1 each), quintile-binned by length.
    """
    sample_by_id: dict[str, dict] = {}
    if SAMPLE_PATH.exists():
        with SAMPLE_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                sample_by_id[r["id"]] = r

    baseline_by_case_id: dict[str, dict] = {}
    with BASELINE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            baseline_by_case_id[r["case_id"]] = r

    all_cases: list[dict] = []
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

    rng = random.Random(seed)
    by_task = defaultdict(list)
    for c in all_cases:
        by_task[c["task"]].append(c)

    total = len(all_cases)
    quotas = {t: max(1, round(n * len(v) / total)) for t, v in by_task.items()}

    # Adjust to exactly `n`
    diff = n - sum(quotas.values())
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
        # Quintile bins by length to span the size distribution
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
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="Number of cases (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=15, help="TF-IDF top-k (recommended 15-20)")
    parser.add_argument("--rate", type=float, default=0.5,
                        help="LLMLingua rate (only used with --use-llmlingua)")
    parser.add_argument("--iter-size", type=int, default=COMPRESS_ITERATIVE_SIZE,
                        help="LLMLingua iterative_size (1024 = ~2x faster)")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Sleep between cases (seconds)")
    parser.add_argument("--judge-profile", default="nim",
                        help="LLMJudge profile: nim (default, fastest) | openrouter | gemini")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSONL path (default: results/phase-02-llmlingua-20.jsonl)")
    parser.add_argument("--use-llmlingua", action="store_true", default=False,
                        help="Enable LLMLingua-2 compression on top of TF-IDF preselect")
    args = parser.parse_args()

    summary_path = args.out.with_suffix(".summary.json")

    mode = "TF-IDF + LLMLingua-2" if args.use_llmlingua else "TF-IDF only"
    print(f"=== {mode} on {args.n} stratified cases ===")
    print(f"  Model:        {GEMINI_MODEL}")
    print(f"  Preselect:    TF-IDF k={args.k}")
    if args.use_llmlingua:
        print(f"  Compress:     LLMLingua-2 rate={args.rate}, iter_size={args.iter_size}")
    print(f"  Judge:        {args.judge_profile}")
    print(f"  Seed:         {args.seed}")
    print(f"  Output:       {args.out}")
    print(f"  Summary:      {summary_path}")
    print()

    # 1. Load stratified sample
    print("[1/4] Loading stratified sample...")
    sample = load_stratified_20(seed=args.seed, n=args.n)
    if len(sample) < args.n:
        print(f"  WARNING: only {len(sample)} cases available, requested {args.n}")
    sample = sample[: args.n]
    print(f"  Loaded {len(sample)} cases")
    by_task = defaultdict(int)
    for r in sample:
        by_task[r["task"]] += 1
    for t in sorted(by_task):
        print(f"    {t:<18}: {by_task[t]}")

    # 2. Run compression pipeline
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    all_records: list[dict] = []
    statuses: dict[str, int] = defaultdict(int)
    total = len(sample)

    print(f"\n[2/4] Running pipeline...")
    for i, rec in enumerate(sample):
        case_id = rec["case_id"]
        task = rec["task"]
        row = rec["row"]
        baseline_rec = rec["baseline"]
        gold = baseline_rec.get("gold", "")

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

        # Step 3: Gemini call (shared session)
        prompt = _format_eval_prompt(cc["compressed_text"], question)
        llm = call_gemini(prompt)
        if llm["status"] != "ok":
            print(f"  [{i+1:>2}/{total}] {case_id} LLM ERROR: {llm.get('error')}")
            statuses["llm_error"] += 1
            continue

        # Step 4: Judge (unified LLMJudge)
        jj = judge_case(question, gold, llm["text"], profile=args.judge_profile)

        # Build record
        ps_ms = float(ps["embed_ms"])
        cc_ms = float(cc.get("compress_ms", 0))
        llm_ms = float(llm["latency_ms"])
        total_ms = ps_ms + cc_ms + llm_ms

        ratio_raw = cc.get("ratio", 0)
        ratio = float(str(ratio_raw).replace("x", "")) if ratio_raw else 0.0

        baseline_in = baseline_rec.get("input_tokens")
        token_sav = None
        if baseline_in and llm.get("input_tokens"):
            token_sav = (baseline_in - llm["input_tokens"]) / max(1, baseline_in)

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_id": case_id,
            "task": task,
            "question": question[:200],
            "gold": gold[:200],
            "pred": llm["text"][:500],
            "status": "ok",
            "mode": "tfidf+llmlingua" if args.use_llmlingua else "tfidf_only",
            "preselect_chars": int(ps["selected_chars"]),
            "preselect_sentences": int(ps.get("n_selected", ps.get("n_sentences", 0))),
            "preselect_ms": round(ps_ms, 1),
            "compress_origin_tokens": int(cc.get("origin_tokens", 0)),
            "compress_tokens": int(cc.get("compressed_tokens", 0)),
            "compress_ratio": ratio,
            "compress_ms": round(cc_ms, 1),
            "total_compress_ms": round(ps_ms + cc_ms, 1),
            "llm_latency_ms": llm_ms,
            "total_ms": round(total_ms, 1),
            "input_tokens": int(llm["input_tokens"]) if llm.get("input_tokens") else None,
            "output_tokens": int(llm["output_tokens"]) if llm.get("output_tokens") else None,
            "baseline_input_tokens": baseline_in,
            "baseline_latency_ms": baseline_rec.get("latency_ms"),
            "token_saving": token_sav,
            **jj,
            "k": args.k,
            "rate": args.rate if args.use_llmlingua else None,
            "iterative_size": args.iter_size if args.use_llmlingua else None,
        }
        all_records.append(record)
        statuses["ok"] += 1

        # Write incrementally
        with args.out.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        marker = "OK" if jj["judge_correct"] else ("X" if jj["judge_correct"] is False else "—")
        comp_tok = int(cc.get("compressed_tokens", 0))
        save_pct = token_sav or 0.0
        print(f"  [{i+1:>2}/{total}] {marker} {case_id:<30} task={task:<14} "
              f"ctx={len(context):>5,} → presel={ps['selected_chars']:>5,} "
              f"→ comp={comp_tok:>4}tok({ratio:.1f}x) "
              f"tok_sav={save_pct:>5.1%} "
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

    in_toks = [r["input_tokens"] for r in ok_records if r.get("input_tokens")]
    out_toks = [r["output_tokens"] for r in ok_records if r.get("output_tokens")]
    comp_ms = [r["compress_ms"] for r in ok_records]
    ps_ms_list = [r["preselect_ms"] for r in ok_records]
    total_ms_list = [r["total_ms"] for r in ok_records]
    llm_ms_list = [r["llm_latency_ms"] for r in ok_records]
    token_sav_list = [r["token_saving"] for r in ok_records if r.get("token_saving") is not None]

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
            "preselect_median": round(statistics.median(ps_ms_list)) if ps_ms_list else None,
            "compress_median": round(statistics.median(comp_ms)) if comp_ms else None,
            "llm_median": round(statistics.median(llm_ms_list)) if llm_ms_list else None,
            "total_median": round(statistics.median(total_ms_list)) if total_ms_list else None,
            "p95": round(pctile(total_ms_list, 0.95)) if total_ms_list else None,
        },
        "token_saving": {
            "median": round(statistics.median(token_sav_list), 4) if token_sav_list else None,
            "min": round(min(token_sav_list), 4) if token_sav_list else None,
            "max": round(max(token_sav_list), 4) if token_sav_list else None,
        },
        "compression": {
            "median_ratio": round(statistics.median([r["compress_ratio"] for r in ok_records]), 2)
            if ok_records else None,
        },
        "baseline_comparison": {
            "this_input_median": statistics.median(in_toks) if in_toks else None,
            "baseline_input_median": statistics.median(baseline_toks) if baseline_toks else None,
            "this_total_median_ms": round(statistics.median(total_ms_list)) if total_ms_list else None,
            "baseline_latency_median_ms": round(statistics.median(baseline_ms)) if baseline_ms else None,
        },
        "params": {
            "k": args.k,
            "rate": args.rate if args.use_llmlingua else None,
            "iterative_size": args.iter_size if args.use_llmlingua else None,
            "model": GEMINI_MODEL,
            "judge_profile": args.judge_profile,
            "seed": args.seed,
            "use_llmlingua": args.use_llmlingua,
        },
    }

    with summary_path.open("w") as fh:
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
    if token_sav_list:
        print(f"  Token saving (median): {summary['token_saving']['median']:.1%}")
    if ok_records:
        print(f"  Compression ratio (median): {summary['compression']['median_ratio']:.2f}x")
    print(f"  Total latency (median): {summary['latency_ms']['total_median']:.0f}ms")
    print(f"\nWrote: {args.out}")
    print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
