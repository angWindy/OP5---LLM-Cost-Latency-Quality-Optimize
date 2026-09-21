#!/usr/bin/env python3
"""
LongLLMLingua RAG optimizer — pick 2 stable configs + tune accuracy.

Based on paper "LongLLMLingua" (Jiang et al., ACL 2024):
- Question-aware coarse-grained compression (r_k via conditional PPL)
- Document reordering (lost-in-the-middle mitigation)
- Contrastive Perplexity for fine-grained token scoring
- Dynamic compression ratio per-document
- Subsequence recovery

Per-paper best practice on ZeroSCROLLS / NaturalQuestions:
- Coarse filter to K' most relevant docs (we simulate via sentence-level preselect)
- Compress at rate=0.5 with `condition_compare=True` (contrastive perplexity)
- Reorder so highest-score docs go to start/end

This script runs 2 configs we found most STABLE in our sweep:
  C1: LongLLMLingua paper-style — question-aware compress, rate=0.5, with reorder
  C2: Combo (preselect TF-IDF top-k=10 + LLMLingua-2 rate=0.5) — our PoC winner

Goal: tune (k, rate) over 20+ ZeroSCROLLS cases to maximize accuracy.

Output:
  results/phase-01-longllmlingua-opt-n20.jsonl       (per-case records)
  results/phase-01-longllmlingua-opt-n20-summary.json (aggregated)

Usage:
  conda activate vsf
  python scripts/phase-01/longllmlingua_opt.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
# Use GPT2-small as LongLLMLingua backbone — paper ablation §4.3 shows
# GPT2-small gives ~70.1% (vs 70.8% with LLaMA-2-7B-Chat) at 1/100th the size.
# LLaMA-2-7B is ~14GB download + slow CPU inference → impractical for Phase 1 sweep.
LONGLLM_MODEL = "gpt2"
GEMINI_MODEL = os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite")
MISTRAL_MODEL = os.getenv("OP5_MISTRAL_MODEL", "ministral-8b-latest")
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "zero_scrolls_dev95.jsonl"
FORCE_TOKENS = ["!", ".", "?", "\n"]
SLEEP = 1.0


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------
# Lazy resources
# ---------------------------------------------------------------
_pc = None  # LLMLingua-2 instance
_longllm = None  # LongLLMLingua instance
_gemini = None
_mistral_client = None


def get_pc():
    global _pc
    if _pc is None:
        print("  [init] Loading LLMLingua-2 on CPU...")
        from llmlingua import PromptCompressor
        _pc = PromptCompressor(
            model_name=COMPRESSOR_MODEL,
            use_llmlingua2=True,
            device_map="cpu",
        )
        print("  [init] Compressor ready.")
    return _pc


def get_gemini():
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _gemini = genai.GenerativeModel(GEMINI_MODEL)
        print(f"  [init] Gemini ready: {GEMINI_MODEL}")
    return _gemini


def get_mistral():
    global _mistral_client
    if _mistral_client is None:
        from mistralai.client import Mistral
        _mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        print(f"  [init] Mistral ready: {MISTRAL_MODEL}")
    return _mistral_client


# ---------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    try:
        import nltk
        for name in ("tokenizer/punkt", "tokenizer/punkt_tab"):
            try:
                nltk.data.find(name)
            except LookupError:
                nltk.download(name.split("/")[-1], quiet=True)
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        if len(sents) > 1:
            return sents
    except Exception:
        pass
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    return parts if len(parts) > 1 else [text]


def preselect_tfidf(context: str, question: str, k: int) -> dict:
    """TF-IDF cosine top-k sentence preselect (CPU, ~16ms warm)."""
    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"status": "error", "error": "scikit-learn not installed"}

    sents = split_sentences(context)
    if not sents:
        return {"status": "error", "error": "no sentences"}

    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vec = vec.fit_transform(sents)
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, sent_vec).flatten()
    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    selected = " ".join(sents[i] for i in top_idx)
    return {
        "status": "ok",
        "preselect_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "n_sentences": len(sents),
        "n_kept": k,
        "preselect_chars": len(selected),
        "selected_text": selected,
    }


# ---------------------------------------------------------------
# Compression
# ---------------------------------------------------------------
def compress_llmlingua2(text: str, question: str, rate: float) -> dict:
    """LLMLingua-2 compression (paper-agnostic, fast, ~1.5s/4k chars CPU)."""
    pc = get_pc()
    t0 = time.perf_counter()
    try:
        if len(text) > 12000:  # safety cap (avoids stalls)
            text = text[:12000]
        r = pc.compress_prompt(
            text,
            question=question,
            rate=rate,
            force_tokens=FORCE_TOKENS,
            drop_consecutive=True,
        )
        return {
            "status": "ok",
            "compress_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "compressed_text": r.get("compressed_prompt", ""),
            "origin_tokens": r.get("origin_tokens"),
            "compressed_tokens": r.get("compressed_tokens"),
            "ratio_str": r.get("ratio", ""),
        }
    except Exception as exc:
        return {
            "status": "error",
            "compress_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": f"{type(exc).__name__}: {str(exc)[:150]}",
        }


def compress_longllmlingua(text: str, question: str, rate: float) -> dict:
    """LongLLMLingua paper-style (question-aware, contrastive PPL, reorder, dynamic ratio)."""
    global _longllm
    if _longllm is None:
        print("  [init] Loading LongLLMLingua (microsoft/llmlingua LLaMA-2-7B) on CPU...")
        from llmlingua import PromptCompressor
        _longllm = PromptCompressor(
            model_name=LONGLLM_MODEL,
            device_map="cpu",
        )
        print("  [init] LongLLMLingua ready.")
    t0 = time.perf_counter()
    try:
        # Paper defaults for best quality:
        #   rank_method="longllmlingua"  -> question-aware coarse filter
        #   condition_in_question="after_condition" -> restrict prompt
        #   reorder_context="sort" -> reorder docs by score (lost-in-middle fix)
        #   dynamic_context_compression_ratio=0.3 -> per-doc budget spread
        #   condition_compare=True -> contrastive perplexity
        #   context_budget="+100" -> tolerance
        if len(text) > 6000:  # cap to keep runtime sane
            text = text[:6000]
        r = _longllm.compress_prompt(
            text,
            question=question,
            rate=rate,
            rank_method="longllmlingua",
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
        )
        return {
            "status": "ok",
            "compress_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "compressed_text": r.get("compressed_prompt", ""),
            "origin_tokens": r.get("origin_tokens"),
            "compressed_tokens": r.get("compressed_tokens"),
            "ratio_str": r.get("ratio", ""),
        }
    except Exception as exc:
        return {
            "status": "error",
            "compress_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "error": f"{type(exc).__name__}: {str(exc)[:150]}",
        }


# ---------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------
def build_qa_prompt(context: str, question: str) -> str:
    if question.strip():
        return (
            f"NGU CANH:\n{context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "Tra loi ngan gon (toi da 2-3 tu) dua tren ngu canh. "
            "Neu khong co thong tin, tra 'Khong ro'."
        )
    return (
        f"NGU CANH:\n{context}\n\n"
        "Dua tren ngu canh tren, hay thuc hien yeu cau cu the (tom tat / trich xuat / tinh toan)."
    )


def call_gemini(model, prompt: str, max_tokens: int = 256) -> dict:
    """Single Gemini call. Retries on 429 quota exhaustion (up to 3 times, 10s/20s/30s)."""
    last_err = ""
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            r = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
            )
            text = (r.text or "").strip()
            usage = getattr(r, "usage_metadata", None)
            return {
                "text": text,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                "status": "ok",
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            err_str = f"{type(exc).__name__}: {str(exc)[:150]}"
            last_err = err_str
            if "429" in err_str or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                wait = 10 * (attempt + 1)
                print(f"    [429] sleeping {wait}s (try {attempt+1}/3)...")
                time.sleep(wait)
                continue
            # Non-429 error: don't retry
            return {
                "text": "",
                "latency_ms": elapsed_ms,
                "status": "error",
                "error": err_str,
            }
    return {
        "text": "",
        "latency_ms": 0,
        "status": "error",
        "error": last_err,
    }


def call_mistral(client, prompt: str, max_tokens: int = 256) -> dict:
    t0 = time.perf_counter()
    try:
        r = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = (r.choices[0].message.content or "").strip()
        usage = r.usage
        return {
            "text": text,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "text": "",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:150]}",
        }


# ---------------------------------------------------------------
# Heuristic judge (substring + token F1 + numeric)
# ---------------------------------------------------------------
def judge(gold: str, pred: str) -> dict:
    if not pred or not gold:
        return {"judge_correct": False, "judge_reason": "empty pred or gold"}

    g = gold.strip().lower().rstrip(".,;:!?")
    p = pred.strip().lower().rstrip(".,;:!?")
    if g == p:
        return {"judge_correct": True, "judge_reason": "exact match"}
    if g in p or p in g:
        return {"judge_correct": True, "judge_reason": "substring match"}

    def _tokens(s):
        return set(t for t in re.findall(r"\w+", s.lower()) if len(t) > 1)
    g_t, p_t = _tokens(g), _tokens(p)
    if g_t and p_t:
        common = g_t & p_t
        if common:
            f1 = 2 * len(common) / (len(g_t) + len(p_t))
            if f1 >= 0.5:
                return {"judge_correct": True, "judge_reason": f"token F1={f1:.2f}"}

    g_num, p_num = re.findall(r"\d+\.?\d*", gold), re.findall(r"\d+\.?\d*", pred)
    if g_num and p_num and g_num[0] == p_num[0]:
        return {"judge_correct": True, "judge_reason": "numeric match"}

    if gold.strip().lower() in ("no", "yes", "unanswerable", "none", "khong ro"):
        gp = pred.strip().lower()
        if gold.strip().lower() == "no" and any(w in gp for w in ("khong", "no ", "khong")):
            return {"judge_correct": True, "judge_reason": "no paraphrase"}
        if gold.strip().lower() == "yes" and any(w in gp for w in ("co ", "yes", "co")):
            return {"judge_correct": True, "judge_reason": "yes paraphrase"}
        if gold.strip().lower() == "none" and any(w in gp for w in ("khong ro", "khong co")):
            return {"judge_correct": True, "judge_reason": "none paraphrase"}

    return {"judge_correct": False, "judge_reason": "heuristic miss"}


# ---------------------------------------------------------------
# Dataset loader (ZeroSCROLLS dev95)
# ---------------------------------------------------------------
def load_dev(n: int, seed: int = 42) -> list[dict]:
    decoder = json.JSONDecoder()
    buf = DEFAULT_DEV.read_text()
    pos, rows = 0, []
    while pos < len(buf):
        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        if pos >= len(buf):
            break
        try:
            obj, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            break
        rows.append(obj)
        pos = end

    cleaned = []
    for r in rows:
        inp = r.get("input", "")
        ds = int(r.get("document_start_index", 0))
        de = int(r.get("document_end_index", 0))
        qs = int(r.get("query_start_index", 0))
        qe = int(r.get("query_end_index", 0))
        ctx = inp[ds:de]
        q = inp[qs:qe]
        ans = r.get("output", "")
        if isinstance(ans, list) and ans:
            ans = " ".join(str(x) for x in ans)
        ans = str(ans)
        if not ctx or not ans:
            continue
        cleaned.append({
            "_id": r.get("id", ""),
            "task": r.get("task", ""),
            "context": ctx,
            "question": q,
            "answer": ans,
        })
    print(f"  [load] {len(cleaned)} cases (from {len(rows)} raw)")

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


# ---------------------------------------------------------------
# Two STABLE configs (paper-backed)
# ---------------------------------------------------------------
def run_config(cfg_name: str, row: dict, k: int, rate: float) -> dict:
    """Run one of the 2 stable configs on a single case."""
    context = row["context"]
    question = row["question"]
    gold = row["answer"]

    # Step 1: TF-IDF preselect (always, to bound input size for the compressor)
    ps = preselect_tfidf(context, question, k=k)
    if ps["status"] != "ok":
        return {"status": "preselect_error", "error": ps.get("error", "")}

    # Step 2: compression — paper-style vs combo
    if cfg_name == "C1_longllmlingua_paper":
        cc = compress_longllmlingua(ps["selected_text"], question, rate=rate)
    else:  # C2_combo_poc
        cc = compress_llmlingua2(ps["selected_text"], question, rate=rate)
    if cc["status"] != "ok":
        return {"status": "compress_error", "error": cc.get("error", "")}

    # Step 3: Gemini call
    prompt = build_qa_prompt(cc["compressed_text"], question)
    g = get_gemini()
    ll = call_gemini(g, prompt)
    if ll["status"] != "ok":
        return {"status": "llm_error", "error": ll.get("error", "")}

    # Step 4: judge
    jj = judge(gold, ll["text"])

    return {
        "status": "ok",
        "config": cfg_name,
        "k": k,
        "rate": rate,
        "ps_ms": ps["preselect_ms"],
        "ps_chars": ps["preselect_chars"],
        "comp_ms": cc["compress_ms"],
        "comp_tok_in": cc.get("origin_tokens"),
        "comp_tok_out": cc.get("compressed_tokens"),
        "comp_ratio": cc.get("ratio_str", ""),
        "llm_ms": ll["latency_ms"],
        "input_tokens": ll.get("input_tokens"),
        "total_ms": round(ps["preselect_ms"] + cc["compress_ms"] + ll["latency_ms"], 1),
        "pred": ll["text"][:200],
        "gold": gold[:200],
        "judge_correct": jj["judge_correct"],
        "judge_reason": jj["judge_reason"],
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Cases per config")
    parser.add_argument("--k", type=int, default=10, help="Preselect top-k")
    parser.add_argument("--rate", type=float, default=0.5, help="Compress rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--configs", type=str, default="both",
        help="C1_longllmlingua_paper, C2_combo_poc, or 'both'",
    )
    args = parser.parse_args()

    out_path = args.out or (REPO_ROOT / "results" / f"phase-01-longllmlingua-opt-n{args.n}.jsonl")
    summary_path = args.summary or (REPO_ROOT / "results" / f"phase-01-longllmlingua-opt-n{args.n}-summary.json")

    CONFIGS = ["C1_longllmlingua_paper", "C2_combo_poc"]
    if args.configs != "both":
        CONFIGS = [c.strip() for c in args.configs.split(",")]

    print("=== LongLLMLingua RAG optimizer - 2 stable configs ===")
    print(f"  model: {GEMINI_MODEL}")
    print(f"  cases: {args.n} (k={args.k}, rate={args.rate}, seed={args.seed})")
    print(f"  configs: {CONFIGS}")

    rows = load_dev(args.n, seed=args.seed)
    print(f"  [loaded] {len(rows)} cases")
    for r in rows[:3]:
        print(f"    {r['_id'][:8]} task={r['task']:<14s} ctx={len(r['context']):>5,}c gold={r['answer'][:30]!r}")
    if len(rows) > 3:
        print(f"    ... and {len(rows)-3} more")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    all_records = []
    with out_path.open("w", encoding="utf-8") as fh:
        for cfg_name in CONFIGS:
            print(f"\n=== CONFIG: {cfg_name} (k={args.k}, rate={args.rate}) ===")
            cfg_records = []
            for i, row in enumerate(rows):
                case_id = f"L{i:02d}-{row['_id'][:8]}"
                t0 = time.perf_counter()
                rec = run_config(cfg_name, row, k=args.k, rate=args.rate)
                rec["ts"] = now_iso()
                rec["case_id"] = case_id
                rec["task"] = row.get("task", "?")
                rec["ctx_chars"] = len(row["context"])
                cfg_records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

                if rec.get("status") == "ok":
                    marker = "OK" if rec["judge_correct"] else "X"
                    print(f"  [{case_id}] task={row['task'][:10]:10s} "
                          f"ctx={len(row['context']):>5,} ps={rec['ps_chars']:>5,} "
                          f"comp={rec['comp_ms']:>5.0f}ms llm={rec['llm_ms']:>4.0f}ms "
                          f"gold={rec['gold'][:25]:<25s} -> {marker}")
                else:
                    print(f"  [{case_id}] ERROR: {rec.get('error', rec.get('status',''))[:80]}")
                time.sleep(SLEEP)
                _ = (time.perf_counter() - t0)

            all_records.extend(cfg_records)
            ok = sum(1 for r in cfg_records if r.get("status") == "ok")
            corr = sum(1 for r in cfg_records if r.get("judge_correct"))
            print(f"  -> {cfg_name}: {corr}/{ok} correct")

    summary = {"ts": now_iso(), "n_cases": len(rows), "k": args.k, "rate": args.rate, "configs": {}}
    for cfg_name in CONFIGS:
        recs = [r for r in all_records if r.get("config") == cfg_name and r.get("status") == "ok"]
        if not recs:
            continue
        toks = [r["input_tokens"] for r in recs if r.get("input_tokens")]
        comp_ms = [r["comp_ms"] for r in recs if r.get("comp_ms")]
        ps_ms = [r["ps_ms"] for r in recs]
        llm_ms = [r["llm_ms"] for r in recs]
        total_ms = [r["total_ms"] for r in recs]
        correct = sum(1 for r in recs if r["judge_correct"])

        per_task = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in recs:
            per_task[r["task"]]["n"] += 1
            if r["judge_correct"]:
                per_task[r["task"]]["correct"] += 1

        summary["configs"][cfg_name] = {
            "n_ok": len(recs),
            "accuracy": round(correct / len(recs), 3) if recs else None,
            "correct_count": correct,
            "avg_input_tokens": round(statistics.mean(toks), 1) if toks else None,
            "avg_ps_ms": round(statistics.mean(ps_ms), 1) if ps_ms else None,
            "avg_comp_ms": round(statistics.mean(comp_ms), 1) if comp_ms else None,
            "avg_llm_ms": round(statistics.mean(llm_ms), 1) if llm_ms else None,
            "avg_total_ms": round(statistics.mean(total_ms), 1) if total_ms else None,
            "per_task": {t: {"n": d["n"], "correct": d["correct"],
                              "acc": round(d["correct"]/d["n"], 3)}
                         for t, d in per_task.items()},
        }

    with summary_path.open("w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" SUMMARY - LongLLMLingua RAG optimizer (2 stable configs)")
    print("=" * 100)
    print(f"  {'Config':<28} {'N':>3} {'Acc':>9} {'InTok':>8} {'PsMs':>6} {'CompMs':>7} {'LlmMs':>6} {'TotMs':>7}")
    print("  " + "-" * 90)
    for cfg_name in CONFIGS:
        c = summary["configs"].get(cfg_name, {})
        if not c:
            print(f"  {cfg_name:<28}  no records")
            continue
        print(f"  {cfg_name:<28} {c['n_ok']:>3} "
              f"{c['correct_count']}/{c['n_ok']:>5} ({c['accuracy']*100:>4.1f}%) "
              f"{c['avg_input_tokens'] or 0:>8.0f} "
              f"{c['avg_ps_ms'] or 0:>6.0f} "
              f"{c['avg_comp_ms'] or 0:>7.0f} "
              f"{c['avg_llm_ms'] or 0:>6.0f} "
              f"{c['avg_total_ms'] or 0:>7.0f}")

    all_tasks = set()
    for c in summary["configs"].values():
        all_tasks.update(c.get("per_task", {}).keys())
    if all_tasks:
        print(f"\n  Per-task accuracy")
        print(f"  {'Config':<28} " + " ".join(f"{t[:9]:>9}" for t in sorted(all_tasks)))
        print("  " + "-" * (28 + 10 * len(all_tasks)))
        for cfg_name in CONFIGS:
            c = summary["configs"].get(cfg_name, {})
            row_str = f"  {cfg_name:<28} "
            for t in sorted(all_tasks):
                tc = c.get("per_task", {}).get(t, {})
                if tc.get("n"):
                    row_str += f"{tc['correct']}/{tc['n']:<6}".rjust(10)
                else:
                    row_str += f"{'-':>9}"
            print(row_str)

    print(f"\nWrote: {out_path}")
    print(f"       {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
