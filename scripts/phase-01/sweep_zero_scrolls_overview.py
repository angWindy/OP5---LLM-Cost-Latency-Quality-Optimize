#!/usr/bin/env python3
"""
Sweep tổng hợp trên ZeroSCROLLS dev95 — chạy NHIỀU config trên CÙNG subset
để có cái nhìn tổng quan về combo preselect + compress.

Configs:
  1. Baseline        — full context -> Gemini (floor)
  2. Compress-only   — full -> LLMLingua-2 r=0.5 -> Gemini
  3. Preselect-only  — semantic k=20 -> Gemini (no compress)
  4. Combo (sem)     — semantic k=20 -> LLMLingua-2 r=0.5 -> Gemini (winner hiện tại)
  5. Combo (hybrid)  — hybrid k=20 -> LLMLingua-2 r=0.5 -> Gemini
  6. Combo + Mistral — semantic k=20 -> LLMLingua-2 r=0.5 -> Mistral
  7. Aggressive combo — semantic k=20 -> LLMLingua-2 r=0.3 -> Gemini

Output:
  results/phase-01-sweep-zero-scrolls-{n}.jsonl         (per-case rows)
  results/phase-01-sweep-zero-scrolls-{n}-summary.json  (aggregate per-config)

Usage:
  conda activate vsf
  python scripts/phase-01/sweep_zero_scrolls_overview.py [--n 20]
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
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRESELECT_K = 20
COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
FORCE_TOKENS = ["!", ".", "?", "\n"]
GEMINI_MODEL = os.getenv("OP5_GEMINI_MODEL", "gemini-3.1-flash-lite")
MISTRAL_MODEL = os.getenv("OP5_MISTRAL_MODEL", "ministral-8b-latest")
JUDGE_MODEL = os.getenv("OP5_JUDGE_MODEL", "gemini-3.1-flash-lite")
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "zero_scrolls_dev95.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------
# Lazy-loaded resources
# ---------------------------------------------------------------
_pc = None
_st_model = None
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


def get_st():
    global _st_model
    if _st_model is None:
        print("  [init] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [init] ST ready.")
    return _st_model


def get_gemini(model_name: str | None = None):
    """Get Gemini model — defaults to GEMINI_MODEL (3.1-flash-lite)."""
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY missing")
        genai.configure(api_key=key)
        mname = model_name or GEMINI_MODEL
        _gemini = genai.GenerativeModel(mname)
        print(f"  [init] Gemini ready: {mname}")
    return _gemini


def get_mistral():
    global _mistral_client
    if _mistral_client is None:
        from mistralai.client import Mistral
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY missing")
        _mistral_client = Mistral(api_key=key)
    return _mistral_client


# ---------------------------------------------------------------
# Sentences + preselect
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


def _cosine(a, b):
    import math
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma == 0 or mb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (ma * mb)


def preselect_semantic(context: str, question: str, k: int) -> dict:
    t0 = time.perf_counter()
    sents = split_sentences(context)
    if not sents:
        return {"status": "error", "error": "no sentences"}
    model = get_st()
    sent_embs = model.encode(sents, batch_size=64, show_progress_bar=False)
    q_emb = model.encode([question], show_progress_bar=False)[0]
    scores = sorted(
        [(_cosine(se.tolist(), q_emb.tolist()), s) for se, s in zip(sent_embs, sents)],
        reverse=True,
    )
    selected = " ".join(s for _, s in scores[:k])
    return {
        "status": "ok",
        "preselect_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "n_sentences": len(sents),
        "n_kept": k,
        "preselect_chars": len(selected),
        "selected_text": selected,
    }


def preselect_hybrid(context: str, question: str, k: int) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    t0 = time.perf_counter()
    sents = split_sentences(context)
    if not sents:
        return {"status": "error", "error": "no sentences"}

    # BM25 (TF-IDF)
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vec = vec.fit_transform(sents)
    q_vec = vec.transform([question])
    tfidf = cosine_similarity(q_vec, sent_vec).flatten()
    tf_max = tfidf.max() if tfidf.max() > 0 else 1.0
    tfidf_norm = tfidf / tf_max

    # Semantic
    model = get_st()
    sent_embs = model.encode(sents, batch_size=64, show_progress_bar=False)
    q_emb = model.encode([question], show_progress_bar=False)[0]
    sem = [max(_cosine(se.tolist(), q_emb.tolist()), 0) for se in sent_embs]
    sem_max = max(sem) if max(sem) > 0 else 1.0
    sem_norm = [s / sem_max for s in sem]

    # 50/50 ensemble
    ensemble = [0.5 * tfidf_norm[i] + 0.5 * sem_norm[i] for i in range(len(sents))]
    top_idx = sorted(range(len(ensemble)), key=lambda i: ensemble[i], reverse=True)[:k]
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
# Compress + LLM call
# ---------------------------------------------------------------
def compress(pc, text: str, rate: float) -> dict:
    t0 = time.perf_counter()
    try:
        # Cap input to avoid stalls (D-12 issue)
        if len(text) > 12000:
            text = text[:12000]
        r = pc.compress_prompt(text, rate=rate, force_tokens=FORCE_TOKENS, drop_consecutive=True)
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


def call_gemini(model, prompt: str, max_tokens: int = 256) -> dict:
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
        return {
            "text": "",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:150]}",
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
# Scoring: judge bằng Gemini (LLM-as-judge) + free-form F1 proxy
# ---------------------------------------------------------------
def build_qa_prompt(context: str, question: str) -> str:
    if question.strip():
        return (
            f"NGU CANH:\n{context}\n\n"
            f"CAU HOI:\n{question}\n\n"
            "Tra loi ngan gon (toi da 2-3 tu) dua tren ngu canh. "
            "Neu khong co thong tin, tra 'Khong ro'."
        )
    else:
        # No question — task is summarization or scoring
        return (
            f"NGU CANH:\n{context}\n\n"
            "Dua tren ngu canh tren, hay thuc hien yeu cau cu the (tom tat / trich xuat / tinh toan)."
        )


def judge(model, question: str, gold: str, pred: str, use_mistral: bool = False) -> dict:
    """Judge pred vs gold. Try (1) substring match (2) token-overlap F1 (3) LLM fallback.

    For ZeroSCROLLS free-form answers, substring + token-overlap is usually enough.
    LLM judge is fallback only.
    """
    if not pred or not gold:
        return {"judge_correct": False, "judge_reason": "empty pred or gold"}

    # (1) Exact / substring match (case-insensitive, ignore trailing punctuation)
    g = gold.strip().lower().rstrip(".,;:!?")
    p = pred.strip().lower().rstrip(".,;:!?")
    if g == p:
        return {"judge_correct": True, "judge_reason": "exact match"}
    if g in p or p in g:
        return {"judge_correct": True, "judge_reason": "substring match"}

    # (2) Token F1 (handles partial answers like "Ba Lan" vs "Poland")
    def _tokens(s: str) -> set[str]:
        return set(t for t in re.findall(r"\w+", s.lower()) if len(t) > 1)
    g_t = _tokens(g)
    p_t = _tokens(p)
    if g_t and p_t:
        common = g_t & p_t
        if common:
            f1 = 2 * len(common) / (len(g_t) + len(p_t))
            if f1 >= 0.5:
                return {"judge_correct": True, "judge_reason": f"token F1={f1:.2f}"}

    # (3) Numeric answer (e.g. "70%" vs "70 percent")
    g_num = re.findall(r"\d+\.?\d*", gold)
    p_num = re.findall(r"\d+\.?\d*", pred)
    if g_num and p_num and g_num[0] == p_num[0]:
        return {"judge_correct": True, "judge_reason": "numeric match"}

    # (4) Special: gold == "No" / "Yes" / "unanswerable" / "None"
    if gold.strip().lower() in ("no", "yes", "unanswerable", "none", "khong ro"):
        gp = pred.strip().lower()
        if gold.strip().lower() == "no" and any(w in gp for w in ("khong", "no ", "không")):
            return {"judge_correct": True, "judge_reason": "no paraphrase"}
        if gold.strip().lower() == "yes" and any(w in gp for w in ("co ", "yes", "có")):
            return {"judge_correct": True, "judge_reason": "yes paraphrase"}
        if gold.strip().lower() == "none" and any(w in gp for w in ("khong ro", "không rõ", "none", "khong co")):
            return {"judge_correct": True, "judge_reason": "none paraphrase"}

    # Heuristic failed → judge as wrong (saves 429 calls). User can re-run with --use_llm_judge.
    return {"judge_correct": False, "judge_reason": "heuristic miss"}


# ---------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------
def load_dev(n: int, seed: int = 42) -> list[dict]:
    decoder = json.JSONDecoder()
    rows = []
    with DEFAULT_DEV.open() as f:
        buf = f.read()
    pos = 0
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

    # Extract context + question via ZeroSCROLLS schema
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
            "document_start_index": ds,
            "document_end_index": de,
        })
    print(f"  [load] {len(cleaned)} cases (from {len(rows)} raw)")

    # Stratified sample by task
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in cleaned:
        by_task[r["task"]].append(r)
    rng = random.Random(seed)
    selected = []
    for task, group in by_task.items():
        rng.shuffle(group)
        per = max(1, n // len(by_task))
        selected.extend(group[:per])
    rng.shuffle(selected)
    selected = selected[:n]
    # Sort smallest first to finish early subset faster
    selected.sort(key=lambda r: len(r["context"]))
    return selected


# ---------------------------------------------------------------
# Config runner
# ---------------------------------------------------------------
def run_config(cfg: dict, rows: list[dict], out_fh) -> list[dict]:
    """Run one config over all rows. Returns list of per-case records."""
    print(f"\n=== CONFIG: {cfg['name']} ===")
    print(f"  preselect={cfg.get('preselect','none')}  compress_r={cfg.get('rate','-')}  llm={cfg['llm']}")

    pc = get_pc() if cfg.get("rate") else None
    g31 = get_gemini()  # gemini-3.1-flash-lite (main)
    m = get_mistral() if cfg["llm"] == "mistral" else None
    if cfg["llm"] == "gemini31":
        model = g31
    elif cfg["llm"] == "mistral":
        model = m
    else:
        model = g31

    ps_fn = None
    if cfg.get("preselect") == "semantic":
        ps_fn = preselect_semantic
    elif cfg.get("preselect") == "hybrid":
        ps_fn = preselect_hybrid

    records = []
    for i, row in enumerate(rows):
        case_id = f"S-{i:02d}-{row.get('_id','')[:6]}"
        context = row["context"]
        question = row["question"]
        gold = row["answer"]

        # Step 1: preselect (optional)
        if ps_fn:
            ps = ps_fn(context, question, k=PRESELECT_K)
            if ps["status"] != "ok":
                print(f"  [{case_id}] preselect error: {ps.get('error','')[:80]}")
                continue
            input_text = ps["selected_text"]
            ps_ms = ps["preselect_ms"]
            ps_chars = ps["preselect_chars"]
        else:
            input_text = context
            ps_ms = 0.0
            ps_chars = len(context)

        # Step 2: compress (optional)
        if pc and cfg.get("rate"):
            cc = compress(pc, input_text, rate=cfg["rate"])
            if cc["status"] != "ok":
                print(f"  [{case_id}] compress error: {cc.get('error','')[:80]}")
                continue
            final_input = cc["compressed_text"]
            comp_ms = cc["compress_ms"]
            comp_ratio = cc.get("ratio_str", "")
            comp_tok_in = cc.get("origin_tokens")
            comp_tok_out = cc.get("compressed_tokens")
        else:
            final_input = input_text
            comp_ms = 0.0
            comp_ratio = ""
            comp_tok_in = None
            comp_tok_out = None

        # Step 3: build prompt + call LLM (with retry on 429)
        prompt = build_qa_prompt(final_input, question)
        ll = None
        for attempt in range(3):
            if cfg["llm"] == "mistral":
                ll = call_mistral(model, prompt)
            else:  # gemini31
                ll = call_gemini(model, prompt)
            if ll["status"] == "ok":
                break
            if "429" in (ll.get("error") or "") or "quota" in (ll.get("error") or "").lower():
                wait = 10 * (attempt + 1)
                print(f"  [{case_id}] {cfg['llm']} 429, sleeping {wait}s (try {attempt+1}/3)...")
                time.sleep(wait)
            else:
                break  # non-429 error, don't retry
        if cfg["llm"] == "mistral":
            time.sleep(1.5)
        else:
            time.sleep(0.5)

        if ll["status"] != "ok":
            print(f"  [{case_id}] LLM error: {ll.get('error','')[:80]}")
            continue

        # Step 4: judge (Gemini 3.1 nhanh, dùng làm judge để có signal nhanh)
        jj = judge(g31, question, gold, ll["text"], use_mistral=False)

        rec = {
            "ts": now_iso(),
            "config": cfg["name"],
            "case_id": case_id,
            "task": row.get("task", "?"),
            "ctx_chars": len(context),
            "ps_ms": ps_ms,
            "ps_chars": ps_chars,
            "comp_ms": comp_ms,
            "comp_ratio": comp_ratio,
            "comp_tok_in": comp_tok_in,
            "comp_tok_out": comp_tok_out,
            "llm": cfg["llm"],
            "input_tokens": ll.get("input_tokens"),
            "output_tokens": ll.get("output_tokens"),
            "latency_ms": ll["latency_ms"],
            "total_ms": round(ps_ms + comp_ms + ll["latency_ms"], 1),
            "pred": ll["text"][:200],
            "gold": gold[:200],
            "judge_correct": jj["judge_correct"],
        }
        records.append(rec)
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()

        marker = "OK" if jj["judge_correct"] else "X"
        if jj["judge_correct"] is None:
            marker = "?"
        print(f"  [{case_id}] task={row.get('task','?')[:10]:10s} "
              f"ctx={len(context):>5,} ps={ps_chars:>5,} comp={comp_ms:>5.0f}ms "
              f"llm={ll['latency_ms']:>4.0f}ms gold={gold[:25]:<25s} -> {marker}")

    return records


def summarize(by_cfg: dict) -> dict:
    out = {"ts": now_iso(), "configs": {}}
    for name, recs in by_cfg.items():
        n = len(recs)
        if n == 0:
            out["configs"][name] = {"n": 0}
            continue
        correct = sum(1 for r in recs if r["judge_correct"])
        judged = sum(1 for r in recs if r["judge_correct"] is not None)
        toks = [r["input_tokens"] for r in recs if r.get("input_tokens")]
        lats = [r["total_ms"] for r in recs]
        comp_ms = [r["comp_ms"] for r in recs if r["comp_ms"]]
        ps_ms = [r["ps_ms"] for r in recs if r["ps_ms"]]
        llm_lats = [r["latency_ms"] for r in recs]

        # Per-task
        per_task = {}
        for r in recs:
            t = r.get("task", "?")
            per_task.setdefault(t, []).append(r)
        per_task_summary = {}
        for t, trs in sorted(per_task.items()):
            tc = sum(1 for r in trs if r["judge_correct"])
            per_task_summary[t] = {
                "n": len(trs),
                "correct": tc,
                "acc": round(tc / len(trs), 3) if trs else None,
            }

        out["configs"][name] = {
            "n": n,
            "judged": judged,
            "correct": correct,
            "accuracy": round(correct / judged, 3) if judged else None,
            "avg_input_tokens": round(statistics.mean(toks), 1) if toks else None,
            "avg_total_ms": round(statistics.mean(lats), 1) if lats else None,
            "avg_ps_ms": round(statistics.mean(ps_ms), 1) if ps_ms else None,
            "avg_compress_ms": round(statistics.mean(comp_ms), 1) if comp_ms else None,
            "avg_llm_latency_ms": round(statistics.mean(llm_lats), 1) if llm_lats else None,
            "per_task": per_task_summary,
        }
    return out


def print_summary(summary: dict):
    print("\n" + "=" * 100)
    print(" OVERVIEW — ZeroSCROLLS dev95 sweep")
    print("=" * 100)
    print(f"  {'Config':<26} {'N':>4} {'Acc':>8} {'InputTok':>10} {'TotalMs':>10} {'PsMs':>8} {'CompMs':>8} {'LlmMs':>8}")
    print("  " + "-" * 96)
    for name, c in summary["configs"].items():
        if c.get("n", 0) == 0:
            print(f"  {name:<26}  0  (no cases)")
            continue
        acc_str = f"{c['correct']}/{c['judged']}"
        print(f"  {name:<26} {c['n']:>4} "
              f"{acc_str:>8} "
              f"{c.get('avg_input_tokens') or 0:>10.0f} "
              f"{c.get('avg_total_ms') or 0:>10.0f} "
              f"{c.get('avg_ps_ms') or 0:>8.0f} "
              f"{c.get('avg_compress_ms') or 0:>8.0f} "
              f"{c.get('avg_llm_latency_ms') or 0:>8.0f}")

    # Per-task accuracy
    tasks_seen = set()
    for cfg in summary["configs"].values():
        tasks_seen.update(cfg.get("per_task", {}).keys())
    tasks_seen = sorted(tasks_seen)
    if tasks_seen:
        print(f"\n  Per-task accuracy (correct/n)")
        print(f"  {'Config':<26} " + " ".join(f"{t[:8]:>8}" for t in tasks_seen))
        print("  " + "-" * (26 + 9 * len(tasks_seen)))
        for name, c in summary["configs"].items():
            row = f"  {name:<26} "
            for t in tasks_seen:
                tc = c.get("per_task", {}).get(t, {})
                row += f" {tc.get('correct',0)}/{tc.get('n',0):<6}"[:8].rjust(8)
            print(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Cases per config (sub-sampled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--configs", type=str, default="all",
        help="Comma-separated config names; default 'all'",
    )
    args = parser.parse_args()

    n = args.n
    seed = args.seed
    out_path = args.out or (REPO_ROOT / "results" / f"phase-01-sweep-zero-scrolls-n{n}.jsonl")
    summary_path = args.summary or (REPO_ROOT / "results" / f"phase-01-sweep-zero-scrolls-n{n}-summary.json")

    ALL_CONFIGS = [
        {"name": "baseline_g31",         "preselect": None,       "rate": None, "llm": "gemini31"},
        {"name": "compress_r0.5_g31",    "preselect": None,       "rate": 0.5,  "llm": "gemini31"},
        {"name": "preselect_sem_g31",    "preselect": "semantic", "rate": None, "llm": "gemini31"},
        {"name": "combo_sem_r0.5_g31",   "preselect": "semantic", "rate": 0.5,  "llm": "gemini31"},
        {"name": "combo_hybrid_r0.5",    "preselect": "hybrid",   "rate": 0.5,  "llm": "gemini31"},
        {"name": "combo_sem_r0.3_g31",   "preselect": "semantic", "rate": 0.3,  "llm": "gemini31"},
        {"name": "combo_sem_r0.7_g31",   "preselect": "semantic", "rate": 0.7,  "llm": "gemini31"},
        {"name": "combo_sem_r0.5_mistral","preselect": "semantic","rate": 0.5,  "llm": "mistral"},
    ]

    if args.configs != "all":
        wanted = set(c.strip() for c in args.configs.split(","))
        ALL_CONFIGS = [c for c in ALL_CONFIGS if c["name"] in wanted]

    rows = load_dev(n, seed=seed)
    print(f"\n  Running on {len(rows)} cases (seed={seed}):")
    for r in rows[:5]:
        print(f"    {r.get('_id','')[:8]} task={r.get('task',''):<18s} ctx={len(r['context']):>6,}c "
              f"gold={r['answer'][:40]!r}")
    if len(rows) > 5:
        print(f"    ... and {len(rows)-5} more (sorted smallest first)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    by_cfg = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for cfg in ALL_CONFIGS:
            recs = run_config(cfg, rows, fh)
            by_cfg[cfg["name"]] = recs
            print(f"  -> {cfg['name']}: {sum(1 for r in recs if r['judge_correct'])}/{len(recs)} correct")

    summary = summarize(by_cfg)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nWrote: {out_path}")
    print(f"       {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
