#!/usr/bin/env python3
"""
Resurrect of the aborted 2-compressor comparison (task 17901/17902):
  Compare Mistral vs Gemini on the SAME compressed prompt.

Why this comparison?
  - All previous runs (eval_combo_*, eval_mistral_combo_3) used ONE model only.
  - The aborted script tried to compare "LongLLMLingua vs LLMLingua-2" but those
    are sibling variants; not really orthogonal. Model choice is more actionable.
  - Skip the OCR-review LLM call (that was the runtime killer).

Approach (single pipeline, run on each case):
  1. BM25 preselect  -> top-k=20 sentences   (~50ms)
  2. LLMLingua-2     -> rate=0.5              (~7s on 6k chars)
  3. Same compressed prompt fed to BOTH models:
     - Mistral 8B  (ministral-8b-latest)
     - Gemini 3.5 Flash Lite

5 hard cases, smallest first from dev_first95.jsonl (proven fastest).

Outputs:
  results/phase-01-mistral-vs-gemini-compressed-n5.jsonl
  results/phase-01-mistral-vs-gemini-compressed-n5-summary.json

Usage:
  conda activate vsf
  python scripts/phase-01/eval_mistral_vs_gemini_compressed.py [--n 5]
"""
from __future__ import annotations

import argparse
import json
import os
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

# ---- Config ----
DEFAULT_DEV = REPO_ROOT / "data" / "processed" / "dev_first95.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-mistral-vs-gemini-compressed-n5.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-mistral-vs-gemini-compressed-n5-summary.json"

MISTRAL_MODEL = os.getenv("OP5_MISTRAL_MODEL", "ministral-8b-latest")
GEMINI_MODEL = os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite")
PRESELECT_K = 20
COMPRESS_RATE = 0.5  # proven optimal from eval_combo_tuned
FORCE_TOKENS = ["!", ".", "?", "\n"]
COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
SLEEP_BETWEEN = 3.0  # Mistral free-tier rate limit
SLEEP_GEMINI = 1.0  # Gemini more lenient


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------
# Shared infra (load once)
# ---------------------------------------------------------------
_pc = None


def get_llmlingua2():
    global _pc
    if _pc is None:
        from llmlingua import PromptCompressor
        print(f"  Loading LLMLingua-2 ({COMPRESSOR_MODEL}) on CPU...")
        _pc = PromptCompressor(
            model_name=COMPRESSOR_MODEL,
            use_llmlingua2=True,
            device_map="cpu",
        )
        print("  Compressor loaded.")
    return _pc


def get_mistral_client():
    from mistralai.client import Mistral
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")
    return Mistral(api_key=key)


def get_gemini_model():
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


# ---------------------------------------------------------------
# Helpers (reuse existing patterns from eval_mistral_combo_3.py)
# ---------------------------------------------------------------
import re


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


def preselect_bm25(context: str, question: str, k: int) -> dict:
    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"status": "error", "error": "scikit-learn not installed"}

    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vec = vec.fit_transform(sentences)
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, sent_vec).flatten()
    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    selected = " ".join(sentences[i] for i in top_idx)
    return {
        "status": "ok",
        "preselect_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "n_sentences": len(sentences),
        "n_kept": k,
        "preselect_chars": len(selected),
        "selected_text": selected,
    }


# SentenceTransformer model (loaded once, reused across cases)
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        print("  Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  SentenceTransformer loaded.")
    return _st_model


def _cosine_sim(a, b):
    """Normalised dot product."""
    import math
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (mag_a * mag_b)


def preselect_semantic(context: str, question: str, k: int) -> dict:
    t0 = time.perf_counter()
    model = _get_st_model()
    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    load_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    t1 = time.perf_counter()
    sent_embs = model.encode(sentences, batch_size=64, show_progress_bar=False)
    embed_ms = round((time.perf_counter() - t1) * 1000.0, 1)

    t2 = time.perf_counter()
    q_emb = model.encode([question], show_progress_bar=False)[0]
    q_encode_ms = round((time.perf_counter() - t2) * 1000.0, 1)

    scores = [(_cosine_sim(se.tolist(), q_emb.tolist()), i, s)
              for i, (se, s) in enumerate(zip(sent_embs, sentences))]
    scores.sort(reverse=True)
    top_k = scores[:k]
    selected = " ".join(s for _, _, s in top_k)

    return {
        "status": "ok",
        "preselect_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "load_ms": load_ms,
        "embed_ms": embed_ms,
        "question_encode_ms": q_encode_ms,
        "n_sentences": len(sentences),
        "n_kept": k,
        "preselect_chars": len(selected),
        "selected_text": selected,
    }


def preselect_hybrid(context: str, question: str, k: int) -> dict:
    """BM25 + semantic ensemble: merge ranked lists with weighted scores."""
    t0 = time.perf_counter()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    # BM25 scores (normalised to [0,1])
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vec = vec.fit_transform(sentences)
    q_vec = vec.transform([question])
    tfidf_sims = cosine_similarity(q_vec, sent_vec).flatten()
    tfidf_max = tfidf_sims.max() if tfidf_sims.max() > 0 else 1.0
    tfidf_norm = tfidf_sims / tfidf_max

    # Semantic scores (normalised to [0,1])
    model = _get_st_model()
    sent_embs = model.encode(sentences, batch_size=64, show_progress_bar=False)
    q_emb = model.encode([question], show_progress_bar=False)[0]
    sem_scores = [max(_cosine_sim(se.tolist(), q_emb.tolist()), 0) for se in sent_embs]
    sem_max = max(sem_scores) if max(sem_scores) > 0 else 1.0
    sem_norm = [s / sem_max for s in sem_scores]

    # Ensemble: 0.5*BM25 + 0.5*semantic
    ALPHA = 0.5
    ensemble = [ALPHA * tfidf_norm[i] + (1 - ALPHA) * sem_norm[i] for i in range(len(sentences))]
    top_idx = sorted(range(len(ensemble)), key=lambda i: ensemble[i], reverse=True)[:k]
    selected = " ".join(sentences[i] for i in top_idx)

    return {
        "status": "ok",
        "preselect_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "n_sentences": len(sentences),
        "n_kept": k,
        "preselect_chars": len(selected),
        "selected_text": selected,
    }


def compress_llmlingua2(pc, text: str, rate: float) -> dict:
    t0 = time.perf_counter()
    try:
        result = pc.compress_prompt(
            text,
            rate=rate,
            force_tokens=FORCE_TOKENS,
            drop_consecutive=True,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "ok",
            "compress_ms": round(ms, 1),
            "compressed_text": result.get("compressed_prompt", ""),
            "origin_tokens": result.get("origin_tokens"),
            "compressed_tokens": result.get("compressed_tokens"),
            "ratio_str": result.get("ratio", ""),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "compress_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }


def build_prompt(context: str, question: str, choices: dict) -> str:
    choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
    return (
        "Doc doan van ban sau va tra loi cau hoi. CHI tra ve mot chu cai (A, B, C hoac D) "
        "dung nhat. Khong giai thich.\n\n"
        f"CAU HOI: {question}\n\n"
        f"LUA CHON:\n{choices_str}\n\n"
        f"DOAN VAN BAN:\n{context}\n\n"
        "DAP AN (1 chu cai):"
    )


def normalize_pred(text: str) -> str:
    s = text.strip().upper()
    for ch in s:
        if ch in "ABCD":
            return ch
    return ""


def call_mistral(client, prompt: str, max_tokens: int = 256) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return {
            "text": text,
            "latency_ms": elapsed_ms,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "text": "",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


def call_gemini(model, prompt: str, max_tokens: int = 256) -> dict:
    t0 = time.perf_counter()
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        return {
            "text": text,
            "latency_ms": elapsed_ms,
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "text": "",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


# ---------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------
def load_n_random_cases(n: int, seed: int = 42, only_hard: bool = True) -> list[dict]:
    """
    Random sample n cases. Default: hard only (matches prior scripts).
    Set only_hard=False to sample across all difficulties for a balanced set.
    """
    # Use raw_decode so embedded newlines in context fields don't split a row
    rows: list[dict] = []
    decoder = json.JSONDecoder()
    with DEFAULT_DEV.open() as fh:
        buf = fh.read()
    pos = 0
    while pos < len(buf):
        # skip whitespace between rows
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

    if only_hard:
        pool = [r for r in rows if r.get("difficulty") == "hard"]
    else:
        pool = rows

    import random
    rng = random.Random(seed)
    sampled = rng.sample(pool, min(n, len(pool)))
    # Sort by context length (smallest first) so the run finishes quickly
    sampled.sort(key=lambda r: len(r.get("context", "")))
    return sampled


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_one_case(case_id: str, row: dict, client, gemini, pc, out_fh, preselect_name: str, preselect_fn) -> dict:
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    domain = row.get("domain", "?")

    print(f"\n[{case_id}] domain={domain} ctx={len(context):,}c gold={gold}")

    # Step 1: preselect
    ps = preselect_fn(context, question, k=PRESELECT_K)
    if ps["status"] != "ok":
        err = {"ts": now_iso(), "case_id": case_id, "status": "preselect_error",
               "error": ps.get("error", "")[:200]}
        out_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
        out_fh.flush()
        return err

    print(f"  preselect: {ps['preselect_ms']:.0f}ms "
          f"({ps['n_sentences']}->{ps['n_kept']} sents, {ps['preselect_chars']:,}c, "
          f"{ps['preselect_chars']/len(context)*100:.1f}% kept)")

    # Step 2: compress
    cc = compress_llmlingua2(pc, ps["selected_text"], rate=COMPRESS_RATE)
    if cc["status"] != "ok":
        err = {"ts": now_iso(), "case_id": case_id, "status": "compress_error",
               "error": cc.get("error", "")[:200]}
        out_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
        out_fh.flush()
        return err

    print(f"  compress:  {cc['compress_ms']:.0f}ms "
          f"ratio={cc.get('ratio_str','?')} "
          f"({cc.get('origin_tokens','?')}->{cc.get('compressed_tokens','?')} tok)")

    # Step 3: same prompt -> both models
    prompt = build_prompt(cc["compressed_text"], question, choices)

    # Mistral
    rm = call_mistral(client, prompt)
    pred_m = normalize_pred(rm["text"])
    correct_m = 1 if pred_m == gold else 0
    print(f"  Mistral ({MISTRAL_MODEL}):  "
          f"tokens={rm.get('input_tokens')} lat={rm['latency_ms']:.0f}ms "
          f"pred={pred_m!r} {'OK' if correct_m else 'X'}")
    time.sleep(SLEEP_BETWEEN)

    # Gemini
    rg = call_gemini(gemini, prompt)
    pred_g = normalize_pred(rg["text"])
    correct_g = 1 if pred_g == gold else 0
    print(f"  Gemini  ({GEMINI_MODEL}):  "
          f"tokens={rg.get('input_tokens')} lat={rg['latency_ms']:.0f}ms "
          f"pred={pred_g!r} {'OK' if correct_g else 'X'}")
    time.sleep(SLEEP_GEMINI)

    # Combined record
    rec = {
        "ts": now_iso(),
        "track": "qa_mc_compressed",
        "case_id": case_id,
        "config": "C_preselect+compress",
        "status": "ok",
        "gold": gold,
        "domain": domain,
        "context_chars": len(context),
        "preselect_ms": ps["preselect_ms"],
        "preselect_n_sentences": ps["n_sentences"],
        "preselect_n_kept": ps["n_kept"],
        "preselect_chars": ps["preselect_chars"],
        "compress_ms": cc["compress_ms"],
        "origin_tokens": cc.get("origin_tokens"),
        "compressed_tokens": cc.get("compressed_tokens"),
        "ratio_str": cc.get("ratio_str", ""),
        "mistral": {
            "model": MISTRAL_MODEL,
            "pred": pred_m,
            "correct": correct_m,
            "input_tokens": rm.get("input_tokens"),
            "output_tokens": rm.get("output_tokens"),
            "latency_ms": round(rm["latency_ms"], 1),
            "total_ms": round(ps["preselect_ms"] + cc["compress_ms"] + rm["latency_ms"], 1),
            "status": rm["status"],
            "raw_text": rm["text"][:200],
        },
        "gemini": {
            "model": GEMINI_MODEL,
            "pred": pred_g,
            "correct": correct_g,
            "input_tokens": rg.get("input_tokens"),
            "output_tokens": rg.get("output_tokens"),
            "latency_ms": round(rg["latency_ms"], 1),
            "total_ms": round(ps["preselect_ms"] + cc["compress_ms"] + rg["latency_ms"], 1),
            "status": rg["status"],
            "raw_text": rg["text"][:200],
        },
    }
    if rm["status"] == "error":
        rec["mistral"]["error"] = rm.get("error", "")[:200]
    if rg["status"] == "error":
        rec["gemini"]["error"] = rg.get("error", "")[:200]

    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of cases (random sample)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--include-all-difficulties", action="store_true",
                        help="Sample across all difficulties (default: hard only)")
    parser.add_argument("--preselect", choices=["bm25", "semantic", "hybrid"], default="bm25",
                        help="Preselect strategy (default: bm25)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    # Build preselect function from --preselect flag
    PRESELECT_MAP = {
        "bm25": preselect_bm25,
        "semantic": preselect_semantic,
        "hybrid": preselect_hybrid,
    }
    preselect_fn = PRESELECT_MAP[args.preselect]

    # Auto-name output files based on args
    tag = f"n{args.n}-{args.preselect}"
    out_path = REPO_ROOT / "results" / f"phase-01-mistral-vs-gemini-compressed-{tag}.jsonl"
    summary_path = REPO_ROOT / "results" / f"phase-01-mistral-vs-gemini-compressed-{tag}-summary.json"

    selected = load_n_random_cases(args.n, seed=args.seed, only_hard=not args.include_all_difficulties)
    if not selected:
        print("ERROR: no hard cases found in dev_first95.jsonl", file=sys.stderr)
        return 2

    print("=== Mistral vs Gemini on SAME Compressed Prompt ===")
    print(f"  preselect: {args.preselect} k={PRESELECT_K}")
    print(f"  compress:   LLMLingua-2 rate={COMPRESS_RATE} ({COMPRESSOR_MODEL})")
    print(f"  Mistral:    {MISTRAL_MODEL}")
    print(f"  Gemini:     {GEMINI_MODEL}")
    print(f"  cases:      {args.n} hard, smallest first, seed={args.seed}")
    for i, r in enumerate(selected):
        print(f"    D-{i:02d}-{r['_id'][:6]}: ctx={len(r.get('context','')):,}c "
              f"gold={r.get('answer','').strip().upper()}")

    client = get_mistral_client()
    gemini = get_gemini_model()
    pc = get_llmlingua2()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    per_case = []
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"D-{i:02d}-{row['_id'][:6]}"
            rec = run_one_case(case_id, row, client, gemini, pc, fh,
                               preselect_name=args.preselect, preselect_fn=preselect_fn)
            per_case.append((case_id, rec))
            if i < len(selected) - 1:
                time.sleep(2.0)  # cooldown between cases

    # ---- Aggregate ----
    mistral_correct = 0
    gemini_correct = 0
    n_ok = 0
    mistral_lats = []
    gemini_lats = []
    mistral_toks = []
    gemini_toks = []

    per_case_summary = []
    for cid, rec in per_case:
        cs = {"case_id": cid, "gold": rec.get("gold")}
        if rec.get("status") == "ok":
            n_ok += 1
            m = rec["mistral"]
            g = rec["gemini"]
            if m["status"] == "ok":
                mistral_correct += m["correct"]
                mistral_lats.append(m["latency_ms"])
                if m["input_tokens"]:
                    mistral_toks.append(m["input_tokens"])
            if g["status"] == "ok":
                gemini_correct += g["correct"]
                gemini_lats.append(g["latency_ms"])
                if g["input_tokens"]:
                    gemini_toks.append(g["input_tokens"])
            cs["mistral"] = {
                "pred": m["pred"], "correct": m["correct"],
                "input_tokens": m["input_tokens"],
                "latency_ms": m["latency_ms"],
                "total_ms": m["total_ms"],
            }
            cs["gemini"] = {
                "pred": g["pred"], "correct": g["correct"],
                "input_tokens": g["input_tokens"],
                "latency_ms": g["latency_ms"],
                "total_ms": g["total_ms"],
            }
            cs["shared_compress_ms"] = rec["compress_ms"]
        else:
            cs["status"] = rec.get("status")
            cs["error"] = rec.get("error", "")
        per_case_summary.append(cs)

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    summary = {
        "ts": now_iso(),
        "n_cases": len(per_case),
        "n_ok": n_ok,
        "config": {
            "preselect": args.preselect,
            "preselect_k": PRESELECT_K,
            "compress_rate": COMPRESS_RATE,
            "compressor": COMPRESSOR_MODEL,
        },
        "models": {"mistral": MISTRAL_MODEL, "gemini": GEMINI_MODEL},
        "totals": {
            "mistral": {
                "correct": f"{mistral_correct}/{n_ok}",
                "accuracy": round(mistral_correct / n_ok, 3) if n_ok else None,
                "avg_input_tokens": mean(mistral_toks),
                "avg_latency_ms": mean(mistral_lats),
            },
            "gemini": {
                "correct": f"{gemini_correct}/{n_ok}",
                "accuracy": round(gemini_correct / n_ok, 3) if n_ok else None,
                "avg_input_tokens": mean(gemini_toks),
                "avg_latency_ms": mean(gemini_lats),
            },
        },
        "per_case": per_case_summary,
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- Print ----
    print("\n=== Summary ===")
    print(f"  N cases: {n_ok}/{len(per_case)} ok")
    print(f"  {'Model':<12} {'Acc':>10} {'Tokens':>10} {'Latency':>12}")
    print("  " + "-" * 50)
    m = summary["totals"]["mistral"]
    g = summary["totals"]["gemini"]
    print(f"  {'Mistral':<12} {m['correct']:>10} "
          f"{m['avg_input_tokens'] or 0:>10.0f} {m['avg_latency_ms'] or 0:>10.1f}ms")
    print(f"  {'Gemini':<12} {g['correct']:>10} "
          f"{g['avg_input_tokens'] or 0:>10.0f} {g['avg_latency_ms'] or 0:>10.1f}ms")

    # Per-case table
    print("\n=== Per-case ===")
    print(f"  {'case':<14} {'gold':>4} {'M_pred':>6} {'M_OK':>4} "
          f"{'G_pred':>6} {'G_OK':>4} {'M_lat':>8} {'G_lat':>8}")
    print("  " + "-" * 64)
    for cs in per_case_summary:
        if "mistral" not in cs:
            print(f"  {cs['case_id']:<14} {cs.get('gold','?'):>4} ERR")
            continue
        m = cs["mistral"]
        g = cs["gemini"]
        print(f"  {cs['case_id']:<14} {cs['gold']:>4} "
              f"{m['pred'] or '?':>6} {'OK' if m['correct'] else 'X':>4} "
              f"{g['pred'] or '?':>6} {'OK' if g['correct'] else 'X':>4} "
              f"{m['latency_ms']:>7.0f}ms {g['latency_ms']:>7.0f}ms")

    print(f"\nWrote: {out_path}")
    print(f"       {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
