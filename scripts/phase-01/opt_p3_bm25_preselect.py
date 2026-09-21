#!/usr/bin/env python3
"""
Optimization experiment P3: Replace SentenceTransformer with BM25 for preselect.

Hypothesis:
  BM25 is ~10-100x faster than SentenceTransformer embedding.
  On CPU, BM25 for 381 sentences takes ~10-50ms vs 2,300ms for SentenceTransformer.
  Trade-off: BM25 uses lexical matching, may lose semantic recall.

Method:
  Compare preselect-only (approach B) with:
    - SentenceTransformer (all-MiniLM-L6-v2) 
    - BM25 (rank_bm25, using NLTK tokens)
    - TF-IDF (sklearn, for comparison)
  Measure: embed_ms, recall_proxy, accuracy on same case.

Usage:
  conda activate vsf
  python scripts/phase-01/opt_p3_bm25_preselect.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-opt-p3-bm25.json"

from compare_3_approaches import (
    load_shortest_case, split_sentences, cosine_sim,
    preselect_top_k, build_qa_prompt, call_gemini, normalize_pred, now_iso,
)


def tokenize_simple(text: str) -> list[str]:
    """Simple word tokenization, lowercase."""
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    # Remove stopwords
    stops = {
        "a","an","the","and","or","but","is","are","was","were","be","been",
        "being","have","has","had","do","does","did","will","would","could",
        "should","may","might","can","to","of","in","for","on","with","at",
        "by","from","as","into","through","during","before","after","above",
        "below","between","under","again","further","then","once","here",
        "there","when","where","why","how","all","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so",
        "than","too","very","just","also","now","it","its","this","that",
        "these","those","i","me","my","myself","we","our","ours","ourselves",
        "you","your","yours","yourself","yourselves","he","him","his",
        "himself","she","her","hers","herself","they","them","their",
        "theirs","themselves","what","which","who","whom","his","her","their",
    }
    return [w for w in words if w not in stops]


def preselect_bm25(context: str, question: str, k: int = 10) -> dict:
    """Preselect using BM25. Returns dict."""
    t0 = time.perf_counter()
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return {"status": "error", "error": "rank_bm25 not installed. Run: pip install rank-bm25"}

    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    # Tokenize
    tokenized = [tokenize_simple(s) for s in sentences]
    q_tokens = tokenize_simple(question)

    t1 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    t2 = time.perf_counter()

    scores = bm25.get_scores(q_tokens)
    t3 = time.perf_counter()

    # Top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    selected = [sentences[i] for i in top_indices]
    selected_text = " ".join(selected)

    bm25_build_ms = (t2 - t1) * 1000.0
    scoring_ms = (t3 - t2) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "ok",
        "total_ms": round(total_ms, 1),
        "bm25_build_ms": round(bm25_build_ms, 1),
        "scoring_ms": round(scoring_ms, 1),
        "embed_ms": round(bm25_build_ms + scoring_ms, 1),
        "n_sentences": len(sentences),
        "n_kept": k,
        "selected_text": selected_text,
        "selected_chars": len(selected_text),
        "char_ratio": round(len(selected_text) / max(len(context), 1), 4),
    }


def preselect_tfidf(context: str, question: str, k: int = 10) -> dict:
    """Preselect using TF-IDF. Returns dict."""
    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {"status": "error", "error": "scikit-learn not installed"}

    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    t1 = time.perf_counter()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vectors = vectorizer.fit_transform(sentences)
    t2 = time.perf_counter()

    q_vec = vectorizer.transform([question])
    t3 = time.perf_counter()

    # Cosine similarity (sparse)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_vec, sent_vectors).flatten()
    t4 = time.perf_counter()

    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    selected = [sentences[i] for i in top_indices]
    selected_text = " ".join(selected)

    fit_ms = (t2 - t1) * 1000.0
    transform_q_ms = (t3 - t2) * 1000.0
    sim_ms = (t4 - t3) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "ok",
        "total_ms": round(total_ms, 1),
        "fit_ms": round(fit_ms, 1),
        "transform_q_ms": round(transform_q_ms, 1),
        "sim_ms": round(sim_ms, 1),
        "embed_ms": round(fit_ms + transform_q_ms + sim_ms, 1),
        "n_sentences": len(sentences),
        "n_kept": k,
        "selected_text": selected_text,
        "selected_chars": len(selected_text),
        "char_ratio": round(len(selected_text) / max(len(context), 1), 4),
    }


def eval_preselect_method(method_name: str, context: str, question: str,
                           choices: dict, gold: str, gemini,
                           k: int, preselect_fn) -> dict:
    """Run preselect + Gemini, return full record."""
    t0 = time.perf_counter()
    result = preselect_fn(context, question, k)
    if result["status"] != "ok":
        return {"method": method_name, "status": "error", "error": result.get("error")}
    ctx = result["selected_text"]

    prompt = build_qa_prompt(ctx, question, choices)
    gr = call_gemini(gemini, prompt)
    pred = normalize_pred(gr["text"])

    return {
        "method": method_name,
        "k": k,
        "embed_ms": result["embed_ms"],
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "gemini_ms": gr["latency_ms"],
        "gemini_input_tokens": gr.get("input_tokens"),
        "preselect_chars": result["selected_chars"],
        "char_ratio": result["char_ratio"],
        "pred": pred,
        "correct": 1 if pred == gold else 0,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top sentences to keep (default 10)")
    args = parser.parse_args()

    row = load_shortest_case()
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    if gold not in "ABCD":
        gold = gold[0] if gold else "?"
    choices = {
        "A": row.get("choice_A", ""),
        "B": row.get("choice_B", ""),
        "C": row.get("choice_C", ""),
        "D": row.get("choice_D", ""),
    }

    print("=== Opt P3: BM25 vs SentenceTransformer preselect ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars")
    print(f"  gold:     {gold}")
    print(f"  k:        {args.k}")
    print()

    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return 1
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite"))

    SLEEP_BETWEEN = 1.5

    # Pre-load SentenceTransformer once
    print("Loading SentenceTransformer (cold)...")
    from compare_3_approaches import _get_sentence_model
    t0 = time.perf_counter()
    _get_sentence_model()
    st_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  SentenceTransformer load: {st_load_ms:.0f} ms")
    print()

    results = []

    # Method 1: SentenceTransformer (approach B)
    print("--- SentenceTransformer (warm) ---")
    r_st = eval_preselect_method(
        "SentenceTransformer", context, question, choices, gold, gemini,
        args.k, lambda c, q, k: preselect_top_k(c, q, k)
    )
    r_st["st_load_ms"] = st_load_ms
    results.append(r_st)
    if r_st["status"] == "ok":
        print(f"  embed={r_st['embed_ms']:>7.0f} ms  total={r_st['total_ms']:>7.0f} ms  "
              f"chars={r_st['preselect_chars']:>6,}  pred={r_st['pred']} correct={r_st['correct']}")
    else:
        print(f"  ERROR: {r_st.get('error')}")
    time.sleep(SLEEP_BETWEEN)

    # Method 2: BM25
    print("--- BM25 ---")
    r_bm25 = eval_preselect_method(
        "BM25", context, question, choices, gold, gemini,
        args.k, lambda c, q, k: preselect_bm25(c, q, k)
    )
    results.append(r_bm25)
    if r_bm25["status"] == "ok":
        print(f"  build={r_bm25.get('bm25_build_ms',0):>6.0f} ms  "
              f"score={r_bm25.get('scoring_ms',0):>5.0f} ms  "
              f"embed={r_bm25['embed_ms']:>7.0f} ms  "
              f"total={r_bm25['total_ms']:>7.0f} ms  "
              f"chars={r_bm25['preselect_chars']:>6,}  pred={r_bm25['pred']} correct={r_bm25['correct']}")
    else:
        print(f"  ERROR: {r_bm25.get('error')}")
    time.sleep(SLEEP_BETWEEN)

    # Method 3: TF-IDF
    print("--- TF-IDF ---")
    r_tfidf = eval_preselect_method(
        "TF-IDF", context, question, choices, gold, gemini,
        args.k, lambda c, q, k: preselect_tfidf(c, q, k)
    )
    results.append(r_tfidf)
    if r_tfidf["status"] == "ok":
        print(f"  fit={r_tfidf.get('fit_ms',0):>6.0f} ms  "
              f"q_trans={r_tfidf.get('transform_q_ms',0):>5.0f} ms  "
              f"sim={r_tfidf.get('sim_ms',0):>4.0f} ms  "
              f"embed={r_tfidf['embed_ms']:>7.0f} ms  "
              f"total={r_tfidf['total_ms']:>7.0f} ms  "
              f"chars={r_tfidf['preselect_chars']:>6,}  pred={r_tfidf['pred']} correct={r_tfidf['correct']}")
    else:
        print(f"  ERROR: {r_tfidf.get('error')}")
    time.sleep(SLEEP_BETWEEN)

    # Comparison table
    print()
    print("=== Summary ===")
    print(f"{'Method':<20} {'embed_ms':>10} {'total_ms':>10} {'Chars':>7} {'Tokens':>7} {'Pred':>4} {'Acc':>4}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get("total_ms", 999999)):
        if r.get("status") == "ok":
            print(
                f"  {r['method']:<20} "
                f"{r.get('embed_ms',0):>10.0f} "
                f"{r.get('total_ms',0):>10.0f} "
                f"{r.get('preselect_chars',0):>7,} "
                f"{r.get('gemini_input_tokens',0) or 0:>7} "
                f"{r.get('pred','?'):>4} "
                f"{r.get('correct','?'):>4}"
            )

    # Delta vs SentenceTransformer
    st_rec = next((r for r in results if r.get("method") == "SentenceTransformer" and r.get("status") == "ok"), None)
    if st_rec:
        print()
        print(f"{'Method':<20} {'embed_delta':>12} {'total_delta':>12} {'Acc_delta':>10}")
        print("-" * 56)
        for r in results:
            if r.get("status") != "ok":
                continue
            embed_delta = r["embed_ms"] - st_rec["embed_ms"]
            total_delta = r["total_ms"] - st_rec["total_ms"]
            acc_delta = (r.get("correct", 0) - st_rec.get("correct", 0)) * 100
            print(
                f"  {r['method']:<20} "
                f"{embed_delta:>+12.0f} ms "
                f"{total_delta:>+12.0f} ms "
                f"{acc_delta:>+10.0f}pp"
            )

    # Verdict
    print()
    print("=== Verdict ===")
    ok_results = [r for r in results if r.get("status") == "ok"]
    if st_rec and ok_results:
        best = min(ok_results, key=lambda r: r["total_ms"])
        # Only count as improvement if same accuracy AND faster
        speedup = st_rec["total_ms"] / max(best["total_ms"], 1)
        ms_saved = st_rec["total_ms"] - best["total_ms"]
        same_accuracy = all(r.get("correct") == st_rec.get("correct") for r in ok_results)

        print(f"  SentenceTransformer:  {st_rec['total_ms']:.0f} ms (embed={st_rec['embed_ms']:.0f})")
        for r in ok_results:
            if r["method"] != "SentenceTransformer":
                delta = r["total_ms"] - st_rec["total_ms"]
                print(f"  {r['method']:<20}  {r['total_ms']:.0f} ms (delta={delta:>+8.0f} ms) "
                      f"acc={r['correct']} vs {st_rec['correct']}")

        if ms_saved > 500 and same_accuracy:
            print(f"  DECISION: IMPROVE -- {best['method']} is {speedup:.2f}x faster, same accuracy")
            print(f"  ACTION: replace SentenceTransformer with {best['method']} for preselect")
            verdict = "IMPROVE"
            best_method = best["method"]
        elif ms_saved > 500:
            print(f"  DECISION: CONDITIONAL -- {best['method']} faster but accuracy differs")
            print(f"  Need more cases to verify recall quality")
            verdict = "CONDITIONAL"
            best_method = best["method"]
        else:
            print(f"  DECISION: NO_IMPROVEMENT -- fastest method saves < 500ms")
            verdict = "NO_IMPROVEMENT"
            best_method = "SentenceTransformer"
    else:
        verdict = "ERROR"
        best_method = "SentenceTransformer"

    print()

    # Save
    out_data = {
        "ts": now_iso(),
        "case_id": row.get("_id", ""),
        "context_chars": len(context),
        "gold": gold,
        "k": args.k,
        "results": results,
        "verdict": verdict,
        "best_method": best_method,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
