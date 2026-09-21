#!/usr/bin/env python3
"""
Benchmark RAG preselection (sentence embedding + top-k) on 1 case.

Pipeline:
  1. NLTK sentence split (context -> sentences)
  2. Encode all sentences with SentenceTransformer('all-MiniLM-L6-v2')
  3. Encode question -> cosine similarity
  4. Keep top-k sentences -> reconstruct context

Metrics:
  - embed_time_ms: time to encode all sentences
  - n_sentences_total / n_sentences_kept
  - recall: fraction of gold keywords present in top-k
  - chars_kept: original chars vs selected chars

Usage:
  conda activate vsf
  python scripts/phase-01/preselect_benchmark.py
"""
from __future__ import annotations

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

DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-preselect-benchmark.json"


def load_shortest_case():
    test_path = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found -- run setup_dataset.py first")
    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    return rows[0]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristic (NLTK fallback)."""
    # Try NLTK first
    try:
        import nltk
        nltk.data.find("tokenizer/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        if len(sents) > 1:
            return sents
    except Exception:
        pass

    # Fallback: split on . ! ? followed by space/capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    if len(parts) > 1:
        return parts

    # Last resort: split on newlines then sentence-like patterns
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines

    return [text]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product (normalized vectors)."""
    import math
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (mag_a * mag_b)


def preselect_top_k(
    context: str,
    question: str,
    k: int = 20,
    embed_batch_size: int = 64,
) -> dict:
    """
    RAG preselect: embed sentences, score vs question, keep top-k.
    Returns dict with timing, recall, and selected text.
    """
    from sentence_transformers import SentenceTransformer

    # 1. Split
    t0 = time.perf_counter()
    sentences = split_sentences(context)
    split_ms = (time.perf_counter() - t0) * 1000.0

    if not sentences:
        return {"status": "error", "error": "No sentences extracted"}

    # 2. Load model and encode
    t1 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    sent_embeddings = model.encode(sentences, batch_size=embed_batch_size, show_progress_bar=False)
    embed_ms = (time.perf_counter() - t2) * 1000.0

    # 3. Encode question
    t3 = time.perf_counter()
    question_emb = model.encode([question], show_progress_bar=False)[0]
    question_encode_ms = (time.perf_counter() - t3) * 1000.0

    # 4. Score sentences
    scores = []
    for i, sent_emb in enumerate(sent_embeddings):
        sim = cosine_similarity(sent_emb.tolist(), question_emb.tolist())
        scores.append((sim, i, sentences[i]))

    # 5. Top-k
    scores.sort(reverse=True)
    top_k = scores[:k]
    selected_sents = [s for _, _, s in top_k]
    selected_text = " ".join(selected_sents)

    # 6. Recall proxy: check if answer choice texts appear in selected
    # (approximate -- we don't have exact answer passage, but we can check
    #  that key words from the gold answer or question appear)
    gold_keywords = _extract_keywords(question)
    selected_lower = selected_text.lower()
    keywords_found = sum(1 for kw in gold_keywords if kw.lower() in selected_lower)
    recall_proxy = keywords_found / max(len(gold_keywords), 1)

    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "ok",
        "k": k,
        "n_sentences_total": len(sentences),
        "n_sentences_kept": len(selected_sents),
        "chars_original": len(context),
        "chars_selected": len(selected_text),
        "char_ratio": round(len(selected_text) / max(len(context), 1), 4),
        "split_ms": round(split_ms, 1),
        "load_ms": round(load_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "question_encode_ms": round(question_encode_ms, 1),
        "total_ms": round(total_ms, 1),
        "sentences_per_sec": round(len(sentences) / max(embed_ms / 1000.0, 0.001)),
        "recall_proxy": round(recall_proxy, 3),
        "recall_proxy_detail": {
            "keywords_found": keywords_found,
            "total_keywords": len(gold_keywords),
            "keywords": gold_keywords,
        },
        "selected_text_preview": selected_text[:500],
    }


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction: lowercase words >= 5 chars, stopword filter."""
    try:
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        stops = set(stopwords.words("english"))
    except Exception:
        stops = set()

    words = re.findall(r"[a-zA-Z]{5,}", text.lower())
    return [w for w in words if w not in stops][:20]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=20,
                        help="Number of top sentences to keep (default 20)")
    args = parser.parse_args()

    row = load_shortest_case()
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "")

    print("=== RAG Preselect Benchmark ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars")
    print(f"  question: {question[:80]}...")
    print(f"  gold:     {gold}")
    print(f"  k:        {args.k}")
    print()

    result = preselect_top_k(context, question, k=args.k)

    if result["status"] == "ok":
        print(f"  Split time:      {result['split_ms']:.0f} ms")
        print(f"  Model load:      {result['load_ms']:.0f} ms")
        print(f"  Embed sentences: {result['embed_ms']:.0f} ms ({result['sentences_per_sec']:.0f} sents/s)")
        print(f"  Encode question: {result['question_encode_ms']:.0f} ms")
        print(f"  Total:           {result['total_ms']:.0f} ms")
        print()
        print(f"  Sentences:       {result['n_sentences_total']} total -> {result['n_sentences_kept']} kept")
        print(f"  Chars:           {result['chars_original']:,} -> {result['chars_selected']:,} "
              f"({result['char_ratio']*100:.1f}%)")
        print(f"  Recall proxy:    {result['recall_proxy']:.1%} "
              f"({result['recall_proxy_detail']['keywords_found']}/{result['recall_proxy_detail']['total_keywords']} keywords)")
        print()
        print(f"  Selected preview: {result['selected_text_preview'][:200]}...")
    else:
        print(f"  ERROR: {result.get('error', 'unknown')}")

    out_data = {
        "case_id": row.get("_id", ""),
        "domain": row.get("domain", ""),
        "context_chars": len(context),
        "question_chars": len(question),
        "gold": gold,
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
