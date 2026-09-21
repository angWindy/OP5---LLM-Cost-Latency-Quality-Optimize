#!/usr/bin/env python3
"""
Compare 3 compression approaches on 1 case (Legal 70k chars):

  Baseline  : Gemini only, no compression
  Approach A: LLMLingua-2 compress only (rate=0.5)
  Approach B: RAG preselect (embed + top-k) only, no compressor
  Approach C: RAG preselect + LLMLingua-2 compress

Writes:
  results/phase-01-3-approach-1case.jsonl  (one record per approach)
  results/phase-01-3-approach-1case.json   (summary)

Usage:
  conda activate vsf
  python scripts/phase-01/compare_3_approaches.py [--rate 0.5] [--k 20]
"""
from __future__ import annotations

import json
import re
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

DEFAULT_OUT_JSONL  = REPO_ROOT / "results" / "phase-01-3-approach-1case.jsonl"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "results" / "phase-01-3-approach-1case.json"
SLEEP_BETWEEN = 1.5

# ---------------------------------------------------------------
# Shared model instances (load once)
# ---------------------------------------------------------------
_st_model = None
_llmlingua_pc = None
_compressor_model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def _get_sentence_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _get_llmlingua():
    global _llmlingua_pc
    if _llmlingua_pc is None:
        from llmlingua import PromptCompressor
        _llmlingua_pc = PromptCompressor(
            model_name=_compressor_model_name,
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _llmlingua_pc


def _gemini_model():
    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite"))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_shortest_case():
    test_path = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    return rows[0]


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
    # Fallback: split on .!? followed by space+capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    if len(parts) > 1:
        return parts
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines
    return [text]


def cosine_sim(a, b):
    import math
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma == 0 or mb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (ma * mb)


def preselect_top_k(context: str, question: str, k: int = 10) -> dict:
    """RAG preselect: TF-IDF cosine similarity, keep top-k. Returns dict."""
    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
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

    sims = cosine_similarity(q_vec, sent_vectors).flatten()
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    selected = [sentences[i] for i in top_indices]
    selected_text = " ".join(selected)

    fit_ms = (t2 - t1) * 1000.0
    transform_q_ms = (t3 - t2) * 1000.0
    embed_ms = (t3 - t1) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "ok",
        "total_ms": round(total_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "n_sentences": len(sentences),
        "n_kept": k,
        "selected_text": selected_text,
        "selected_chars": len(selected_text),
        "char_ratio": round(len(selected_text) / max(len(context), 1), 4),
    }


def compress_llmlingua(context: str, question: str, rate: float = 0.5) -> tuple[str, float, dict]:
    """Compress with LLMLingua-2. Returns (compressed, total_ms, meta)."""
    pc = _get_llmlingua()
    t0 = time.perf_counter()
    try:
        result = pc.compress_prompt(
            context,
            question=question,
            rate=rate,
            force_tokens=["!", ".", "?", "\n"],
            drop_consecutive=True,
            return_word_label=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        compressed = result.get("compressed_prompt", "")
        return (
            compressed,
            ms,
            {
                "origin_tokens": result.get("origin_tokens"),
                "compressed_tokens": result.get("compressed_tokens"),
                "ratio_str": result.get("ratio", ""),
                "char_ratio": round(len(compressed) / max(len(context), 1), 4),
            },
        )
    except Exception as exc:
        return context, 0, {"status": "error", "error": str(exc)}


def build_qa_prompt(context: str, question: str, choices: dict[str, str]) -> str:
    choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
    return (
        "Doc doan van ban sau va tra loi cau hoi. CHI tra ve mot chu cai (A, B, C hoac D) "
        "dung nhat. Khong giai thich.\n\n"
        f"CAU HOI: {question}\n\n"
        f"LUA CHON:\n{choices_str}\n\n"
        f"DOAN VAN BAN:\n{context}\n\n"
        "DAP AN (1 chu cai):"
    )


def call_gemini(model, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 256},
        )
    except Exception as exc:
        return {"text": "", "latency_ms": 0, "error": str(exc)}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = (resp.text or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def normalize_pred(text: str) -> str:
    s = text.strip().upper()
    for ch in s:
        if ch in "ABCD":
            return ch
    return ""


def run_approach(
    name: str,
    context: str,
    question: str,
    choices: dict[str, str],
    gold: str,
    gemini,
    rate: float,
    k: int,
    use_preselect: bool,
    use_compress: bool,
) -> dict:
    """
    Run one approach.
    use_preselect: if True, run RAG preselect first
    use_compress:  if True, run LLMLingua-2 compress on the current context
    """
    t_total = time.perf_counter()

    record = {
        "approach": name,
        "rate": rate,
        "k": k,
        "context_chars": len(context),
        "question_chars": len(question),
        "gold": gold,
        "status": "ok",
    }

    ctx = context  # current working context
    steps = []

    # Step 1: preselect
    if use_preselect:
        ps = preselect_top_k(ctx, question, k=k)
        if ps["status"] != "ok":
            record["status"] = "preselect_error"
            record["error"] = ps.get("error", "unknown")
            record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
            return record
        record["preselect_ms"] = ps["total_ms"]
        record["preselect_n_sentences"] = ps["n_sentences"]
        record["preselect_n_kept"] = ps["n_kept"]
        record["preselect_chars"] = ps["selected_chars"]
        record["preselect_char_ratio"] = ps["char_ratio"]
        ctx = ps["selected_text"]
        steps.append(f"preselect({ps['total_ms']:.0f}ms)")
    else:
        record["preselect_ms"] = 0
        record["preselect_n_sentences"] = None
        record["preselect_n_kept"] = None
        record["preselect_chars"] = len(context)
        record["preselect_char_ratio"] = 1.0

    # Step 2: compress
    if use_compress:
        comp_text, comp_ms, meta = compress_llmlingua(ctx, question, rate=rate)
        if meta.get("status") == "error":
            record["status"] = "compress_error"
            record["error"] = meta.get("error", "")
            record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
            return record
        record["compress_ms"] = round(comp_ms, 1)
        record["compress_ratio_str"] = meta.get("ratio_str", "")
        record["compress_char_ratio"] = meta.get("char_ratio", 1.0)
        ctx = comp_text
        steps.append(f"compress({comp_ms:.0f}ms)")
    else:
        record["compress_ms"] = 0
        record["compress_ratio_str"] = "1.0x"
        record["compress_char_ratio"] = 1.0

    # Step 3: Gemini
    prompt = build_qa_prompt(ctx, question, choices)
    gr = call_gemini(gemini, prompt)
    if gr.get("error"):
        record["status"] = "gemini_error"
        record["error"] = gr.get("error", "")
        record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)
        return record

    record["gemini_ms"] = round(gr["latency_ms"], 1)
    record["gemini_input_tokens"] = gr.get("input_tokens")
    record["gemini_output_tokens"] = gr.get("output_tokens")
    record["gemini_text"] = gr["text"][:300]

    # Score
    pred = normalize_pred(gr["text"])
    record["pred"] = pred
    record["correct"] = 1 if pred == gold else 0
    record["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)

    return record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    # Load case
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

    print("=== Compare 3 Approaches (1 case) ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars")
    print(f"  question: {question[:80]}...")
    print(f"  gold:     {gold}")
    print(f"  rate={args.rate}, k={args.k}")
    print()

    gemini = _gemini_model()

    # 4 configs: (name, use_preselect, use_compress)
    configs = [
        ("baseline",    False, False),  # Gemini only
        ("A_compress", False, True),   # LLMLingua-2 only
        ("B_preselect", True, False),  # RAG preselect only
        ("C_combo",    True,  True),    # Preselect + compress
    ]

    records = []
    for name, use_ps, use_cp in configs:
        print(f"--- {name} ---")
        rec = run_approach(
            name=name,
            context=context,
            question=question,
            choices=choices,
            gold=gold,
            gemini=gemini,
            rate=args.rate,
            k=args.k,
            use_preselect=use_ps,
            use_compress=use_cp,
        )
        rec["ts"] = now_iso()
        rec["case_id"] = row.get("_id", "")
        records.append(rec)

        if rec.get("status") == "ok":
            print(f"  preselect:  {rec.get('preselect_ms',0):>8.0f} ms  "
                  f"(chars {rec.get('preselect_chars','?'):,} / {rec.get('preselect_char_ratio',1)*100:.1f}%)")
            print(f"  compress:   {rec.get('compress_ms',0):>8.0f} ms  "
                  f"ratio={rec.get('compress_ratio_str','?')}")
            print(f"  gemini:     {rec.get('gemini_ms',0):>8.0f} ms  "
                  f"tokens={rec.get('gemini_input_tokens','?')}")
            print(f"  TOTAL:    {rec.get('total_ms',0):>8.0f} ms  pred={rec.get('pred','?')} correct={rec.get('correct','?')}")
        else:
            print(f"  ERROR: {rec.get('error', rec.get('status'))}")

        print()
        time.sleep(SLEEP_BETWEEN)

    # Summary table
    print("=== Summary ===")
    print(f"{'Approach':<16} {'PreSel ms':>10} {'Compress ms':>12} {'Gemini ms':>10} {'Total ms':>10} {'Chars kept':>9} {'Pred':>4} {'OK?':>4}")
    print("-" * 84)
    for rec in records:
        if rec.get("status") == "ok":
            print(
                f"  {rec['approach']:<16} "
                f"{rec.get('preselect_ms',0):>10.0f} "
                f"{rec.get('compress_ms',0):>12.0f} "
                f"{rec.get('gemini_ms',0):>10.0f} "
                f"{rec.get('total_ms',0):>10.0f} "
                f"{rec.get('preselect_chars', len(context)):>9,} "
                f"{rec.get('pred','?'):>4} "
                f"{rec.get('correct','?'):>4}"
            )
        else:
            print(f"  {rec['approach']:<16}  ERROR: {rec.get('error','')}")

    # Delta vs baseline
    baseline = next((r for r in records if r["approach"] == "baseline"), None)
    if baseline and baseline.get("status") == "ok":
        print()
        print(f"{'Approach':<16} {'Delta Total':>12} {'Chars saved':>10} {'Accuracy delta':>14}")
        print("-" * 56)
        for rec in records:
            if rec.get("status") != "ok":
                continue
            delta_ms = rec["total_ms"] - baseline["total_ms"]
            bl_chars = baseline.get("preselect_chars", len(context))
            rc_chars = rec.get("preselect_chars", bl_chars)
            chars_saved = round((1 - rc_chars / max(bl_chars, 1)) * 100, 1)
            acc_delta = (rec.get("correct", 0) - baseline.get("correct", 0)) * 100
            print(
                f"  {rec['approach']:<16} "
                f"{delta_ms:>+12.0f} "
                f"{chars_saved:>+9.1f}% "
                f"{acc_delta:>+13.0f}pp"
            )

    # Write JSONL
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl.exists():
        args.out_jsonl.unlink()
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "ts": now_iso(),
        "case_id": row.get("_id", ""),
        "domain": row.get("domain", ""),
        "context_chars": len(context),
        "gold": gold,
        "args": {"rate": args.rate, "k": args.k},
        "records": records,
    }
    with args.out_summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nWrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
