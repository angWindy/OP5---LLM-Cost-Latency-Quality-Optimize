#!/usr/bin/env python3
"""
Mistral combo eval — same 3 cases as baseline.
For each case run 3 configs:
  A: Mistral full context  (already have baseline)
  B: Mistral + LLMLingua-2 compress only (rate=0.5)
  C: Mistral + preselect (BM25 top-20) + LLMLingua-2 compress

Goal: does compressor+preselect BEAT Mistral's 7.4s baseline at L5-01 (89k tok)?

Output:
  results/phase-01-mistral-combo-3cases.jsonl
  results/phase-01-mistral-combo-summary.json

Usage:
    conda activate vsf
    python scripts/phase-01/eval_mistral_combo_3.py
"""
from __future__ import annotations

import json
import os
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

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-mistral-combo-3cases.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-mistral-combo-summary.json"

# Match baseline's model + Mistral tier
MISTRAL_MODEL = os.getenv("OP5_MISTRAL_MODEL", "ministral-8b-latest")
SLEEP = 4.0  # free tier is strict
COMPRESS_RATE = 0.5
PRESELECT_K = 20
FORCE_TOKENS = ["!", ".", "?", "\n"]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_mistral_client():
    from mistralai.client import Mistral
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")
    return Mistral(api_key=key)


def get_llmlingua2():
    from llmlingua import PromptCompressor
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )


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


def preselect_bm25(context: str, question: str, k: int = 20) -> dict:
    """BM25 (sklearn TF-IDF) top-k sentence preselect."""
    t0 = time.perf_counter()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"status": "error", "error": "scikit-learn not installed"}

    sentences = split_sentences(context)
    if not sentences:
        return {"status": "error", "error": "No sentences"}

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    sent_vec = vectorizer.fit_transform(sentences)
    q_vec = vectorizer.transform([question])
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


def compress_llmlingua2(pc, text: str, rate: float = 0.5) -> dict:
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
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}", "compress_ms": 0.0}


def run_one_case(case_id: str, row: dict, client, pc, out_fh) -> list:
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    domain = row.get("domain", "?")

    results = []

    # =========================================================
    # Config A: baseline (full context, no compressor)
    # =========================================================
    print(f"\n[{case_id}] A_baseline_mistral (full {len(context):,}c)")
    prompt_a = build_prompt(context, question, choices)
    ra = call_mistral(client, prompt_a)
    pred_a = normalize_pred(ra["text"])
    results.append({
        "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
        "config": "A_baseline_mistral", "model": MISTRAL_MODEL, "status": ra["status"],
        "gold": gold, "pred": pred_a, "correct": 1 if pred_a == gold else 0,
        "domain": domain, "context_chars": len(context),
        "preselect_ms": 0, "compress_ms": 0,
        "input_tokens": ra.get("input_tokens"), "output_tokens": ra.get("output_tokens"),
        "mistral_ms": ra["latency_ms"], "total_ms": ra["latency_ms"],
        "raw_text": ra["text"][:300],
    })
    print(f"  A: tokens={ra.get('input_tokens')} lat={ra['latency_ms']:.0f}ms pred={pred_a!r}")
    if ra["status"] == "error":
        print(f"  ERR: {ra.get('error','')[:100]}")
    time.sleep(SLEEP)

    # =========================================================
    # Config B: LLMLingua-2 compress only (rate=0.5)
    # =========================================================
    print(f"[{case_id}] B_compress_only (full ctx → LLMLingua-2 rate={COMPRESS_RATE})")
    cb = compress_llmlingua2(pc, context, rate=COMPRESS_RATE)
    if cb["status"] == "ok":
        prompt_b = build_prompt(cb["compressed_text"], question, choices)
        rb = call_mistral(client, prompt_b)
        pred_b = normalize_pred(rb["text"])
        results.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "B_compress_only", "model": MISTRAL_MODEL, "status": rb["status"],
            "gold": gold, "pred": pred_b, "correct": 1 if pred_b == gold else 0,
            "domain": domain, "context_chars": len(context),
            "preselect_ms": 0, "compress_ms": cb["compress_ms"],
            "input_tokens": rb.get("input_tokens"), "output_tokens": rb.get("output_tokens"),
            "mistral_ms": rb["latency_ms"],
            "total_ms": cb["compress_ms"] + rb["latency_ms"],
            "origin_tokens": cb.get("origin_tokens"),
            "compressed_tokens": cb.get("compressed_tokens"),
            "ratio_str": cb.get("ratio_str", ""),
            "raw_text": rb["text"][:300],
        })
        print(
            f"  B: comp={cb['compress_ms']:.0f}ms ratio={cb.get('ratio_str','?')} "
            f"mistral={rb['latency_ms']:.0f}ms total={cb['compress_ms']+rb['latency_ms']:.0f}ms pred={pred_b!r}"
        )
    else:
        results.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "B_compress_only", "status": "compress_error",
            "error": cb.get("error", ""),
        })
        print(f"  B: COMPRESS ERR {cb.get('error','')[:100]}")
    time.sleep(SLEEP)

    # =========================================================
    # Config C: preselect (BM25 top-20) + LLMLingua-2 compress
    # =========================================================
    print(f"[{case_id}] C_preselect+compress (BM25 k={PRESELECT_K} → LLMLingua-2 rate={COMPRESS_RATE})")
    ps = preselect_bm25(context, question, k=PRESELECT_K)
    if ps["status"] == "ok":
        cc = compress_llmlingua2(pc, ps["selected_text"], rate=COMPRESS_RATE)
        if cc["status"] == "ok":
            prompt_c = build_prompt(cc["compressed_text"], question, choices)
            rc = call_mistral(client, prompt_c)
            pred_c = normalize_pred(rc["text"])
            total_c = ps["preselect_ms"] + cc["compress_ms"] + rc["latency_ms"]
            results.append({
                "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                "config": "C_preselect_compress", "model": MISTRAL_MODEL, "status": rc["status"],
                "gold": gold, "pred": pred_c, "correct": 1 if pred_c == gold else 0,
                "domain": domain, "context_chars": len(context),
                "preselect_ms": ps["preselect_ms"],
                "preselect_n_sentences": ps["n_sentences"],
                "preselect_n_kept": ps["n_kept"],
                "preselect_chars": ps["preselect_chars"],
                "compress_ms": cc["compress_ms"],
                "input_tokens": rc.get("input_tokens"), "output_tokens": rc.get("output_tokens"),
                "mistral_ms": rc["latency_ms"], "total_ms": total_c,
                "origin_tokens": cc.get("origin_tokens"),
                "compressed_tokens": cc.get("compressed_tokens"),
                "ratio_str": cc.get("ratio_str", ""),
                "raw_text": rc["text"][:300],
            })
            print(
                f"  C: presel={ps['preselect_ms']:.0f}ms comp={cc['compress_ms']:.0f}ms "
                f"ratio={cc.get('ratio_str','?')} mistral={rc['latency_ms']:.0f}ms "
                f"TOTAL={total_c:.0f}ms pred={pred_c!r}"
            )
        else:
            results.append({
                "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
                "config": "C_preselect_compress", "status": "compress_error",
                "error": cc.get("error", ""),
            })
            print(f"  C: COMPRESS ERR {cc.get('error','')[:100]}")
    else:
        results.append({
            "ts": now_iso(), "track": "qa_mc", "case_id": case_id,
            "config": "C_preselect_compress", "status": "preselect_error",
            "error": ps.get("error", ""),
        })
        print(f"  C: PRESELECT ERR {ps.get('error','')[:100]}")

    for r in results:
        out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_fh.flush()
    return results


def main() -> int:
    if not DEFAULT_TEST.exists():
        print(f"ERROR: {DEFAULT_TEST} not found.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in DEFAULT_TEST.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))

    # Use first 3 cases (L5-00/01/02). L5-03 (627k) and L5-04 (2.5M) failed in baseline
    # — Mistral likely rejected due to context window.
    selected = rows[:3]
    print(f"=== Mistral Combo — {len(selected)} cases ===")
    print(f"  model: {MISTRAL_MODEL}")
    print(f"  preselect_k={PRESELECT_K}, compress_rate={COMPRESS_RATE}, ft={FORCE_TOKENS}")
    for i, row in enumerate(selected):
        print(f"  L5-{i:02d}: ctx={len(row.get('context','')):,}c  gold={row.get('answer','').strip().upper()}")

    client = get_mistral_client()
    pc = get_llmlingua2()  # load once (~30s, but amortized across calls)
    print("\nCompressor loaded.\n")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_OUT.exists():
        DEFAULT_OUT.unlink()

    per_case = []
    with DEFAULT_OUT.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"L5-{i:02d}-{row.get('_id', '?')[:6]}"
            res = run_one_case(case_id, row, client, pc, fh)
            per_case.append((case_id, res))
            if i < len(selected) - 1:
                time.sleep(SLEEP)

    # Summary
    summary = {"ts": now_iso(), "model": MISTRAL_MODEL, "per_case": []}
    for cid, recs in per_case:
        cs = {"case_id": cid}
        for r in recs:
            cfg = r.get("config")
            if r.get("status") == "ok":
                cs[cfg] = {
                    "mistral_ms": round(r["mistral_ms"], 1),
                    "compress_ms": r.get("compress_ms", 0),
                    "preselect_ms": r.get("preselect_ms", 0),
                    "total_ms": round(r["total_ms"], 1),
                    "input_tokens": r.get("input_tokens"),
                    "compressed_tokens": r.get("compressed_tokens"),
                    "correct": r.get("correct"),
                }
            else:
                cs[cfg] = {"status": r.get("status"), "error": r.get("error", "")[:100]}
        summary["per_case"].append(cs)

    # Add comparison vs baseline (from phase-01-mistral-baseline-5cases.jsonl)
    baseline_path = REPO_ROOT / "results" / "phase-01-mistral-baseline-5cases.jsonl"
    if baseline_path.exists():
        b_rows = [json.loads(l) for l in baseline_path.read_text().splitlines() if l.strip()]
        b_by_id = {r["case_id"]: r for r in b_rows}
        for cs in summary["per_case"]:
            cid = cs["case_id"]
            bl = b_by_id.get(cid)
            if not bl or bl.get("status") != "ok":
                continue
            for cfg in ("B_compress_only", "C_preselect_compress"):
                if cfg in cs and isinstance(cs[cfg], dict) and "total_ms" in cs[cfg]:
                    cs[cfg]["speedup_vs_baseline"] = round(
                        bl["total_ms"] / cs[cfg]["total_ms"], 2
                    )
                    cs[cfg]["saved_ms"] = round(bl["total_ms"] - cs[cfg]["total_ms"], 1)

    with DEFAULT_SUMMARY.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Print table
    print("\n=== Mistral Combo Summary ===")
    print(
        f"{'case':<14} {'A_ms':>7} {'B_ms':>7} {'C_ms':>7} "
        f"{'B_speed':>8} {'C_speed':>8} {'B_acc':>5} {'C_acc':>5}"
    )
    print("-" * 80)
    for cs in summary["per_case"]:
        cid = cs["case_id"]
        a = cs.get("A_baseline_mistral", {}).get("mistral_ms", 0)
        b = cs.get("B_compress_only", {})
        c = cs.get("C_preselect_compress", {})
        b_ms = b.get("total_ms", 0) if isinstance(b, dict) else 0
        c_ms = c.get("total_ms", 0) if isinstance(c, dict) else 0
        b_spd = b.get("speedup_vs_baseline", "-") if isinstance(b, dict) else "-"
        c_spd = c.get("speedup_vs_baseline", "-") if isinstance(c, dict) else "-"
        b_acc = b.get("correct", "-") if isinstance(b, dict) else "-"
        c_acc = c.get("correct", "-") if isinstance(c, dict) else "-"
        print(
            f"{cid:<14} {a:>7.0f} {b_ms:>7.0f} {c_ms:>7.0f} "
            f"{str(b_spd):>8} {str(c_spd):>8} {str(b_acc):>5} {str(c_acc):>5}"
        )
    print(f"\nWrote: {DEFAULT_OUT}")
    print(f"       {DEFAULT_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
