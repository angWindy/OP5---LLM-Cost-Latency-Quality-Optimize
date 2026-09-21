#!/usr/bin/env python3
"""
Mistral baseline eval — same 5 LongBench cases as Gemini.
Goal: measure context-length scaling of Mistral latency
      (does Mistral have more 'context tax' than Gemini flash-lite?).

For each of 5 cases (smallest→largest), call Mistral with FULL uncompressed context,
record:
  - input tokens, output tokens
  - latency_ms
  - normalized prediction

Output:
  results/phase-01-mistral-baseline-5cases.jsonl
  results/phase-01-mistral-baseline-summary.json
  results/phase-01-mistral-vs-gemini-latency.json   (comparison)

Usage:
    conda activate vsf
    python scripts/phase-01/eval_mistral_baseline_5.py
"""
from __future__ import annotations

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

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-mistral-baseline-5cases.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-mistral-baseline-summary.json"
DEFAULT_CMP = REPO_ROOT / "results" / "phase-01-mistral-vs-gemini-latency.json"

# Mistral free-tier (la Plateforme). mistral-tiny is cheapest+fastest.
MISTRAL_MODEL = os.getenv("OP5_MISTRAL_MODEL", "mistral-tiny")
SLEEP = 3.0  # avoid rate-limit (tier free ~2 req/s)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_mistral_client():
    """Return mistralai.client.Mistral (v2.x SDK layout)."""
    try:
        from mistralai.client import Mistral
    except ImportError as e:
        raise RuntimeError(
            "mistralai not installed — `pip install mistralai`"
        ) from e
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")
    return Mistral(api_key=key)


def call_mistral(client, prompt: str, max_tokens: int = 256) -> dict:
    """Single Mistral chat completion call. Returns dict."""
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
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": "",
            "latency_ms": elapsed_ms,
            "input_tokens": None,
            "output_tokens": None,
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


def build_prompt(context: str, question: str, choices: dict) -> str:
    """Same prompt format as eval_llmlingua_v2.py for fair comparison."""
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


def run_one_case(case_id: str, row: dict, client, out_fh) -> dict:
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "").strip().upper()
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    domain = row.get("domain", "?")

    prompt = build_prompt(context, question, choices)
    print(f"\n[{case_id}] domain={domain} ctx={len(context):,}c gold={gold}")

    r = call_mistral(client, prompt)
    pred = normalize_pred(r["text"])
    correct = 1 if pred == gold else 0

    rec = {
        "ts": now_iso(),
        "track": "qa_mc",
        "case_id": case_id,
        "config": "A_mistral_baseline",
        "model": MISTRAL_MODEL,
        "status": r["status"],
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "domain": domain,
        "context_chars": len(context),
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
        "gemini_ms": r["latency_ms"],   # field name reused for compare script
        "compressor_ms": 0.0,
        "total_ms": r["latency_ms"],
        "raw_text": r["text"][:300],
    }
    if r["status"] == "error":
        rec["error"] = r["error"]
    print(
        f"  tokens={r['input_tokens']} lat={r['latency_ms']:.0f}ms "
        f"pred={pred!r} {'OK' if correct else 'X'}"
    )
    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()
    return rec


def main() -> int:
    if not DEFAULT_TEST.exists():
        print(f"ERROR: {DEFAULT_TEST} not found.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in DEFAULT_TEST.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    selected = rows[:5]

    print("=== Mistral Baseline — 5 cases (smallest→largest) ===")
    print(f"  model: {MISTRAL_MODEL}")
    for i, row in enumerate(selected):
        print(f"  L5-{i:02d}: ctx={len(row.get('context','')):,}c  gold={row.get('answer','').strip().upper()}")

    client = get_mistral_client()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_OUT.exists():
        DEFAULT_OUT.unlink()

    per_case = []
    with DEFAULT_OUT.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"L5-{i:02d}-{row.get('_id', '?')[:6]}"
            r = run_one_case(case_id, row, client, fh)
            per_case.append((case_id, r))
            if i < len(selected) - 1:
                time.sleep(SLEEP)

    # Summary
    n_ok = sum(1 for _, r in per_case if r["status"] == "ok")
    if n_ok:
        avg_toks = sum(r["input_tokens"] or 0 for _, r in per_case) / n_ok
        avg_lat = sum(r["gemini_ms"] for _, r in per_case) / n_ok
        acc = sum(r["correct"] for _, r in per_case) / n_ok
    else:
        avg_toks = avg_lat = acc = 0

    summary = {
        "ts": now_iso(),
        "model": MISTRAL_MODEL,
        "n_cases": len(per_case),
        "n_ok": n_ok,
        "avg_input_tokens": round(avg_toks, 1),
        "avg_latency_ms": round(avg_lat, 1),
        "accuracy": round(acc, 3),
        "per_case": [
            {"case_id": cid, "ctx_chars": r["context_chars"],
             "input_tokens": r["input_tokens"], "latency_ms": r["gemini_ms"],
             "pred": r["pred"], "correct": r["correct"]}
            for cid, r in per_case
        ],
    }
    with DEFAULT_SUMMARY.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Compare vs Gemini (from phase-01-llmlingua-corrected-5cases.jsonl)
    gemini_path = REPO_ROOT / "results" / "phase-01-llmlingua-corrected-5cases.jsonl"
    cmp_data = {"ts": now_iso(), "mistral_model": MISTRAL_MODEL, "per_case": []}
    if gemini_path.exists():
        gemini_rows = [json.loads(l) for l in gemini_path.read_text().splitlines() if l.strip()]
        gem_baseline = {r["case_id"]: r for r in gemini_rows if r.get("config") == "A_baseline_gemini"}
        # Match by case_id
        for cid, r in per_case:
            g = gem_baseline.get(cid)
            if not g:
                continue
            g_toks = g.get("input_tokens") or 0
            m_toks = r["input_tokens"] or 0
            g_lat = g.get("gemini_ms") or 0
            m_lat = r["gemini_ms"] or 0
            cmp_data["per_case"].append({
                "case_id": cid,
                "context_chars": r["context_chars"],
                "gemini": {"input_tokens": g_toks, "latency_ms": round(g_lat, 1), "correct": g.get("accuracy")},
                "mistral": {"input_tokens": m_toks, "latency_ms": round(m_lat, 1), "correct": r["correct"]},
                "mistral_latency_overhead_ms": round(m_lat - g_lat, 1),
                "mistral_latency_ratio": round(m_lat / g_lat, 2) if g_lat else None,
            })
    with DEFAULT_CMP.open("w", encoding="utf-8") as fh:
        json.dump(cmp_data, fh, ensure_ascii=False, indent=2)

    # Print summary
    print("\n=== Mistral Baseline Summary ===")
    print(f"{'case':<14} {'ctx_chars':>9} {'toks':>7} {'lat_ms':>8} {'pred':>4} {'OK?':>4}")
    print("-" * 55)
    for cid, r in per_case:
        print(
            f"{cid:<14} {r['context_chars']:>9} {r['input_tokens'] or 0:>7} "
            f"{r['gemini_ms']:>8.0f} {r['pred'] or '?':>4} "
            f"{'OK' if r['correct'] else 'X':>4}"
        )
    print(f"\nAvg tokens: {avg_toks:.0f}  Avg latency: {avg_lat:.0f}ms  Accuracy: {acc:.1%}")
    print(f"\nWrote: {DEFAULT_OUT}")
    print(f"       {DEFAULT_SUMMARY}")
    print(f"       {DEFAULT_CMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
