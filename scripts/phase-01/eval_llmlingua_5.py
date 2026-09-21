#!/usr/bin/env python3
"""
Phase 1 - LLMLingua eval 5 cases.

For each row in data/processed/llmlingua_test5.jsonl, run 3 configs and compare:
  - baseline         (no compression, prompt goc -> Gemini)
  - compressed_pip   (LLMLingua-2 via pip-direct PromptCompressor -> Gemini)
  - compressed_lc    (LLMLingua-2 via LangChain LLMLinguaCompressor -> Gemini)

Three metrics per case:
  - tokens_in   (% token saved vs baseline)
  - latency_ms  (% time saved; total = compressor + gemini)
  - accuracy    (exact-match normalized string vs gold 'A'|'B'|'C'|'D')

Output:
  - results/phase-01-llmlingua-5cases.jsonl     (per-config records)
  - results/phase-01-llmlingua-5cases-summary.json  (aggregated + delta %)

Usage:
    conda activate vsf
    python scripts/phase-01/eval_llmlingua_5.py [--n 5] [--max-context-chars 200000]
"""
from __future__ import annotations

import argparse
import json
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
from smoke_both_paths import compress_pip_direct, compress_langchain, chunk_context

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-llmlingua-5cases.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "phase-01-llmlingua-5cases-summary.json"

CHUNK_CHARS = 1500
RATE = 0.33
GEMINI_MODEL = "gemini-3.5-flash-lite"
SLEEP_BETWEEN_CALLS = 0.5


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_gemini_model():
    import google.generativeai as genai
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY chua duoc set trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(os.getenv("OP5_GEMINI_MODEL", GEMINI_MODEL))


def call_gemini(model, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict:
    t0 = time.perf_counter()
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return {
        "text": text,
        "latency_ms": elapsed_ms,
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def build_qa_prompt(context: str, question: str, choices: dict[str, str]) -> str:
    """Prompt nhieu lua chon A/B/C/D, yeu cau tra ve dung 1 chu cai."""
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
    """Lay chu cai dau tien A/B/C/D trong text, lower, strip."""
    s = text.strip().upper()
    for ch in s:
        if ch in "ABCD":
            return ch
    return s[:10]  # fallback: 10 char dau de debug


def normalize_gold(gold: str) -> str:
    return gold.strip().upper()


def run_one_case(
    case_id: str,
    row: dict,
    model,
    out_fh,
) -> dict:
    """Run 3 configs for 1 case. Write records to out_fh. Return per-case metrics."""
    context = row.get("context", "")
    question = row.get("question", "")
    gold = normalize_gold(row.get("answer", ""))
    choices = {
        "A": row.get("choice_A", ""),
        "B": row.get("choice_B", ""),
        "C": row.get("choice_C", ""),
        "D": row.get("choice_D", ""),
    }
    domain = row.get("domain", "?")
    n_chunks = len(chunk_context(context))

    print(f"\n[{case_id}] domain={domain} chunks={n_chunks} gold={gold}")

    metrics: dict[str, dict] = {}

    # --- baseline ---
    prompt_b = build_qa_prompt(context, question, choices)
    rb = call_gemini(model, prompt_b)
    pred_b = normalize_pred(rb["text"])
    correct_b = (pred_b == gold)
    tokens_b = rb["input_tokens"] or (len(prompt_b) // 4)
    metrics["baseline"] = {
        "input_tokens": tokens_b,
        "compressor_latency_ms": 0.0,
        "gemini_latency_ms": rb["latency_ms"],
        "total_latency_ms": rb["latency_ms"],
        "accuracy": 1 if correct_b else 0,
        "pred": pred_b,
        "raw_text": rb["text"][:300],
        "compression_ratio": 1.0,
    }
    print(f"  baseline: tokens={tokens_b} lat={rb['latency_ms']:.0f}ms pred={pred_b!r} {'OK' if correct_b else 'X'}")
    time.sleep(SLEEP_BETWEEN_CALLS)

    # --- compressed_pip ---
    try:
        compressed_a, comp_ms_a = compress_pip_direct(context, question, rate=RATE)
        prompt_a = build_qa_prompt(compressed_a, question, choices)
        ra = call_gemini(model, prompt_a)
        pred_a = normalize_pred(ra["text"])
        correct_a = (pred_a == gold)
        tokens_a = ra["input_tokens"] or (len(prompt_a) // 4)
        ratio_a = len(compressed_a) / max(len(context), 1)
        metrics["compressed_pip"] = {
            "input_tokens": tokens_a,
            "compressor_latency_ms": comp_ms_a,
            "gemini_latency_ms": ra["latency_ms"],
            "total_latency_ms": comp_ms_a + ra["latency_ms"],
            "accuracy": 1 if correct_a else 0,
            "pred": pred_a,
            "raw_text": ra["text"][:300],
            "compression_ratio": round(ratio_a, 4),
        }
        print(f"  compressed_pip: ratio={ratio_a*100:.1f}% tokens={tokens_a} "
              f"comp={comp_ms_a:.0f}+gem={ra['latency_ms']:.0f}ms "
              f"pred={pred_a!r} {'OK' if correct_a else 'X'}")
    except Exception as exc:
        print(f"  compressed_pip FAILED: {exc}", file=sys.stderr)
        metrics["compressed_pip"] = {"error": str(exc)}
    time.sleep(SLEEP_BETWEEN_CALLS)

    # --- compressed_lc ---
    try:
        compressed_b, comp_ms_b = compress_langchain(context, question, rate=RATE)
        prompt_lc = build_qa_prompt(compressed_b, question, choices)
        rl = call_gemini(model, prompt_lc)
        pred_l = normalize_pred(rl["text"])
        correct_l = (pred_l == gold)
        tokens_l = rl["input_tokens"] or (len(prompt_lc) // 4)
        ratio_l = len(compressed_b) / max(len(context), 1)
        unchanged_l = (abs(len(compressed_b) - len(context)) < 50)
        metrics["compressed_lc"] = {
            "input_tokens": tokens_l,
            "compressor_latency_ms": comp_ms_b,
            "gemini_latency_ms": rl["latency_ms"],
            "total_latency_ms": comp_ms_b + rl["latency_ms"],
            "accuracy": 1 if correct_l else 0,
            "pred": pred_l,
            "raw_text": rl["text"][:300],
            "compression_ratio": round(ratio_l, 4),
            "langchain_unchanged": unchanged_l,
        }
        print(f"  compressed_lc:  ratio={ratio_l*100:.1f}% tokens={tokens_l} "
              f"comp={comp_ms_b:.0f}+gem={rl['latency_ms']:.0f}ms "
              f"pred={pred_l!r} {'OK' if correct_l else 'X'}"
              f"{' (LC fell back to raw context)' if unchanged_l else ''}")
    except Exception as exc:
        print(f"  compressed_lc FAILED: {exc}", file=sys.stderr)
        metrics["compressed_lc"] = {"error": str(exc)}
    time.sleep(SLEEP_BETWEEN_CALLS)

    # Write EvalLog-style records
    for config_name, m in metrics.items():
        if "error" in m and m.get("input_tokens") is None:
            rec = {
                "ts": now_iso(),
                "track": "qa_mc",
                "case_id": case_id,
                "config": config_name,
                "status": "error",
                "error": m["error"],
                "gold": gold,
                "domain": domain,
                "n_chunks": n_chunks,
                "context_chars": len(context),
                "question_chars": len(question),
            }
        else:
            rec = {
                "ts": now_iso(),
                "track": "qa_mc",
                "case_id": case_id,
                "config": config_name,
                "status": "ok",
                "model": GEMINI_MODEL,
                "gold": gold,
                "domain": domain,
                "n_chunks": n_chunks,
                "context_chars": len(context),
                "question_chars": len(question),
                "input_tokens": m["input_tokens"],
                "output_tokens": m.get("output_tokens"),
                "compression_ratio": m["compression_ratio"],
                "compressor_latency_ms": round(m["compressor_latency_ms"], 1),
                "gemini_latency_ms": round(m["gemini_latency_ms"], 1),
                "total_latency_ms": round(m["total_latency_ms"], 1),
                "accuracy": m["accuracy"],
                "pred": m["pred"],
                "raw_pred_text": m["raw_text"],
            }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_fh.flush()
    return metrics


def aggregate(per_case: list[tuple[str, dict]]) -> dict:
    """Build summary across all 3 configs."""
    summary: dict = {"by_config": {}, "delta_vs_baseline": {}}

    for config in ("baseline", "compressed_pip", "compressed_lc"):
        tokens = []
        total_lat = []
        gem_lat = []
        comp_lat = []
        accs = []
        ratios = []
        n = 0
        for case_id, m in per_case:
            cm = m.get(config)
            if not cm or "error" in cm:
                continue
            if "input_tokens" in cm and cm["input_tokens"] is not None:
                tokens.append(cm["input_tokens"])
            if "total_latency_ms" in cm:
                total_lat.append(cm["total_latency_ms"])
            if "gemini_latency_ms" in cm:
                gem_lat.append(cm["gemini_latency_ms"])
            if "compressor_latency_ms" in cm:
                comp_lat.append(cm["compressor_latency_ms"])
            if "accuracy" in cm:
                accs.append(cm["accuracy"])
            if "compression_ratio" in cm:
                ratios.append(cm["compression_ratio"])
            n += 1

        def stat(xs):
            return round(statistics.mean(xs), 1) if xs else None

        summary["by_config"][config] = {
            "n_cases": n,
            "avg_input_tokens": stat(tokens),
            "avg_gemini_latency_ms": stat(gem_lat),
            "avg_compressor_latency_ms": stat(comp_lat),
            "avg_total_latency_ms": stat(total_lat),
            "accuracy": stat(accs),
            "avg_compression_ratio": stat(ratios),
        }

    # Delta so voi baseline
    b = summary["by_config"]["baseline"]
    for config in ("compressed_pip", "compressed_lc"):
        c = summary["by_config"][config]
        if b["avg_input_tokens"] and c["avg_input_tokens"]:
            token_saved_pct = (b["avg_input_tokens"] - c["avg_input_tokens"]) / b["avg_input_tokens"] * 100
        else:
            token_saved_pct = None
        if b["avg_total_latency_ms"] and c["avg_total_latency_ms"]:
            time_saved_pct = (b["avg_total_latency_ms"] - c["avg_total_latency_ms"]) / b["avg_total_latency_ms"] * 100
        else:
            time_saved_pct = None
        accuracy_delta_pp = (
            (c["accuracy"] - b["accuracy"]) * 100
            if c["accuracy"] is not None and b["accuracy"] is not None
            else None
        )
        summary["delta_vs_baseline"][config] = {
            "token_saved_pct": round(token_saved_pct, 1) if token_saved_pct is not None else None,
            "time_saved_pct": round(time_saved_pct, 1) if time_saved_pct is not None else None,
            "accuracy_delta_pp": round(accuracy_delta_pp, 1) if accuracy_delta_pp is not None else None,
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval LLMLingua on 5 cases")
    parser.add_argument("--n", type=int, default=5,
                        help=f"Number of cases (default 5)")
    parser.add_argument("--max-context-chars", type=int, default=200_000,
                        help="Skip rows whose context exceeds this many chars (default 200k)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"ERROR: {args.test_file} not found. Run setup_dataset.py first.",
              file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.test_file.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))  # nho truoc (it nhat chunk)

    selected = []
    skipped = []
    for r in rows:
        if len(selected) >= args.n:
            break
        if len(r.get("context", "")) > args.max_context_chars:
            skipped.append(r["_id"])
            continue
        selected.append(r)

    print("=== Eval LLMLingua 5 cases ===")
    print(f"  model:           {GEMINI_MODEL}")
    print(f"  rate / chunk:    {RATE} / {CHUNK_CHARS} chars")
    print(f"  cases selected:  {len(selected)}")
    print(f"  cases skipped:   {len(skipped)} (context > {args.max_context_chars:,} chars)")
    print(f"  out:             {args.out}")
    print(f"  summary:         {args.summary}")
    print()

    model = get_gemini_model()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    per_case: list[tuple[str, dict]] = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(selected):
            case_id = f"L5-{i:02d}-{row.get('_id', '?')[:6]}"
            m = run_one_case(case_id, row, model, fh)
            per_case.append((case_id, m))

    summary = aggregate(per_case)
    summary["meta"] = {
        "model": GEMINI_MODEL,
        "rate": RATE,
        "chunk_chars": CHUNK_CHARS,
        "n_selected": len(selected),
        "n_skipped_large": len(skipped),
        "skipped_ids": skipped,
        "ts": now_iso(),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print()
    print("=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Done. Records: {args.out}")
    print(f"Summary:      {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
