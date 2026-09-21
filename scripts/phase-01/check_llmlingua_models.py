#!/usr/bin/env python3
"""
Check what LLMLingua models are available and benchmark the original
(microsoft/llmlingua) vs LLMLingua-2 on 1 case.

Runs in ~2 min on CPU.
"""
from __future__ import annotations

import json
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

DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-llmlingua-model-check.json"


def load_shortest_case():
    test_path = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found -- run setup_dataset.py first")
    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    return rows[0]


def test_model(
    model_name: str,
    use_llmlingua2: bool,
    context: str,
    question: str,
    rate: float = 0.5,
) -> dict:
    from llmlingua import PromptCompressor

    t0 = time.perf_counter()
    try:
        pc = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=use_llmlingua2,
            device_map="cpu",
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return {
            "model_name": model_name,
            "use_llmlingua2": use_llmlingua2,
            "rate": rate,
            "status": "load_error",
            "error": str(exc),
        }

    t1 = time.perf_counter()
    try:
        result = pc.compress_prompt(
            context,
            question=question,
            rate=rate,
            force_tokens=["!", ".", "?", "\n"],
            drop_consecutive=True,
            return_word_label=False,
        )
        compress_ms = (time.perf_counter() - t1) * 1000.0
        compressed = result.get("compressed_prompt", "")
        origin_tok = result.get("origin_tokens", 0)
        compressed_tok = result.get("compressed_tokens", 0)
        ratio_str = result.get("ratio", "")
        char_ratio = len(compressed) / max(len(context), 1)

        return {
            "model_name": model_name,
            "use_llmlingua2": use_llmlingua2,
            "rate": rate,
            "status": "ok",
            "load_ms": round(load_ms, 1),
            "compress_ms": round(compress_ms, 1),
            "total_ms": round(load_ms + compress_ms, 1),
            "origin_tokens": origin_tok,
            "compressed_tokens": compressed_tok,
            "token_ratio": round(compressed_tok / max(origin_tok, 1), 4),
            "char_ratio": round(char_ratio, 4),
            "ratio_str": ratio_str,
            "origin_chars": len(context),
            "compressed_chars": len(compressed),
            "chars_per_sec": round(len(context) / max(compress_ms / 1000.0, 0.001)),
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "use_llmlingua2": use_llmlingua2,
            "rate": rate,
            "status": "compress_error",
            "load_ms": round(load_ms, 1),
            "error": str(exc),
        }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    row = load_shortest_case()
    context = row.get("context", "")
    question = row.get("question", "")
    gold = row.get("answer", "")

    print("=== LLMLingua model check ===")
    print(f"  case:     {row.get('_id','?')} ({row.get('domain','?')})")
    print(f"  context:  {len(context):,} chars  (~{len(context)//4:,} tokens)")
    print(f"  question: {question[:80]}...")
    print(f"  gold:     {gold}")
    print()

    models_to_test = [
        ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", True,  "LLMLingua-2 560M (known-good)"),
        ("microsoft/llmlingua",                                False, "LLMLingua original (unknown backbone)"),
        ("microsoft/LLMLingua-Llama-7b-4bit",                  False, "LLMLingua 7B 4bit (small)"),
    ]

    results = []
    for model_name, use_llm2, desc in models_to_test:
        print(f"--- Testing: {desc} ---")
        r = test_model(model_name, use_llm2, context, question, rate=0.5)
        r["description"] = desc
        results.append(r)

        if r["status"] == "ok":
            print(f"  status:    OK")
            print(f"  load:     {r['load_ms']:.0f} ms")
            print(f"  compress: {r['compress_ms']:.0f} ms")
            print(f"  total:    {r['total_ms']:.0f} ms")
            print(f"  chars/s:  {r['chars_per_sec']}")
            print(f"  ratio:    {r['ratio_str']} (token), {r['char_ratio']:.3f} (char)")
            print(f"  tokens:   {r['origin_tokens']} -> {r['compressed_tokens']}")
        elif r["status"] == "load_error":
            print(f"  status:   LOAD_ERROR -- {r.get('error','?')}")
        else:
            print(f"  status:   {r['status']} -- {r.get('error','?')}")
        print()

    print("=== Summary ===")
    valid = [r for r in results if r["status"] == "ok"]
    if valid:
        valid.sort(key=lambda r: r["total_ms"])
        for r in valid:
            saved_pct = round((1 - r["char_ratio"]) * 100, 1)
            print(
                f"  {r['description']:<50s} "
                f"total={r['total_ms']:>7.0f}ms  "
                f"chars_saved={saved_pct:>5.1f}%  "
                f"c/s={r['chars_per_sec']:>5.0f}"
            )

    out_data = {
        "case_id": row.get("_id", ""),
        "context_chars": len(context),
        "question_chars": len(question),
        "gold": gold,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
