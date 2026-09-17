#!/usr/bin/env python3
"""
Phase 1 - Track 2 PoC: LongLLMLingua vs baseline on Gemini.

Paired run: for each of N cases from LongBench-v2, run baseline (full context + question)
and compressed (LongLLMLingua via LangChain, question-aware) -> Gemini, log to JSONL.

LongLLMLingua preserves passages relevant to the query -> mitigates "lost in the middle"
and matches lever C in the master plan for Track 2.

Usage (from repo root):
    conda activate po5
    python scripts/phase-01/poc_track2.py --n 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    build_prompt_track2,
    call_gemini,
    count_tokens_approx,
    detect_fields,
    get_gemini_model,
    judge_answer_gemini,
    now_iso,
    stream_longbench_v2,
    stream_squad_v2,
)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "phase-01-track2-poc.jsonl"
DEFAULT_N = 15


def get_compressor():
    """Build LangChain LLMLinguaCompressor configured for LongLLMLingua (query-aware)."""
    try:
        from langchain_community.document_compressors import LLMLinguaCompressor
    except ImportError as exc:
        raise RuntimeError(
            "langchain-community chua duoc cai hoac khong co LLMLinguaCompressor. "
            "pip install langchain-community llmlingua"
        ) from exc

    # LongLLMLingua uses the original LLMLingua model (not llmlingua-2) because
    # it relies on perplexity-based scoring with the question as context.
    return LLMLinguaCompressor(
        model_name="microsoft/llmlingua",
        device_map="cpu",
        model_config={"revision": "main"},
        open_api_config={
            "api_base": "",
            "api_key": "",
        },
    )


def compress_prompt(compressor, context: str, question: str) -> tuple[str, float]:
    """Run LongLLMLingua compression, question-aware. Returns (compressed_text, compressor_latency_ms)."""
    t0 = time.perf_counter()
    result = compressor.compress_documents(
        [type("Doc", (), {"page_content": context, "metadata": {}})()],
        query=question,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    compressed_text = result[0].page_content if result else ""
    return compressed_text, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 Track 2 PoC (LongLLMLingua)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of cases (default {DEFAULT_N})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL path (default {DEFAULT_OUT.name})")
    parser.add_argument("--judge-all", action="store_true",
                        help="Run LLM-as-judge on every case (default: first 5)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between Gemini calls (seconds)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    judge_sample = args.n if args.judge_all else 5

    print(f"Track 2 PoC - LongLLMLingua vs baseline")
    print(f"  n = {args.n}")
    print(f"  out = {args.out}")
    print(f"  judge_sample = {judge_sample}")
    print("---")

    model = get_gemini_model()
    print("Loading LongLLMLingua compressor...")
    try:
        compressor = get_compressor()
    except Exception as exc:
        print(f"FATAL: Khong load duoc compressor: {exc}", file=sys.stderr)
        return 2

    print(f"Streaming LongBench-v2...")
    rows_done = 0
    records_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(stream_longbench_v2()):
            if rows_done >= args.n:
                break
            fields = detect_fields(row)
            context = fields["context"]
            question = fields["question"]
            if not context or not question or len(context) < 500:
                continue

            case_id = f"T2-C{rows_done:03d}"
            print(f"\n[{rows_done+1}/{args.n}] case_id={case_id} "
                  f"context_len={len(context)} chars, q_len={len(question)} chars")

            # --- A2: baseline ---
            prompt_a = build_prompt_track2(context, question)
            result_a = call_gemini(model, prompt_a, max_tokens=512)
            rec_a = {
                "ts": now_iso(),
                "track": "track2_rag_qa",
                "case_id": case_id,
                "config": "A2_baseline",
                "compressor": "none",
                "input_tokens": result_a["input_tokens"],
                "output_tokens": result_a["output_tokens"],
                "compression_ratio": 1.0,
                "compressor_latency_ms": None,
                "gemini_latency_ms": round(result_a["latency_ms"], 1),
                "pred": {"answer_text": result_a["text"][:1000]},
                "ref": {"context_len": len(context),
                        "question": question[:300],
                        "gold_answer": fields["answer"][:300] if fields["answer"] else "",
                        "gold_choice": fields["choice"][:300] if fields["choice"] else ""},
            }
            fh.write(json.dumps(rec_a, ensure_ascii=False) + "\n")
            records_written += 1
            print(f"  A2: {result_a['latency_ms']:.0f} ms, "
                  f"in={result_a['input_tokens']} out={result_a['output_tokens']}")

            time.sleep(args.sleep)

            # --- C2: LongLLMLingua ---
            try:
                compressed, comp_ms = compress_prompt(compressor, context, question)
                prompt_c = build_prompt_track2(compressed, question)
                result_c = call_gemini(model, prompt_c, max_tokens=512)
                ratio = (count_tokens_approx(compressed) /
                         max(1, count_tokens_approx(context)))
                rec_c = {
                    "ts": now_iso(),
                    "track": "track2_rag_qa",
                    "case_id": case_id,
                    "config": "C2_longllmlingua",
                    "compressor": "longllmlingua",
                    "input_tokens": result_c["input_tokens"],
                    "output_tokens": result_c["output_tokens"],
                    "compression_ratio": round(ratio, 3),
                    "compressor_latency_ms": round(comp_ms, 1),
                    "gemini_latency_ms": round(result_c["latency_ms"], 1),
                    "pred": {"answer_text": result_c["text"][:1000]},
                    "ref": {"context_len": len(context),
                            "compressed_len": len(compressed),
                            "question": question[:300]},
                }
                fh.write(json.dumps(rec_c, ensure_ascii=False) + "\n")
                records_written += 1
                print(f"  C2: ratio={ratio:.2f}, comp={comp_ms:.0f} ms, "
                      f"gemini={result_c['latency_ms']:.0f} ms")
            except Exception as exc:
                print(f"  C2 FAILED: {exc}", file=sys.stderr)

            # LLM-as-judge on sample, comparing A2 vs C2 against gold
            if rows_done < judge_sample:
                gold = fields["answer"] or fields["choice"] or ""
                if gold:
                    judge_a = judge_answer_gemini(model, question, gold, result_a["text"])
                    fh.write(json.dumps({
                        "ts": now_iso(),
                        "track": "track2_rag_qa",
                        "case_id": case_id,
                        "config": "judge_baseline",
                        "score": judge_a,
                    }, ensure_ascii=False) + "\n")
                    records_written += 1
                    print(f"  judge(A2): correct={judge_a['judge_correct']}")

                    try:
                        judge_c = judge_answer_gemini(model, question, gold, result_c["text"])
                        fh.write(json.dumps({
                            "ts": now_iso(),
                            "track": "track2_rag_qa",
                            "case_id": case_id,
                            "config": "judge_compressed",
                            "score": judge_c,
                        }, ensure_ascii=False) + "\n")
                        records_written += 1
                        print(f"  judge(C2): correct={judge_c['judge_correct']}")
                    except Exception:
                        pass

            fh.flush()
            rows_done += 1
            time.sleep(args.sleep)

    print(f"\nDone. Wrote {records_written} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
