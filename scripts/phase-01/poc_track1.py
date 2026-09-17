#!/usr/bin/env python3
"""
Phase 1 — Track 1 PoC: LLMLingua-2 vs baseline on Gemini.

Paired run: for each of N cases from LongBench-v2, run baseline (full prompt) and
compressed (LLMLingua-2 via LangChain) → Gemini, log to JSONL.

Usage (from repo root):
    conda activate po5
    python scripts/phase-01/poc_track1.py --n 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    build_prompt_track1,
    call_gemini,
    count_tokens_approx,
    detect_fields,
    get_gemini_model,
    judge_answer_gemini,
    now_iso,
    stream_longbench_v2,
)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "phase-01-track1-poc.jsonl"
DEFAULT_N = 15


def get_compressor():
    """Build LangChain LLMLinguaCompressor configured for LLMLingua-2 (task-agnostic)."""
    try:
        from langchain_community.document_compressors import LLMLinguaCompressor
    except ImportError as exc:
        raise RuntimeError(
            "langchain-community chua duoc cai hoac khong co LLMLinguaCompressor. "
            "pip install langchain-community llmlingua"
        ) from exc

    return LLMLinguaCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetment",
        device_map="cpu",
        model_config={"revision": "main"},
    )


def compress_prompt(compressor, context: str, question: str | None = None) -> tuple[str, float]:
    """Run LLMLingua-2 compression. Returns (compressed_text, compressor_latency_ms)."""
    t0 = time.perf_counter()
    result = compressor.compress_documents(
        [type("Doc", (), {"page_content": context, "metadata": {}})()],
        query=question or "",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    compressed_text = result[0].page_content if result else ""
    return compressed_text, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 Track 1 PoC (LLMLingua-2)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of cases (default {DEFAULT_N})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL path (default {DEFAULT_OUT.name})")
    parser.add_argument("--judge-sample", type=int, default=5,
                        help="Number of cases to run LLM-as-judge on (default 5)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between Gemini calls (seconds)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    print(f"Track 1 PoC - LLMLingua-2 vs baseline")
    print(f"  n = {args.n}")
    print(f"  out = {args.out}")
    print(f"  judge_sample = {args.judge_sample}")
    print("---")

    model = get_gemini_model()
    print("Loading LLMLingua-2 compressor...")
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
            if not context or len(context) < 500:
                continue

            case_id = f"T1-C{rows_done:03d}"
            print(f"\n[{rows_done+1}/{args.n}] case_id={case_id} "
                  f"context_len={len(context)} chars")

            prompt_a = build_prompt_track1(context)
            result_a = call_gemini(model, prompt_a, max_tokens=512)
            rec_a = {
                "ts": now_iso(),
                "track": "track1_extraction",
                "case_id": case_id,
                "config": "A1_baseline",
                "compressor": "none",
                "input_tokens": result_a["input_tokens"],
                "output_tokens": result_a["output_tokens"],
                "compression_ratio": 1.0,
                "compressor_latency_ms": None,
                "gemini_latency_ms": round(result_a["latency_ms"], 1),
                "pred": {"answer_text": result_a["text"][:1000]},
                "ref": {"context_len": len(context)},
            }
            fh.write(json.dumps(rec_a, ensure_ascii=False) + "\n")
            records_written += 1
            print(f"  A1: {result_a['latency_ms']:.0f} ms, "
                  f"in={result_a['input_tokens']} out={result_a['output_tokens']}")

            time.sleep(args.sleep)

            try:
                compressed, comp_ms = compress_prompt(compressor, context)
                prompt_b = build_prompt_track1(compressed)
                result_b = call_gemini(model, prompt_b, max_tokens=512)
                ratio = (count_tokens_approx(compressed) /
                         max(1, count_tokens_approx(context)))
                rec_b = {
                    "ts": now_iso(),
                    "track": "track1_extraction",
                    "case_id": case_id,
                    "config": "B1_llmlingua2",
                    "compressor": "llmlingua-2",
                    "input_tokens": result_b["input_tokens"],
                    "output_tokens": result_b["output_tokens"],
                    "compression_ratio": round(ratio, 3),
                    "compressor_latency_ms": round(comp_ms, 1),
                    "gemini_latency_ms": round(result_b["latency_ms"], 1),
                    "pred": {"answer_text": result_b["text"][:1000]},
                    "ref": {"context_len": len(context),
                            "compressed_len": len(compressed)},
                }
                fh.write(json.dumps(rec_b, ensure_ascii=False) + "\n")
                records_written += 1
                print(f"  B1: ratio={ratio:.2f}, comp={comp_ms:.0f} ms, "
                      f"gemini={result_b['latency_ms']:.0f} ms")
            except Exception as exc:
                print(f"  B1 FAILED: {exc}", file=sys.stderr)

            if rows_done < args.judge_sample:
                gold = fields["answer"] or fields["choice"] or ""
                if gold:
                    judge_a = judge_answer_gemini(model, "(extraction task)", gold, result_a["text"])
                    judge_rec = {
                        "ts": now_iso(),
                        "track": "track1_extraction",
                        "case_id": case_id,
                        "config": "judge_baseline",
                        "score": judge_a,
                    }
                    fh.write(json.dumps(judge_rec, ensure_ascii=False) + "\n")
                    records_written += 1
                    print(f"  judge(A1): correct={judge_a['judge_correct']}")

            fh.flush()
            rows_done += 1
            time.sleep(args.sleep)

    print(f"\nDone. Wrote {records_written} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
