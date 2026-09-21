#!/usr/bin/env python3
"""
Phase 1 — Phân rã 186 giây compressor thành các giai đoạn nhỏ.

LLMLingua-2 chạy 186s trên context 70k chars với iter_size=200. Trong đó:
  - Load model:                ~23s (đã biết từ ablation run trước)
  - Tokenization (BPE):        ?s
  - Forward pass per chunk:    ?s × ? chunks
  - Sentence selection (rank): ?s
  - Decode + reconstruct:      ?s

Cách đo: monkey-patch vào các method nội bộ của PromptCompressor để time từng
giai đoạn. Log ra từng stage.

Mục tiêu: tìm bottleneck để biết nên tối ưu chỗ nào.

Usage:
    conda activate vsf
    python scripts/phase-01/compressor_stage_profile.py
"""
from __future__ import annotations

import json
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
from smoke_both_paths import _get_llmlingua2, COMPRESSOR_MODEL


DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-compressor-stages.json"


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StageTimer:
    """Profile context manager for per-stage timing."""
    def __init__(self):
        self.stages = []
        self._t = None
        self._name = None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self._t) * 1000
        self.stages.append({"stage": self._name, "ms": round(elapsed_ms, 2)})


def patch_pc_for_profiling(pc, timer):
    """
    Monkey-patch key methods of PromptCompressor to record per-stage timing.
    Target methods (best guess based on llmlingua source):
      - pc.model.tokenize(...)           -> per-chunk tokenization
      - pc._compress(...)                -> main loop
      - pc.model(...)                    -> forward pass (hard to patch directly)
    """
    # Try to find the tokenizer + model
    model_obj = pc.model  # AutoModelForTokenClassification
    tokenizer = pc.tokenizer

    # Wrap tokenize
    orig_tokenize = tokenizer.tokenize

    def timed_tokenize(*args, **kwargs):
        with timer("tokenize"):
            return orig_tokenize(*args, **kwargs)

    tokenizer.tokenize = timed_tokenize

    # Wrap __call__ of model to count forward passes
    forward_count = [0]
    forward_total_ms = [0.0]
    orig_call = model_obj.__class__.__call__

    def timed_call(self, *args, **kwargs):
        t = time.perf_counter()
        r = orig_call(self, *args, **kwargs)
        forward_total_ms[0] += (time.perf_counter() - t) * 1000
        forward_count[0] += 1
        return r

    model_obj.__class__.__call__ = timed_call

    return forward_count, forward_total_ms


def main():
    rows = [json.loads(l) for l in DEFAULT_TEST.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    row = rows[0]
    context = row.get("context", "")[:80000]
    question = row.get("question", "")
    case_id = row.get("_id", "?")

    print(f"=== Compressor stage profile (1 case, iter_size=200) ===")
    print(f"  case: _id={case_id[:8]}  context_chars={len(context):,}")
    print()

    pc = _get_llmlingua2()
    timer = StageTimer()
    fwd_count, fwd_ms = patch_pc_for_profiling(pc, timer)

    print("Running compress_prompt(iter_size=200, rate=0.5, ft_basic)...")
    t_total = time.perf_counter()
    try:
        result = pc.compress_prompt(
            context,
            question=question,
            rate=0.5,
            force_tokens=["!", ".", "?", "\n"],
            drop_consecutive=True,
            return_word_label=False,
            iterative_size=200,
        )
        total_ms = (time.perf_counter() - t_total) * 1000
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    # Analyse
    origin = result.get("origin_tokens") or 0
    compressed = result.get("compressed_tokens") or 0

    # Total stages captured
    tokenize_total = sum(s["ms"] for s in timer.stages if s["stage"] == "tokenize")
    tokenize_calls = sum(1 for s in timer.stages if s["stage"] == "tokenize")

    accounted = tokenize_total + fwd_ms[0]
    other = total_ms - accounted

    print()
    print("=== Breakdown ===")
    print(f"  Total:                 {total_ms:.0f} ms")
    print(f"  Tokenize calls:        {tokenize_calls}  total {tokenize_total:.0f} ms")
    print(f"    avg per tokenize:    {tokenize_total/max(tokenize_calls,1):.1f} ms")
    print(f"  Forward passes:        {fwd_count[0]}  total {fwd_ms[0]:.0f} ms")
    print(f"    avg per forward:     {fwd_ms[0]/max(fwd_count[0],1):.1f} ms")
    print(f"  Other (rank/decode):   {other:.0f} ms")
    print(f"  Accounted:             {accounted:.0f} / {total_ms:.0f} ms ({accounted/total_ms*100:.1f}%)")
    print()
    print(f"  Tokens: {origin} -> {compressed}  (ratio {result.get('ratio')})")
    print(f"  chunks: {origin / 200:.0f} estimated (origin_tokens / iter_size)")
    print()

    # Save
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUT.open("w", encoding="utf-8") as fh:
        json.dump({
            "ts": now_iso(),
            "model": COMPRESSOR_MODEL,
            "case": {"_id": case_id, "context_chars": len(context)},
            "config": {"iter_size": 200, "rate": 0.5, "ft": ["!", ".", "?", "\n"]},
            "totals": {
                "total_ms": round(total_ms, 2),
                "tokenize_total_ms": round(tokenize_total, 2),
                "tokenize_calls": tokenize_calls,
                "forward_total_ms": round(fwd_ms[0], 2),
                "forward_count": fwd_count[0],
                "other_ms": round(other, 2),
            },
            "stage_log": timer.stages,
            "result": {
                "origin_tokens": origin,
                "compressed_tokens": compressed,
                "ratio": result.get("ratio"),
            },
        }, fh, ensure_ascii=False, indent=2)

    # Bottleneck
    print("=== Verdict ===")
    if fwd_ms[0] > 0.7 * total_ms:
        print(f"  Forward passes dominate ({fwd_ms[0]/total_ms*100:.0f}% of total)")
        print(f"  -> iter_size=200 forces ~{origin/200:.0f} sequential forward passes on CPU")
        print(f"  -> Move to GPU (device_map='cuda') OR increase iter_size")
    elif other > 0.5 * total_ms:
        print(f"  Post-processing dominates ({other/total_ms*100:.0f}%)")
        print(f"  -> Likely the rank/sort step or sentence reconstruction")
    else:
        print(f"  Mixed bottlenecks. See JSON for detail.")
    print(f"\nWritten: {DEFAULT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
