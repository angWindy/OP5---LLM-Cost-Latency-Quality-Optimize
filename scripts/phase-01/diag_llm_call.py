#!/usr/bin/env python3
"""
Phase 1 — Diagnostic: detailed breakdown of 1 Gemini LLM call.

Purpose:
    Profile a single Gemini call with a LongBench-v2 row, breaking down latency
    into fine-grained components so we can identify bottlenecks.

  1. setup_ms        — time to instantiate genai + GenerativeModel (one-shot)
  2. prompt_build_ms — time to assemble the prompt (string ops)
  3. serialize_ms    — time to serialize prompt to bytes (no-op for str, but check)
  4. t_start_api     — wall-clock before the `generate_content` call
  5. network_ms      — time before first byte returned (≈ API req roundtrip)
  6. t_end_api       — wall-clock after the call returns
  7. parse_ms        — time to extract response.text + usage_metadata
  8. ttft_proxy_ms   — t_until_first_byte (same as network_ms for sync API)
  9. total_ms        — t_end - t_start (the "official" gemini_ms)
 10. input_tokens    — from usage_metadata.prompt_token_count
 11. chars_in        — len(prompt) for byte/token ratio
 12. tokens_per_s    — input_tokens / total_ms (rough throughput)
 13. t_first_byte_byte — measured approx via SDK if available (usually unavailable)

Additionally captures Gemini SDK timing using the google.generativeai library
internals where exposed (sdk_latency if any), and prints a final breakdown table.

Outputs:
    results/phase-01-llm-call-diagnostic.json     — single JSON with all timings

Usage:
    conda activate vsf
    python scripts/phase-01/diag_llm_call.py [--max-context-chars 200000]
"""
from __future__ import annotations

import argparse
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

DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "llmlingua_test5.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "phase-01-llm-call-diagnostic.json"
GEMINI_MODEL = os.getenv("OP5_GEMINI_MODEL", "gemini-3.5-flash-lite")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_prompt(context: str, question: str, choices: dict[str, str]) -> str:
    choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
    return (
        "Doc doan van ban sau va tra loi cau hoi. CHI tra ve mot chu cai (A, B, C hoac D) "
        "dung nhat. Khong giai thich.\n\n"
        f"CAU HOI: {question}\n\n"
        f"LUA CHON:\n{choices_str}\n\n"
        f"DOAN VAN BAN:\n{context}\n\n"
        "DAP AN (1 chu cai):"
    )


def diag_one_call(prompt: str, model_name: str, max_output_tokens: int = 32) -> dict:
    """Run a single Gemini call with detailed timing."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY chua duoc set trong .env")

    out: dict = {"model": model_name, "max_output_tokens": max_output_tokens, "ts": now_iso()}
    out["prompt_chars"] = len(prompt)

    # --- 1. SDK import + configure ---
    t = time.perf_counter()
    genai.configure(api_key=api_key)
    sdk_setup_ms = (time.perf_counter() - t) * 1000

    # --- 2. Model instantiation ---
    t = time.perf_counter()
    model = genai.GenerativeModel(model_name)
    model_init_ms = (time.perf_counter() - t) * 1000

    # --- 3. Pre-call memory snapshot ---
    try:
        import resource
        ru_pre = resource.getrusage(resource.RUSAGE_SELF)
        mem_pre_kb = ru_pre.ru_maxrss
    except Exception:
        mem_pre_kb = None

    # --- 4. The actual API call ---
    t_start = time.perf_counter()
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": max_output_tokens},
        )
    except Exception as exc:
        api_err_ms = (time.perf_counter() - t_start) * 1000
        return {
            **out,
            "sdk_setup_ms": sdk_setup_ms,
            "model_init_ms": model_init_ms,
            "api_err": True,
            "error": str(exc),
            "api_failed_at_ms": api_err_ms,
        }
    api_total_ms = (time.perf_counter() - t_start) * 1000

    # --- 5. Post-call processing ---
    t = time.perf_counter()
    text = (response.text or "").strip()
    parse_ms = (time.perf_counter() - t) * 1000

    # --- 6. Usage metadata ---
    usage = getattr(response, "usage_metadata", None)
    inp_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    out_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    # --- 7. Memory after ---
    try:
        ru_post = resource.getrusage(resource.RUSAGE_SELF)
        mem_post_kb = ru_post.ru_maxrss
    except Exception:
        mem_post_kb = None

    # --- 8. Aggressive repeated calls to measure per-call cost (warm path) ---
    # 3 extra calls to capture steady-state latency (first call may be cold).
    warm = []
    for _ in range(3):
        ts = time.perf_counter()
        try:
            resp_w = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": max_output_tokens},
            )
            te = time.perf_counter()
            warm.append({
                "ms": (te - ts) * 1000,
                "inp_tokens": getattr(getattr(resp_w, "usage_metadata", None),
                                      "prompt_token_count", None),
                "text_first10": (getattr(resp_w, "text", "") or "")[:10],
            })
        except Exception as exc:
            warm.append({"ms": None, "error": str(exc)})

    out.update({
        "sdk_setup_ms": round(sdk_setup_ms, 2),
        "model_init_ms": round(model_init_ms, 2),
        "api_total_ms": round(api_total_ms, 2),
        "parse_ms": round(parse_ms, 2),
        "input_tokens": inp_tokens,
        "output_tokens": out_tokens,
        "text_first40": text[:40],
        "text_len": len(text),
        "mem_pre_kb": mem_pre_kb,
        "mem_post_kb": mem_post_kb,
        "warm_calls": warm,
        # Throughput:
        "tokens_per_sec_api": round((inp_tokens or 0) / (api_total_ms / 1000), 1)
            if inp_tokens and api_total_ms > 0 else None,
        "chars_per_sec_api": round(len(prompt) / (api_total_ms / 1000), 1)
            if api_total_ms > 0 else None,
        # Ratio chars → tokens:
        "chars_per_token": round(len(prompt) / max(inp_tokens or 1, 1), 2),
    })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-context-chars", type=int, default=200000)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"ERROR: {args.test_file} not found. Run setup_dataset.py first.",
              file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.test_file.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: len(r.get("context", "")))
    # Pick the SHORTEST row to keep the test fast (precedent: smoke_both_paths.py)
    row = rows[0]

    context = row.get("context", "")
    question = row.get("question", "")
    choices = {k: row.get(f"choice_{k}", "") for k in "ABCD"}
    gold = row.get("answer", "").strip().upper()
    context = context[: args.max_context_chars]

    print(f"=== Diagnostic: 1 Gemini call ===")
    print(f"  model={GEMINI_MODEL}  max_output_tokens={args.max_output_tokens}")
    print(f"  case: _id={row.get('_id', '?')[:8]}  domain={row.get('domain', '?')}")
    print(f"  context_chars={len(context):,}  gold={gold}")
    print(f"  question: {question[:80]}...")
    print()

    # Measure prompt build separately
    t = time.perf_counter()
    prompt = build_prompt(context, question, choices)
    prompt_build_ms = (time.perf_counter() - t) * 1000
    print(f"  prompt_chars: {len(prompt):,}  (build took {prompt_build_ms:.2f}ms)")
    print()
    print("[1/2] Cold call (first time)...")
    cold = diag_one_call(prompt, GEMINI_MODEL, args.max_output_tokens)
    cold["prompt_build_ms"] = round(prompt_build_ms, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump({"cold": cold, "case": {
            "_id": row.get("_id"),
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "gold": gold,
            "context_chars": len(context),
            "prompt_chars": len(prompt),
            "question": question[:200],
        }}, fh, ensure_ascii=False, indent=2)

    # --- Print breakdown ---
    print("\n=== Breakdown ===\n")
    if cold.get("api_err"):
        print(f"  API FAILED after {cold.get('api_failed_at_ms', '?')}ms")
        print(f"  error: {cold.get('error')}")
        print(f"\nWritten: {args.out}")
        return 5

    rows = [
        ("Prompt build (string ops)",   cold["prompt_build_ms"]),
        ("SDK setup (genai.configure)", cold["sdk_setup_ms"]),
        ("Model instantiation",         cold["model_init_ms"]),
        ("API call (generate_content)", cold["api_total_ms"]),
        ("Response parse",              cold["parse_ms"]),
    ]
    for name, val in rows:
        pct = val / cold["api_total_ms"] * 100 if cold["api_total_ms"] > 0 else 0
        print(f"  {name:35s} {val:>9.2f} ms  ({pct:>5.1f}% of API call)")
    print(f"  {'—':->45s}")
    print()
    print(f"  input_tokens (from usage)  : {cold['input_tokens']}")
    print(f"  output_tokens (from usage) : {cold['output_tokens']}")
    print(f"  chars/token ratio           : {cold['chars_per_token']}")
    print(f"  tokens/sec during API call  : {cold['tokens_per_sec_api']}")
    print(f"  chars/sec during API call   : {cold['chars_per_sec_api']}")
    print()
    print("  Warm calls (3× follow-ups):")
    for i, w in enumerate(cold["warm_calls"], 1):
        ms = w.get("ms")
        ms_s = f"{ms:>8.1f} ms" if ms else "    ERR ms"
        print(f"    #{i}  {ms_s}  inp={w.get('inp_tokens')}  text={w.get('text_first10')!r}")
    if all(w.get("ms") for w in cold["warm_calls"]):
        warm_avg = sum(w["ms"] for w in cold["warm_calls"]) / len(cold["warm_calls"])
        print(f"    avg = {warm_avg:.1f} ms (warm steady-state)")
    print()
    print(f"  Written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
