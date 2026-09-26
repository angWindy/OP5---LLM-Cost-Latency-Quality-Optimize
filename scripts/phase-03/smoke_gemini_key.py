"""One-shot smoke test for the configured Gemini API key.

Per AGENTS.md, this script is allowed to make 1-2 LLM calls just to verify
the key is alive before kicking off a long batch run. Exit code:

  0 = at least one key returned status='ok' (good to run --llm gemini).
  1 = all keys failed (fall back to --llm stub).
  2 = no keys configured.

Usage:
    conda activate vsf
    python scripts/phase-03/smoke_gemini_key.py
    python scripts/phase-03/smoke_gemini_key.py --model gemini-3.5-flash-lite --prompt "ping"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.llm.gemini_client import GeminiClient  # noqa: E402


def _try_collect_keys() -> list[tuple[str, str]]:
    """Same logic as GeminiClient._load_key_pool but exposed without instantiation."""
    multi = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        return [(f"k{i+1}", k) for i, k in enumerate(keys)]
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single:
        return [("k1", single)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--prompt", default="ping (reply 'pong')")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    keys = _try_collect_keys()
    if not keys:
        print("ERROR: no GOOGLE_API_KEY(S) configured. Set one and retry.", file=sys.stderr)
        return 2

    print(f"Found {len(keys)} key(s) in env. Probing {args.model} ...")
    try:
        client = GeminiClient(timeout_s=args.timeout, global_retries=len(keys))
    except RuntimeError as exc:
        print(f"ERROR: GeminiClient init failed: {exc}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    result = client.generate(args.prompt, model=args.model, max_output_tokens=16)
    elapsed = (time.perf_counter() - t0) * 1000

    status = result.get("status")
    print(f"  status:       {status}")
    print(f"  attempts:     {result.get('attempts')}")
    print(f"  key_id:       {result.get('key_id')}")
    print(f"  latency_ms:   {result.get('latency_ms'):.1f}")
    print(f"  input_tokens: {result.get('input_tokens')}")
    print(f"  output_tokens:{result.get('output_tokens')}")
    print(f"  wallclock_ms: {elapsed:.1f}")
    print(f"  text (first 80 chars): {(result.get('text') or '')[:80]!r}")
    if status != "ok":
        print(f"  error:        {result.get('error')}", file=sys.stderr)
        print(f"\nPool stats: {client.stats_summary()}")
        return 1

    print(f"\nOK: Gemini is reachable via key_id={result.get('key_id')}. Ready for --llm gemini.")
    print(f"Pool stats: {client.stats_summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
