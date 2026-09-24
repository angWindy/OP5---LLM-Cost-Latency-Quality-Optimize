"""
gemini_client.py — Google Gemini Generate Content client with API-key rotation.

Single responsibility:
  - Hold a pool of API keys (1..N)
  - Round-robin through them on success
  - Rotate immediately (no long backoff) on 429 / 503 / network timeout
  - Return the first successful response, record which key was used

The rotation strategy is "fast rotation, short retry":
  - On 429/503: rotate the key, retry once, give up quickly.
  - On Timeout/ConnectionError: rotate + retry, total wallclock budget < 60s.
  - On 4xx (other than 429): do NOT rotate, return error immediately.

This trades a bit of throughput for not stalling on a single dead key.

Usage:
  conda activate vsf

  from op5.llm.gemini_client import GeminiClient
  from dotenv import load_dotenv
  load_dotenv()

  client = GeminiClient()
  result = client.generate("Summarize: ...", model="gemini-3.5-flash-lite")
  # result = {"status": "ok"|"error", "text": ..., "latency_ms": ...,
  #           "input_tokens": ..., "output_tokens": ..., "key_id": ...,
  #           "attempts": ...}

Key pool sources (in priority order):
  1. GOOGLE_API_KEYS env var, comma-separated
  2. GOOGLE_API_KEY env var, single key
"""
from __future__ import annotations

import os
import time

import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_MAX_RETRIES_PER_KEY = 1
DEFAULT_GLOBAL_RETRIES = 4  # total attempts across keys


def _load_key_pool() -> list[tuple[str, str]]:
    """Return [(key_id, key_value), ...] from env.

    `key_id` is a short label used in logs: 'k1', 'k2', ...
    """
    multi = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        return [(f"k{i+1}", k) for i, k in enumerate(keys)]
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single:
        return [("k1", single)]
    return []


class GeminiClient:
    """Thin client around `generativelanguage.googleapis.com` with key rotation.

    Holds no Gemini SDK dependency on purpose — we talk HTTP directly,
    matching what `run_baseline.py` / `run_compressed.py` already do.

    State: tracks per-key cumulative success/error counters (printed on close).
    """

    def __init__(
        self,
        timeout_s: float = 120.0,
        max_retries_per_key: int = DEFAULT_MAX_RETRIES_PER_KEY,
        global_retries: int = DEFAULT_GLOBAL_RETRIES,
        sleep_between_keys_s: float = 1.0,
    ) -> None:
        self.pool = _load_key_pool()
        if not self.pool:
            raise RuntimeError(
                "No API key configured. Set GOOGLE_API_KEYS (comma-separated) "
                "or GOOGLE_API_KEY in env."
            )
        self._idx = 0  # next key to try
        self.timeout_s = timeout_s
        self.max_retries_per_key = max_retries_per_key
        self.global_retries = global_retries
        self.sleep_between_keys_s = sleep_between_keys_s
        # Stats
        self.success_by_key = {kid: 0 for kid, _ in self.pool}
        self.errors_by_key = {kid: 0 for kid, _ in self.pool}
        self.last_used_key = None

    @property
    def n_keys(self) -> int:
        return len(self.pool)

    def _peek_key(self) -> tuple[str, str]:
        kid, kval = self.pool[self._idx]
        return kid, kval

    def _advance(self) -> None:
        self._idx = (self._idx + 1) % len(self.pool)

    def rotate_now(self) -> None:
        """Force-rotate to next key without calling it. Useful before a known risky call."""
        self._advance()

    def _record(self, kid: str, ok: bool) -> None:
        if ok:
            self.success_by_key[kid] += 1
        else:
            self.errors_by_key[kid] += 1
        self.last_used_key = kid

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict:
        """Generate one completion. Returns dict (see module docstring)."""
        if not self.pool:
            return {"status": "error", "error": "no_keys", "latency_ms": 0,
                    "attempts": 0, "key_id": None}

        attempt = 0
        last_error: dict | None = None
        while attempt < self.global_retries:
            kid, kval = self._peek_key()
            for _ in range(self.max_retries_per_key):
                if attempt >= self.global_retries:
                    break
                attempt += 1
                url = GEMINI_URL.format(model=model) + f"?key={kval}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_output_tokens,
                    },
                }
                t0 = time.perf_counter()
                try:
                    resp = requests.post(
                        url, json=payload, timeout=self.timeout_s
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                            .strip()
                        )
                        usage = data.get("usageMetadata", {})
                        self._record(kid, ok=True)
                        self._advance()
                        return {
                            "status": "ok",
                            "text": text,
                            "latency_ms": elapsed_ms,
                            "input_tokens": usage.get("promptTokenCount"),
                            "output_tokens": usage.get("candidatesTokenCount"),
                            "key_id": kid,
                            "attempts": attempt,
                        }
                    if resp.status_code in (429, 503):
                        # rate-limited / overloaded — rotate immediately
                        self._record(kid, ok=False)
                        last_error = {
                            "status": "error",
                            "error": f"HTTP {resp.status_code}",
                            "latency_ms": elapsed_ms,
                            "key_id": kid,
                            "attempts": attempt,
                        }
                        self._advance()
                        kid, kval = self._peek_key()
                        time.sleep(self.sleep_between_keys_s)
                        continue
                    # Other 4xx — give up, don't rotate
                    self._record(kid, ok=False)
                    return {
                        "status": "error",
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "latency_ms": elapsed_ms,
                        "key_id": kid,
                        "attempts": attempt,
                    }
                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._record(kid, ok=False)
                    last_error = {
                        "status": "error",
                        "error": f"{type(exc).__name__}",
                        "latency_ms": elapsed_ms,
                        "key_id": kid,
                        "attempts": attempt,
                    }
                    self._advance()
                    kid, kval = self._peek_key()
                    time.sleep(self.sleep_between_keys_s)
                    continue
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._record(kid, ok=False)
                    return {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": elapsed_ms,
                        "key_id": kid,
                        "attempts": attempt,
                    }

        if last_error is None:
            return {"status": "error", "error": "no_attempts", "latency_ms": 0,
                    "key_id": None, "attempts": 0}
        last_error["status"] = "error"
        last_error["error"] = (
            f"exhausted {attempt} attempts across {self.n_keys} keys: "
            f"{last_error['error']}"
        )
        return last_error

    def stats_summary(self) -> str:
        parts = []
        for kid, _ in self.pool:
            s = self.success_by_key.get(kid, 0)
            e = self.errors_by_key.get(kid, 0)
            parts.append(f"{kid}={s}ok/{e}err")
        return ", ".join(parts)


__all__ = ["GeminiClient"]
