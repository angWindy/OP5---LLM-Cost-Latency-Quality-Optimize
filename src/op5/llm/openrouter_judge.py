"""
OpenRouter LLM-as-judge client with automatic fallback chain.

Primary judge: `nvidia/nemotron-3-ultra-550b-a55b:free`
  — Sep 2026: #1 free model on OpenRouter, GPQA Diamond 86.7%, 1M context.

Fallback chain (tiered, descending priority):
  1. `nvidia/nemotron-3-ultra-550b-a55b:free`   ← primary
  2. `deepseek/deepseek-chat-v3:free`             ← strong reasoning, free
  3. `openrouter/free`                            ← auto-router, fallback
  4. NVIDIA NIM (any available model)             ← last resort when all OR exhausted

Auto-fallback triggers:
  - HTTP 429 (rate limit)
  - HTTP 503 (model unavailable / overloaded)
  - HTTP 401/403 (auth error on specific model, not API key)
  - timeout (> 60 s per call)
  - JSON parse failure after 2 retries on same model

When OpenRouter free tier is exhausted (429/503), NIM fallback activates
automatically using NVIDIA_API_KEY from .env.

Usage:
    from op5.llm import OpenRouterJudge

    judge = OpenRouterJudge()
    result = judge.judge(question="...", gold="...", pred="...", context="...")
    # result = {"verdict": "correct"|"incorrect"|"ambiguous", "correct": True|False|None,
    #           "reason": "...", "confidence": 0.0-1.0, "model_used": "...", "raw": "..."}
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests

# ---------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Judge whether the system's answer is correct.

RULES (mandatory):
- Output JSON only — no explanation before or after
- Do NOT write "The user wants me to..." or any non-JSON text
- JSON must have exactly 3 keys: verdict, reason, confidence

Evaluation criteria:
- "correct": answer starts with correct information from ground truth, or is semantically
  equivalent (e.g., "Poland" = "Ba Lan", "70%" = "70 percent")
- "ambiguous": possibly correct but uncertain (confidence < 0.7)
- "incorrect": main information is wrong or contradicts ground truth

Input:
---
Context: {context}
Question: {question}
Ground truth: {gold}
Prediction: {pred}
---

Respond in the same language as the input. If the question/context is in Vietnamese,
respond in Vietnamese. Otherwise respond in English.

JSON (no explanation):
{{"verdict": "correct"|"incorrect"|"ambiguous", "reason": "1-2 sentences", "confidence": 0.0-1.0}}"""

# ---------------------------------------------------------------
# Fallback model chain (tiered: strongest free → weakest free)
# ---------------------------------------------------------------
# Reference: OpenRouter free model ranking Sep 2026.
#   #1 Nemotron 3 Ultra (GPQA 86.7%, 1M ctx, 550B MoE)
#   #2 DeepSeek Chat V3 (strong reasoning, free tier)
#   #3 openrouter/free (auto-router, fallback)
JUDGE_MODELS_PRIMARY = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "deepseek/deepseek-chat-v3:free",
]
JUDGE_MODELS_FALLBACK = [
    "openrouter/free",
]

# ---------------------------------------------------------------
# NIM fallback — activated when ALL OpenRouter models are exhausted (429/503/timeout)
# ---------------------------------------------------------------
# NIM models available Sep 2026 (confirmed via integration scan):
#   nvidia/nemotron-3-ultra-550b-a55b  — 550B MoE, GPQA 86.7%, ~760ms
#   meta/llama-3.2-11b-vision-instruct — 11B, ~606ms (fast fallback)
# Only used if OpenRouter key is missing or all free models exhausted.
NIM_JUDGE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "meta/llama-3.2-11b-vision-instruct",
]

# ---------------------------------------------------------------
# Dataclass for judge result
# ---------------------------------------------------------------
@dataclass
class JudgeResult:
    verdict: Literal["correct", "incorrect", "ambiguous", "error", "timeout"]
    correct: bool | None          # None = ambiguous / error / timeout
    reason: str
    confidence: float             # 0.0–1.0
    model_used: str
    raw: str                      # raw model response (truncated 500 chars)
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "correct": self.correct,
            "reason": self.reason,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "raw": self.raw[:500],
            "latency_ms": round(self.latency_ms, 1),
        }


# ---------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------
class OpenRouterJudge:
    """
    LLM-as-judge via OpenRouter with automatic fallback.

    Env vars:
        OPENROUTER_API_KEY  — OpenRouter API key (required)
        OP5_JUDGE_TEMPERATURE — sampling temp (default 0.1)

    Auto-fallback: tries models in order until one succeeds.
    Logs every fallback event for ops visibility.
    """

    DEFAULT_TIMEOUT_SEC = 60
    MAX_RETRIES_PER_MODEL = 2       # retries on transient errors (429/5xx)
    RETRY_BACKOFF_FACTOR = 2.0     # exponential backoff multiplier

    def __init__(
        self,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 800,
        timeout_sec: int | None = None,
        models: list[str] | None = None,
        log_path: Path | None = None,
    ):
        from dotenv import load_dotenv

        # Load .env if not already loaded
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")  # legacy alias
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set in .env. "
                "Add: OPENROUTER_API_KEY=sk-or-v1-..."
            )

        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("OP5_JUDGE_TEMPERATURE", "0.1"))
        )
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec or self.DEFAULT_TIMEOUT_SEC

        # Model chain: primary first, then fallback
        self.models = (
            models
            if models is not None
            else JUDGE_MODELS_PRIMARY + JUDGE_MODELS_FALLBACK
        )
        self._primary_models = list(JUDGE_MODELS_PRIMARY)
        self._fallback_models = list(JUDGE_MODELS_FALLBACK)

        # Ops log
        self.log_path = log_path
        self._log_event("init", {"models": self.models, "temperature": self.temperature})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def judge(
        self,
        question: str = "",
        gold: str = "",
        pred: str = "",
        context: str = "",
    ) -> JudgeResult:
        """
        Judge a single prediction.

        Args:
            question: the evaluation question (optional for extraction tasks)
            gold: ground truth answer
            pred: model prediction
            context: optional supporting context

        Returns:
            JudgeResult dataclass
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            context=(context or "")[:2000],
            question=question or "(no question — automatic extraction task)",
            gold=(gold or "")[:500],
            pred=(pred or "")[:500],
        )

        # Try each model in chain
        last_error: str = ""
        for model_id in self.models:
            retries = self.MAX_RETRIES_PER_MODEL

            for attempt in range(retries + 1):
                t0 = time.perf_counter()

                try:
                    raw_response = self._call_openrouter(model_id, prompt)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    result = self._parse_response(raw_response, model_id, latency_ms)

                    # Log successful call
                    self._log_event("success", {
                        "model": model_id,
                        "attempt": attempt + 1,
                        "latency_ms": round(latency_ms, 1),
                        "verdict": result.verdict,
                    })

                    return result

                except _RetryableError as exc:
                    last_error = str(exc)
                    wait = self.RETRY_BACKOFF_FACTOR ** attempt * 2.0
                    self._log_event("retry", {
                        "model": model_id,
                        "attempt": attempt + 1,
                        "error": last_error,
                        "wait_s": round(wait, 1),
                    })
                    time.sleep(wait)
                    continue

                except _FatalError as exc:
                    # Non-retryable error — skip this model immediately
                    last_error = str(exc)
                    self._log_event("model_skip", {
                        "model": model_id,
                        "reason": last_error,
                    })
                    break  # try next model in chain

        # All OpenRouter models failed — try NIM as last resort
        nim_result = self._try_nim_fallback(prompt)
        if nim_result is not None:
            return nim_result

        return JudgeResult(
            verdict="error",
            correct=None,
            reason=f"all OpenRouter and NIM models failed: {last_error}",
            confidence=0.0,
            model_used=self.models[0] if self.models else "none",
            raw="",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # OpenRouter API call
    # ------------------------------------------------------------------
    def _call_openrouter(self, model_id: str, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/op5-llm",
            "X-Title": "OP5 LLM Judge",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
        except requests.Timeout:
            raise _RetryableError(f"timeout after {self.timeout_sec}s") from None
        except requests.ConnectionError as exc:
            raise _RetryableError(f"connection error: {exc}") from None

        status = resp.status_code

        if status == 401:
            raise _FatalError("401 auth — check OPENROUTER_API_KEY")
        if status == 403:
            raise _FatalError(f"403 forbidden — model {model_id!r} may require auth or is unavailable")
        if status == 404:
            raise _FatalError(f"404 — model {model_id!r} not found on OpenRouter")
        if status == 429:
            raise _RetryableError(f"429 rate limit on model {model_id!r}")
        if status == 503:
            raise _RetryableError(f"503 service unavailable for model {model_id!r}")
        if status >= 500:
            raise _RetryableError(f"HTTP {status} from OpenRouter for {model_id!r}")

        if status != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": resp.text[:200]}
            raise _RetryableError(
                f"HTTP {status}: {err_body.get('error', {}).get('message', resp.text[:100])}"
            )

        # Parse response
        try:
            data = resp.json()
        except Exception as exc:
            raise _RetryableError(f"failed to parse JSON response: {exc}") from None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise _RetryableError(f"unexpected response shape: {exc}, body={resp.text[:200]}") from None

        # Check finish_reason for truncation
        finish_reason = data["choices"][0].get("finish_reason", "stop")
        if finish_reason == "length":
            # Truncated → log warning, but still return what we got
            # (caller can detect via finish_reason tag)
            content = f"<<TRUNCATED>>\n{content}"

        return content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(
        self, raw: str, model_used: str, latency_ms: float
    ) -> JudgeResult:
        """Parse model response into JudgeResult.

        Strategy:
          1. Try to extract JSON from response (greedy / multiline).
          2. If no valid JSON → ambiguous (do NOT use keyword heuristic on raw text,
             because model reasoning may contain "correct"/"incorrect" without being
             the final verdict — that produced false positives like L00-3hop1).
        """
        raw_stripped = (raw or "").strip()

        # Strategy 1: greedy JSON match (handles nested braces, multiline)
        verdict, confidence, reason = self._extract_json(raw_stripped)
        if verdict is not None:
            correct = None if verdict == "ambiguous" else (verdict == "correct")
            return JudgeResult(
                verdict=verdict,
                correct=correct,
                reason=reason[:200],
                confidence=confidence,
                model_used=model_used,
                raw=raw_stripped[:500],
                latency_ms=latency_ms,
            )

        # No valid JSON → ambiguous. We deliberately do NOT keyword-match the
        # raw text because model reasoning frequently mentions "correct"/"incorrect"
        # in its deliberation before producing the JSON verdict, leading to
        # false positives (L00-3hop1 predicted "unanswerable" vs gold "Jack Michel"
        # was judged "correct" via raw text keyword match — wrong).
        return JudgeResult(
            verdict="ambiguous",
            correct=None,
            reason=f"no JSON verdict in response (raw: {raw_stripped[:80]})",
            confidence=0.5,
            model_used=model_used,
            raw=raw_stripped[:500],
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_json(text: str) -> tuple[str | None, float, str]:
        """Extract verdict/confidence/reason from JSON in text.

        Tries multiple patterns:
          1. Greedy balanced-brace match (handles nested objects)
          2. Single regex fallback for flat JSON

        Returns (verdict, confidence, reason) — all None/0.0/"" if no match.
        """
        # Pattern 1: find balanced {...} starting from first '{'
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            return OpenRouterJudge._parse_verdict_obj(obj)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            break  # try next strategy
            # No balanced match — try once more from last '{'
            # (sometimes the model prepends commentary, last '{' is the real one)

        # Pattern 2: regex on last {...} block
        matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
        if matches:
            for m in reversed(matches):  # try last match first (most likely the verdict)
                try:
                    obj = json.loads(m.group())
                    return OpenRouterJudge._parse_verdict_obj(obj)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        return None, 0.5, ""

    @staticmethod
    def _parse_verdict_obj(obj: dict) -> tuple[str | None, float, str]:
        """Extract (verdict, confidence, reason) from parsed JSON object."""
        verdict_raw = str(obj.get("verdict", "")).lower().strip()
        if verdict_raw not in ("correct", "incorrect", "ambiguous"):
            return None, 0.5, ""

        try:
            confidence = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reason = str(obj.get("reason", ""))[:200]
        return verdict_raw, confidence, reason


    # ------------------------------------------------------------------
    # NIM fallback — activated when all OpenRouter models are exhausted
    # ------------------------------------------------------------------
    def _try_nim_fallback(self, prompt: str) -> "JudgeResult | None":
        """
        Try NIM judge as last resort. Returns JudgeResult or None on failure.

        NIM models available Sep 2026 (confirmed via integration scan):
          - nvidia/nemotron-3-ultra-550b-a55b  (550B MoE, ~760ms)
          - meta/llama-3.2-11b-vision-instruct (11B, ~606ms)

        Only activates when:
          - NVIDIA_API_KEY is set in .env
          - All OpenRouter models failed with 429/503/timeout
        """
        import requests as _req

        nim_key = os.getenv("NVIDIA_API_KEY")
        if not nim_key:
            self._log_event("nim_skip", {"reason": "no NVIDIA_API_KEY in .env"})
            return None

        # Build a standalone NIM call — no circular import needed
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nim_key}",
            "Content-Type": "application/json",
        }

        for model_id in NIM_JUDGE_MODELS:
            for attempt in range(self.MAX_RETRIES_PER_MODEL + 1):
                t0 = time.perf_counter()
                try:
                    payload = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    }
                    resp = _req.post(url, headers=headers, json=payload,
                                     timeout=self.timeout_sec)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            continue

                        result = self._parse_response(content, model_id, latency_ms)
                        self._log_event("nim_fallback_success", {
                            "model": model_id,
                            "latency_ms": round(latency_ms, 1),
                            "verdict": result.verdict,
                        })
                        return result

                    # 429/503 → retry; 401/403/404 → skip model
                    if resp.status_code in (429, 503):
                        wait = self.RETRY_BACKOFF_FACTOR ** attempt * 2.0
                        time.sleep(wait)
                        continue
                    else:
                        break  # skip to next model

                except (_req.Timeout, _req.ConnectionError):
                    continue

            self._log_event("nim_model_skip", {"model": model_id})

        self._log_event("nim_skip", {"reason": "all NIM models failed"})
        return None

    # ------------------------------------------------------------------
    # Ops logging
    # ------------------------------------------------------------------
    def _log_event(self, event: str, data: dict) -> None:
        """Append a structured log line. No-op if log_path is None."""
        if self.log_path is None:
            return
        try:
            line = json.dumps(
                {
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event": event,
                    **data,
                },
                ensure_ascii=False,
            )
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass  # never fail on logging


# ---------------------------------------------------------------
# Error types
# ---------------------------------------------------------------
class _RetryableError(Exception):
    """Transient error — retry same model."""

    pass


class _FatalError(Exception):
    """Non-retryable error — skip to next model in chain."""

    pass


# ---------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------
def make_judge(
    log_path: str | Path | None = None,
    temperature: float | None = None,
    max_tokens: int = 800,
) -> OpenRouterJudge:
    """
    Create an OpenRouterJudge from .env settings.

    Args:
        log_path: path to ops JSONL log (optional)
        temperature: override OP5_JUDGE_TEMPERATURE env var
        max_tokens: max completion tokens (default 800)

    Returns:
        OpenRouterJudge instance ready to use
    """
    # Ensure .env is loaded before OpenRouterJudge reads OPENROUTER_API_KEY.
    # load_dotenv only sets variables that are NOT already in os.environ,
    # so call it here (in caller's context, after dotenv may have loaded)
    # rather than relying on the module-level call which ran at import time.
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[2]
    load_dotenv(_repo / ".env", override=True)

    return OpenRouterJudge(
        log_path=Path(log_path) if log_path else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
