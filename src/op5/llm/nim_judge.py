"""
NVIDIA NIM LLM-as-judge client with automatic fallback chain.

Primary judge: `nvidia/nemotron-3-ultra-550b-a55b`
  — 550B MoE, GPQA Diamond 86.7%, 1M context, ~760ms latency.

Fallback chain (tiered by capability, descending priority):
  1. `nvidia/nemotron-3-ultra-550b-a55b`    ← primary (strongest)
  2. `meta/llama-3.2-11b-vision-instruct`   ← fast fallback (~600ms)

Auto-fallback triggers:
  - HTTP 429 (rate limit)
  - HTTP 503 (model unavailable / overloaded)
  - HTTP 401/403/404 (auth error or model not available)
  - timeout (> 60 s per call)
  - JSON parse failure after 2 retries on same model

Usage:
    from op5.llm import NIMJudge

    judge = NIMJudge()
    result = judge.judge(question="...", gold="...", pred="...", context="...")
    # result = {"verdict": "correct"|"incorrect"|"ambiguous", "correct": True|False|None,
    #           "reason": "...", "confidence": 0.0-1.0, "model_used": "...", "raw": "..."}
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests

# ---------------------------------------------------------------
# Judge prompt template — shared with OpenRouterJudge
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
# Fallback model chain
# ---------------------------------------------------------------
# Reference: NIM integration scan Sep 2026.
#   #1 Nemotron 3 Ultra 550B (760ms, strongest)
#   #2 Llama 3.2 11B Vision (606ms, fast fallback)
JUDGE_MODELS_PRIMARY = [
    "nvidia/nemotron-3-ultra-550b-a55b",
]
JUDGE_MODELS_FALLBACK = [
    "meta/llama-3.2-11b-vision-instruct",
]

# ---------------------------------------------------------------
# Dataclass for judge result — shared interface with OpenRouterJudge
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
# NVIDIA NIM client
# ---------------------------------------------------------------
class NIMJudge:
    """
    LLM-as-judge via NVIDIA NIM with automatic fallback.

    Env vars:
        NVIDIA_API_KEY  — NVIDIA NIM API key (required)
        OP5_JUDGE_TEMPERATURE — sampling temp (default 0.1)

    Auto-fallback: tries models in order until one succeeds.
    Logs every fallback event for ops visibility.
    """

    NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_TIMEOUT_SEC = 60
    MAX_RETRIES_PER_MODEL = 2
    RETRY_BACKOFF_FACTOR = 2.0

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

        load_dotenv(Path(__file__).resolve().parents[3] / ".env")

        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not set in .env. "
                "Add: NVIDIA_API_KEY=nvapi-..."
            )

        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("OP5_JUDGE_TEMPERATURE", "0.1"))
        )
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec or self.DEFAULT_TIMEOUT_SEC

        self.models = (
            models
            if models is not None
            else JUDGE_MODELS_PRIMARY + JUDGE_MODELS_FALLBACK
        )
        self._primary_models = list(JUDGE_MODELS_PRIMARY)
        self._fallback_models = list(JUDGE_MODELS_FALLBACK)

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

        last_error: str = ""
        for model_id in self.models:
            retries = self.MAX_RETRIES_PER_MODEL

            for attempt in range(retries + 1):
                t0 = time.perf_counter()

                try:
                    raw_response = self._call_nim(model_id, prompt)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    result = self._parse_response(raw_response, model_id, latency_ms)

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
                    last_error = str(exc)
                    self._log_event("model_skip", {
                        "model": model_id,
                        "reason": last_error,
                    })
                    break

        return JudgeResult(
            verdict="error",
            correct=None,
            reason=f"all models failed: {last_error}",
            confidence=0.0,
            model_used=self.models[0] if self.models else "none",
            raw="",
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # NIM API call
    # ------------------------------------------------------------------
    def _call_nim(self, model_id: str, prompt: str) -> str:
        url = f"{self.NIM_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
            raise _FatalError("401 auth — check NVIDIA_API_KEY")
        if status == 403:
            raise _FatalError(f"403 forbidden — model {model_id!r} may require auth or is unavailable")
        if status == 404:
            raise _FatalError(f"404 — model {model_id!r} not found on NIM")
        if status == 429:
            raise _RetryableError(f"429 rate limit on model {model_id!r}")
        if status == 503:
            raise _RetryableError(f"503 service unavailable for model {model_id!r}")
        if status >= 500:
            raise _RetryableError(f"HTTP {status} from NIM for {model_id!r}")

        if status != 200:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": resp.text[:200]}
            raise _RetryableError(
                f"HTTP {status}: {err_body.get('detail', resp.text[:100])}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise _RetryableError(f"failed to parse JSON response: {exc}") from None

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise _RetryableError(f"unexpected response shape: {exc}, body={resp.text[:200]}") from None

        finish_reason = data["choices"][0].get("finish_reason", "stop")
        if finish_reason == "length":
            content = f"<<TRUNCATED>>\n{content}"

        return content

    # ------------------------------------------------------------------
    # Response parsing — identical to OpenRouterJudge
    # ------------------------------------------------------------------
    def _parse_response(
        self, raw: str, model_used: str, latency_ms: float
    ) -> JudgeResult:
        raw_stripped = (raw or "").strip()

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
                            return NIMJudge._parse_verdict_obj(obj)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            break

        matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
        if matches:
            for m in reversed(matches):
                try:
                    obj = json.loads(m.group())
                    return NIMJudge._parse_verdict_obj(obj)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        return None, 0.5, ""

    @staticmethod
    def _parse_verdict_obj(obj: dict) -> tuple[str | None, float, str]:
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
    # Ops logging
    # ------------------------------------------------------------------
    def _log_event(self, event: str, data: dict) -> None:
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
            pass


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
) -> NIMJudge:
    """
    Create a NIMJudge from .env settings.

    Args:
        log_path: path to ops JSONL log (optional)
        temperature: override OP5_JUDGE_TEMPERATURE env var
        max_tokens: max completion tokens (default 800)

    Returns:
        NIMJudge instance ready to use
    """
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[2]
    load_dotenv(_repo / ".env", override=True)

    return NIMJudge(
        log_path=Path(log_path) if log_path else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
