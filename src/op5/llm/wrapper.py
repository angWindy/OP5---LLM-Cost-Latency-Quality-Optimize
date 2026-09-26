"""Live-LLM wrapper for Phase 03 / Phase 04 runners + API.

Wraps `op5.llm.gemini_client.GeminiClient.generate()` into the same
`(prompt) -> str` callable shape that Track 1 / Track 2 runners expect,
while exposing real `cost_usd` and `latency_ms` from the API response.

Single model: gemini-3.5-flash-lite for all tiers.
  - $0.075 / $0.30 per 1M tokens (input / output)

Tier labels (cheap / mid / strong) are kept for observability only —
all tiers route to the same deployment.

If no API key is configured, `LLMWrapper.from_env()` raises RuntimeError so
the CLI scripts can fall back to `stub` mode instead of silently doing the
wrong thing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from op5.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_PROFILES_PATH = Path(__file__).resolve().parent / "profiles" / "gemini.yaml"

# All tiers use gemini-3.5-flash-lite — pricing is identical for every tier.
DEFAULT_PRICING = {
    "gemini-3.5-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
    "gemini-3.1-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
}

CACHE_STATUS_MAP = {
    "ok": "miss",
    "error": "error",
}


@dataclass
class LLMCall:
    """Per-call metadata returned alongside the LLM text."""

    prompt: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cache_status: str
    key_id: str | None
    attempts: int
    raw_status: str
    pricing_effective_date: str
    error: str | None = None


@dataclass
class LLMWrapper:
    """Stateful LLM caller that exposes both the simple (prompt)->str form
    used by the Track 1 / Track 2 runners, and a richer form that returns
    full LLMCall metadata."""

    client: GeminiClient
    default_model: str = "gemini-3.5-flash-lite"
    pricing: dict[str, dict[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PRICING.items()})
    pricing_effective_date: str = "2026-09-26"

    @classmethod
    def from_env(
        cls,
        default_model: str = "gemini-3.5-flash-lite",
        client: GeminiClient | None = None,
    ) -> "LLMWrapper":
        """Build a wrapper from environment variables.

        Raises RuntimeError if no API key is configured (so CLI scripts can
        fall back to stub mode).
        """
        if client is None:
            try:
                client = GeminiClient()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Cannot build live LLM wrapper: {exc}. "
                    "Set GOOGLE_API_KEY or GOOGLE_API_KEYS in env, or pass --llm stub."
                ) from exc
        pricing = _load_pricing_from_yaml(_PROFILES_PATH)
        return cls(
            client=client,
            default_model=default_model,
            pricing=pricing,
            pricing_effective_date="2026-09-26",
        )

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rates = (
            self.pricing.get(model)
            or DEFAULT_PRICING.get(model)
            or DEFAULT_PRICING["gemini-3.5-flash-lite"]
        )
        in_cost = (input_tokens / 1_000_000.0) * rates["input_per_million"]
        out_cost = (output_tokens / 1_000_000.0) * rates["output_per_million"]
        return round(in_cost + out_cost, 8)

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMCall:
        """Call Gemini and return a structured LLMCall.

        On error the returned LLMCall has `raw_status='error'` and `error`
        populated; the runners should treat `text=''` as a failed call.
        """
        model = model or self.default_model
        raw = self.client.generate(
            prompt,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        text = raw.get("text", "") or ""
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        latency_ms = float(raw.get("latency_ms") or 0.0)
        status = raw.get("status", "error")
        cost_usd = self.compute_cost(model, input_tokens, output_tokens) if status == "ok" else 0.0
        return LLMCall(
            prompt=prompt,
            model=model,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cache_status=CACHE_STATUS_MAP.get(status, "error"),
            key_id=raw.get("key_id"),
            attempts=int(raw.get("attempts") or 0),
            raw_status=status,
            pricing_effective_date=self.pricing_effective_date,
            error=raw.get("error") if status != "ok" else None,
        )

    def call(self, prompt: str, *, model: str | None = None, max_output_tokens: int = 512) -> str:
        """Simple `(prompt) -> str` shape used by Track 1 / Track 2 runners."""
        result = self.generate(prompt, model=model, max_output_tokens=max_output_tokens)
        return result.text if result.raw_status == "ok" else ""


def _load_pricing_from_yaml(path: Path) -> dict[str, dict[str, float]]:
    """Load pricing from profiles/gemini.yaml.

    Falls back to DEFAULT_PRICING if the file is missing or malformed.
    """
    if not path.exists():
        return {k: dict(v) for k, v in DEFAULT_PRICING.items()}
    try:
        import yaml

        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        models = cfg.get("models") or []
        pricing = {k: dict(v) for k, v in DEFAULT_PRICING.items()}
        for m in models:
            if isinstance(m, str) and m in DEFAULT_PRICING:
                pricing[m] = dict(DEFAULT_PRICING[m])
        return pricing
    except Exception as exc:
        logger.warning("Could not parse pricing from %s: %s; using defaults", path, exc)
        return {k: dict(v) for k, v in DEFAULT_PRICING.items()}


def make_llm_wrapper(
    client: GeminiClient | None = None,
    default_model: str = "gemini-3.5-flash-lite",
) -> LLMWrapper:
    """Convenience constructor used by CLI scripts."""
    return LLMWrapper.from_env(default_model=default_model, client=client)


__all__ = ["LLMCall", "LLMWrapper", "DEFAULT_PRICING", "make_llm_wrapper"]
