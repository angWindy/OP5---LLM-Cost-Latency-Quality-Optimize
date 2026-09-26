"""LLM client wrappers for OP5."""
from __future__ import annotations

from op5.llm.judge import LLMJudge, JudgeResult, make_judge

# Legacy aliases — existing code keeps working
from op5.llm.nim_judge import NIMJudge
from op5.llm.openrouter_judge import OpenRouterJudge
from op5.llm.gemini_client import GeminiClient
from op5.llm.wrapper import LLMCall, LLMWrapper, make_llm_wrapper

__all__ = [
    # New unified API (recommended)
    "LLMJudge",
    "JudgeResult",
    "make_judge",
    "GeminiClient",
    # Live LLM wrapper for Phase 03 / Phase 04 runners + API
    "LLMCall",
    "LLMWrapper",
    "make_llm_wrapper",
    # Legacy (kept for backward compat)
    "NIMJudge",
    "OpenRouterJudge",
]
