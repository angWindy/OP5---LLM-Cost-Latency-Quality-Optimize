"""LLM client wrappers for OP5."""
from __future__ import annotations

from op5.llm.judge import LLMJudge, JudgeResult, make_judge

# Legacy aliases — existing code keeps working
from op5.llm.nim_judge import NIMJudge
from op5.llm.openrouter_judge import OpenRouterJudge

__all__ = [
    # New unified API (recommended)
    "LLMJudge",
    "JudgeResult",
    "make_judge",
    # Legacy (kept for backward compat)
    "NIMJudge",
    "OpenRouterJudge",
]
