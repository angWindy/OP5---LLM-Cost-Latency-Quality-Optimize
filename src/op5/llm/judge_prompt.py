"""
Single source of truth for the LLM-as-judge prompt used across all judge
implementations (judge.py, nim_judge.py, openrouter_judge.py).

Rules:
  - Improvements come from compression / routing / model changes, NOT from
    prompt tweaks. If a judge prompt edit is needed, it must be made here
    AND applied to every caller at the same time.
  - All three judge classes (LLMJudge unified, NIMJudge, OpenRouterJudge)
    must use this exact prompt to keep verdicts comparable across providers.
"""
from __future__ import annotations

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Judge whether the system's answer is correct.

RULES (mandatory):
- Output JSON only — no explanation before or after
- JSON must have exactly 3 keys: verdict, reason, confidence

Evaluation criteria:
- "correct": answer starts with correct info from ground truth, or is semantically
  equivalent (e.g. "US" = "United States", "70%" = "70 percent")
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


def format_judge_prompt(context: str, question: str, gold: str, pred: str) -> str:
    """Canonical LLM-as-judge prompt used by all 3 judge classes."""
    return JUDGE_PROMPT_TEMPLATE.format(
        context=context, question=question, gold=gold, pred=pred
    )
