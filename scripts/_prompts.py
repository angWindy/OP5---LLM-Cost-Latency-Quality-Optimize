"""
Shared system prompts for all eval phases.

Single source of truth for the prompt sent to the eval LLM (Gemini flash-lite).
Every script that calls Gemini must use `format_eval_prompt(context, question)`
(or `format_track1_prompt` for extraction) from this module — never inline the
prompt in the script.

Rules:
  - Eval script improvements come from compression / routing / model changes,
    NOT from prompt tweaks. If a prompt edit is needed, it must be made here
    AND applied to every caller at the same time.
  - Keep the prompt English (per AGENTS.md strict English policy outside doc/).
"""
from __future__ import annotations

# ---- Track 2 / QA prompt (used by baseline, all compressors, rerun scripts) ----
EVAL_PROMPT_TEMPLATE = (
    "CONTEXT (reference document):\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Answer concisely and accurately based on the context. "
    "If the answer is not present in the context, reply 'Unknown'."
)


def format_eval_prompt(context: str, question: str) -> str:
    """Canonical Track 2 / RAG prompt. Use this from every script."""
    return EVAL_PROMPT_TEMPLATE.format(context=context, question=question)


# ---- Track 1 / Extraction prompt (Phase 4+, reserved) ----
TRACK1_PROMPT_TEMPLATE = (
    "Extract the key fields from the contract below. "
    "Return JSON with keys: party_a, party_b, effective_date, contract_value, "
    "termination_clause. If a field is missing, set it to null.\n\n"
    "CONTRACT:\n{context}"
)


def format_track1_prompt(context: str) -> str:
    """Canonical Track 1 / extraction prompt."""
    return TRACK1_PROMPT_TEMPLATE.format(context=context)


# ---- Judge prompt (LLM-as-judge via Gemini, deprecated; production uses NIMJudge) ----
JUDGE_PROMPT_TEMPLATE = (
    "You are an evaluation assistant. Compare the system's answer against the "
    "reference answer.\n\n"
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "System answer: {pred}\n\n"
    "If the system answer is correct or semantically equivalent (exact wording "
    "is not required), return JSON: {{\"correct\": true, \"reason\": \"...\"}}.\n"
    "If it is wrong or does not address the question, return: "
    "{{\"correct\": false, \"reason\": \"...\"}}.\n"
    "Return JSON only, no extra explanation."
)


def format_judge_prompt(question: str, gold: str, pred: str) -> str:
    """Canonical LLM-as-judge prompt (Gemini-direct, deprecated path)."""
    return JUDGE_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)


# Thin wrapper used by scripts/phase-01/_common.py:judge_answer_gemini (deprecated).
# Kept here so all eval/judge prompts live in one module family.
def _format_gemini_judge_prompt(question: str, gold: str, pred: str) -> str:
    return format_judge_prompt(question, gold, pred)
