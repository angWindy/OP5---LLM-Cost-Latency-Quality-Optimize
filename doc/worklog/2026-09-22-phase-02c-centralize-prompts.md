# 2026-09-22 — Phase 02c: Centralize system prompts (single source of truth)

> **Phase:** Phase 02c — Compression benchmark scripts
> **Reason:** User instruction — *"đảm bảo các phase từ baseline, đến ful B+C+D đều phải dùng chung 1 system prompt, việc cải thiện sẽ được cải tiến trong phần công nghệ chứ không phải là từ system prompt, khi sửa system prompt từ 1 bên phải sửa đồng bộ các bên khác"*

## Problem

Two layers of prompt duplication found:

### Eval prompts (5 phase-02 scripts + 1 phase-01 helper)

All scripts had identical `PROMPT_TEMPLATE` inline. `scripts/phase-01/_common.py` had a
**divergent** variant (`"CONTEXT (contract):"` + `"Answer concisely using only information from the context"`).

If you ever needed to tweak the eval prompt (e.g., for an ablation), you'd have to
edit 6 files — and risk divergence.

### Judge prompts (3 src/op5/llm files)

`judge.py`, `nim_judge.py`, `openrouter_judge.py` each had their own `JUDGE_PROMPT[_TEMPLATE]`
with subtle wording differences:
- `judge.py`: `"info from"` / `"US" = "United States"` (no comma)
- `nim_judge.py`: same as judge.py + extra rule
- `openrouter_judge.py`: `"information from"` / `"US", "United States"` (with comma) + extra rule

This means **NIM judge and OpenRouter judge gave verdicts using slightly different prompts**
— bad for cross-provider comparison.

## Fix — 2 canonical modules

### 1. `scripts/_prompts.py` (eval prompts for Gemini-direct path)

```python
EVAL_PROMPT_TEMPLATE = (
    "CONTEXT (reference document):\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Answer concisely and accurately based on the context. "
    "If the answer is not present in the context, reply 'Unknown'."
)

def format_eval_prompt(context, question): ...
def format_track1_prompt(context): ...  # reserved for Phase 4+
def format_judge_prompt(question, gold, pred): ...  # Gemini-direct (deprecated)
```

### 2. `src/op5/llm/judge_prompt.py` (judge prompt for NIM/OpenRouter/auto-chain)

```python
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. ..."""

def format_judge_prompt(context, question, gold, pred): ...
```

## Files changed

### Eval prompt consolidation

| File | Before | After |
|---|---|---|
| `scripts/phase-02/baseline_200.py` | inline `PROMPT_TEMPLATE` | import `_format_eval_prompt` |
| `scripts/phase-02/rerun_failed_9.py` | inline `PROMPT_TEMPLATE` | import `_format_eval_prompt` |
| `scripts/phase-02/llmlingua_20.py` | inline `PROMPT_TEMPLATE` | import `_format_eval_prompt` |
| `scripts/phase-02/run_longllmlingua.py` | inline `PROMPT_TEMPLATE` | import `_format_eval_prompt` |
| `scripts/phase-02/run_llmlingua_v2.py` | inline `PROMPT_TEMPLATE` | import `_format_eval_prompt` |
| `scripts/phase-01/_common.py` `build_prompt_track2` | divergent inline | delegates to `_format_eval_prompt` |
| `scripts/phase-01/_common.py` `judge_answer_gemini` | inline judge prompt | delegates to `_format_gemini_judge_prompt` |

### Judge prompt consolidation

| File | Before | After |
|---|---|---|
| `src/op5/llm/judge.py` | `JUDGE_PROMPT = "..."` (inline) | `from op5.llm.judge_prompt import JUDGE_PROMPT_TEMPLATE as JUDGE_PROMPT` |
| `src/op5/llm/nim_judge.py` | `JUDGE_PROMPT_TEMPLATE = "..."` (inline, with extra rule) | import canonical (drops extra rule) |
| `src/op5/llm/openrouter_judge.py` | `JUDGE_PROMPT_TEMPLATE = "..."` (inline, with extra rule + comma) | import canonical (drops extra rule, removes comma) |

### New files

| File | Purpose |
|---|---|
| `scripts/_prompts.py` | Single source for eval prompt (Track 2 baseline + extraction reserved + Gemini-direct judge) |
| `src/op5/llm/judge_prompt.py` | Single source for LLM-as-judge prompt (NIM / OpenRouter / auto-chain) |

## Canonical variants chosen

### Eval prompt = phase-02 (English, "reference document", "accurately")

Reason: this is the variant in 5 production scripts and the variant that produced
the 13.2% baseline accuracy numbers (after the 2026-09-22 VN→EN cleanup). Changing
to the phase-01 variant ("contract", "using only") would invalidate comparisons.

### Judge prompt = judge.py (info, no comma, no extra rule)

Reason: this is the simplest, most established variant. The extra rule in
nim/openrouter was added later (a "Do NOT write 'The user wants me to...'" guard)
but it's redundant with the existing "JSON only" rule — model would have already
been trained not to add that preamble.

## Verified

- All 11 files parse OK (ast.parse)
- `grep "Answer concisely"` finds matches ONLY in `scripts/_prompts.py` (1 location)
- `grep "You are an expert evaluator"` finds matches ONLY in `src/op5/llm/judge_prompt.py` (1 location)
- Functional tests pass:
  - `format_eval_prompt('C', 'Q')` produces canonical phase-02 prompt
  - `_format_gemini_judge_prompt('Q', 'G', 'P')` produces canonical Gemini-direct judge prompt

## How to edit prompts in the future

1. **Want to change eval prompt for ALL phases?** Edit `scripts/_prompts.py:EVAL_PROMPT_TEMPLATE` only.
2. **Want to change judge prompt?** Edit `src/op5/llm/judge_prompt.py:JUDGE_PROMPT_TEMPLATE` only.
3. **Don't edit prompts inline anywhere else** — that defeats the single source of truth.

If you really need a one-off ablation prompt, copy the canonical format function
and override in a local script (don't touch the shared module).
