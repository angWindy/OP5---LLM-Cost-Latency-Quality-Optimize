# 2026-09-22 — Phase 1/2: English language cleanup (rule enforcement)

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md) (active) +
> [phase-02-baseline-200](../phases/INDEX.md) (done but needs re-run)
> **Author:** —
> **Reason:** User clarification 2026-09-22 12:24 — verbatim instruction:
> *"ngoại trừ @doc ra thì tất cả đều phải có ngôn ngữ Tiếng Anh, nhất là System Prompt"*
> → confirmed via AskQuestion: strict English outside `doc/`; model still
> responds in the same language as the input context.

## What changed

User reaffirm: **everything outside `doc/` must be English**, especially
system prompts sent to LLMs. AGENTS.md already listed the rule but several
recent edits drifted into Vietnamese.

### Files edited

| File | Vietnamese content removed | Replacement |
|---|---|---|
| `AGENTS.md` | (rule was present but loose) | Added strict policy block + clarified that example values inside English prompts must also be English |
| `scripts/phase-01/_common.py` | Track 1/2 prompt (Tiếng Việt), Gemini judge prompt (Tiếng Việt), error message `"GOOGLE_API_KEY chua duoc set trong .env"`, ~10 inline comments + docstrings | All to English (Track 1/2 + judge prompts translated semantically; comments paraphrased) |
| `src/op5/llm/judge.py` | Judge prompt example `"Poland" = "Ba Lan"` | `"US" = "United States"` |
| `src/op5/llm/nim_judge.py` | Same example | Same replacement |
| `src/op5/llm/openrouter_judge.py` | Same example | Same replacement |
| `scripts/phase-02/baseline_200.py` | Vietnamese system prompt sent to Gemini for baseline eval | English prompt: `"Answer concisely and accurately based on the context. If the answer is not present in the context, reply 'Unknown'."` |
| `scripts/phase-02/rerun_failed_9.py` | Same Vietnamese prompt | Same English replacement |

### Sweep

After all edits, `Grep` for Vietnamese diacritics in `src/` and `scripts/`
returns **0 matches**. All Python files re-parsed with `ast.parse()` → OK.

## ⚠️ Phase 02 baseline numbers are now OBSOLETE

**Impact:** changing the system prompt from Vietnamese → English changes
Gemini's behavior (response phrasing, length, occasionally accuracy).
The numbers in:

- `doc/worklog/2026-09-22-phase-02-baseline-200-wrapup.md`
- `results/phase-02-baseline-200.jsonl` + `-summary.json`

…are based on the **old Vietnamese prompt** and should NOT be cited as
current baseline accuracy.

**Action required (user decision 2026-09-22, option "retranslate"):** Re-run
`baseline_200.py` with the new English prompt to produce a fresh, comparable
baseline.

```
conda activate vsf
python scripts/phase-02/baseline_200.py \
    --n 200 \
    --output results/phase-02-baseline-200-en.jsonl
```

Until re-run, downstream comparisons (Phase 03 OCR bake-off, Phase 04
configs, Phase 05 RAG configs) should either:
- Use `gemini-3.5-flash-lite` raw API (English default) and ignore Phase 02 baseline, OR
- Wait for the re-run.

## Files NOT touched (per existing convention)

- `doc/worklog/*` — historical record, keep as-is even if contains Vietnamese
  (some worklogs are pre-rule and intentionally in Vietnamese).
- `doc/plan/*` and `doc/phases/*` — Vietnamese is the established language
  for documentation & guides.
- `results/*.jsonl` — experimental data, never edit retroactively.

## Verification

- [x] `ast.parse()` on all modified Python files → OK
- [x] `Grep` for Vietnamese diacritics in `src/` and `scripts/` → 0 matches
- [ ] Re-run Phase 02 baseline with English prompt (TODO — user triggered)
- [ ] Update Phase 02 wrapup worklog with new numbers when available
