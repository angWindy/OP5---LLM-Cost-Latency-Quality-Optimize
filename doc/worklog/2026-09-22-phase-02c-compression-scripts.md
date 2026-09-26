# 2026-09-22 — Phase 2c: Compression benchmark scripts

> **Phase:** [INDEX.md](../phases/INDEX.md) Phase 02c
> **Status:** active
> **Author:** —
> **Env:** conda `vsf`

## TL;DR

Created two new benchmark scripts with full CLI flags for compression experiments:
- `run_longllmlingua.py` — LongLLMLingua (question-aware) compressor benchmark
- `run_llmlingua_v2.py` — LLMLingua-2 (task-agnostic) compressor benchmark with TF-IDF preselect

Both scripts support `--judge-profile`, `--n`, `--seed`, `--model`, `--tasks`, `--sleep`,
`--out`, `--summary-out`, `--ops-out`, `--no-judge`.

## Scripts created

### `scripts/phase-02/run_longllmlingua.py`

LongLLMLingua (question-aware compressor, Track 2 lever C in master plan).
Uses `llmlingua` library (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`).

```
--n INT              cases (default 20)
--seed INT           seed (default 42)
--target-token INT   compression target tokens (default 1500)
--model STR          Gemini model (default gemini-3.5-flash-lite)
--judge-profile STR  judge profile (default nim)
--tasks STR          comma-separated task subset
--sleep FLOAT       seconds between Gemini calls (default 3.0)
--max-chars INT      truncate context before compression (default 0 = off)
--out PATH           output JSONL
--summary-out PATH   summary JSON
--ops-out PATH       ops log for LLMJudge (optional)
--no-judge          skip judge step
```

Pipeline: LongLLMLingua → Gemini → LLM-as-judge.
Output: `results/phase-02-longllmlingua.jsonl`, `*-summary.json`

### `scripts/phase-02/run_llmlingua_v2.py`

LLMLingua-2 (task-agnostic compressor, Track 1 lever B in master plan).
TF-IDF preselect → LLMLingua-2 → Gemini → LLM-as-judge.
v2 of `llmlingua_20.py` with full CLI configurability.

```
--n INT              cases (default 20)
--seed INT           seed (default 42)
--k INT              TF-IDF top-k (default 15; 0 = skip preselect)
--rate FLOAT         LLMLingua rate (default 0.5)
--iter-size INT     iterative_size (default 512)
--model STR          Gemini model (default gemini-3.5-flash-lite)
--judge-profile STR judge profile (default nim)
--tasks STR          comma-separated task subset
--sleep FLOAT       seconds between calls (default 3.0)
--no-preselect      skip TF-IDF preselect
--no-compress       skip LLMLingua-2 (just preselect)
--no-judge          skip judge
--out PATH          output JSONL
--summary-out PATH  summary JSON
--ops-out PATH      ops log for LLMJudge
```

Both scripts share:
- Stratified sample loading from `data/processed/zero_scrolls_200.jsonl` + `results/phase-02-baseline-200-en.jsonl`
- HTTP REST calls to Gemini (same as `baseline_200.py`)
- Unified `LLMJudge` for LLM-as-judge
- Per-task breakdown + summary JSON

## Doc audit results

Reviewed all files in `doc/` for inaccuracies vs actual experiments:

| File | Finding |
|---|---|
| `doc/plan/op5-llm-cost-latency-quality-plan.md` | Consistent — `gemini-3.5-flash-lite` for baseline, `gemini-3.1-pro` for tier-strong (decision 2026-09-22), NIM Nemotron for judge |
| `doc/phases/phase-01-llmlingua-poc.md` | Consistent — LongLLMLingua for Track 2, LLMLingua-2 for Track 1 |
| `doc/phases/phase-00-foundation.md` | Consistent — Phase 0 planned, not yet started |
| `doc/worklog/*.md` | All worklogs are historical records; kept as-is |

**No doc edits needed.** All decisions, model names, and phase labels are consistent across the entire `doc/` tree.

## Previous session: ops log backfill

Session also fixed the ops log:
- **22 missing events** (08:27-08:43 UTC gap) backfilled from judged file
- Root cause: ops log started ~16s late (LLMJudge cold init lag)
- Backup: `phase-02-baseline-200-en-judge-ops.bak.jsonl` (267 lines, pre-backfill)
- Final: 289 events (267 original + 22 backfilled), 200 success, 85 retry, 4 model_done
- Ops log now spans full range: 08:27:25Z → 10:33:44Z
