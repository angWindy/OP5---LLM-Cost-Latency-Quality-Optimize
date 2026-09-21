# Phase 0 — Foundation

> **Status:** planned
> **Owner:** —
> **Started:** —

## Why this phase exists

Before any experiment runs, we need:

1. Schemas that don't depend on a specific provider (`OCRResult`, `LLMResponse`, eval log).
2. A **dev set + held-out set** that are locked **before** any prompt or routing is tuned.
3. A **deterministic router** (rule-based) for both tracks — per the OP5 brief, learned
   routing comes *after* the deterministic one is stable.
4. A `promptfoo` entry point that writes JSONL ready for our paired analysis.
5. The conda env `vsf` is usable end-to-end.

Without these, every experiment downstream is incomparable.

## Scope (in)

- Define the 5 schemas in `doc/plan/op5-llm-cost-latency-quality-plan.md` §4.1 as JSON
  Schema files under `src/op5/schemas/`.
- Write a fixture loader that converts an OCRResult + GroundTruth + LLMResponse into one
  EvalLog record. Use synthetic data for the first smoke test.
- Pick the 3 routing thresholds (T1-R1/2/3, T2-R1/2/3) and document them as a Python
  function `route(track, input_features) -> model_id` — no LLM involved.
- Stand up a `promptfoo` config with one trivial provider and one dummy case, confirm
  JSONL output matches the schema.
- Sanity-check `conda activate vsf && python -c "import langchain, promptfoo, pandas, plotly"`.

## Scope (out)

- Real OCR or LLM provider calls (Phase 2+).
- Building the held-out set (that's a data-collection effort, see blocker section).
- Anything beyond one dummy case per track.

## Deliverables

- [ ] `src/op5/schemas/{ocr_result,llm_response,groundtruth_extraction,groundtruth_rag,eval_log}.json`
- [ ] `src/op5/router.py` with `route(track, features) -> model_id`
- [ ] `configs/promptfoo/smoke-test.yaml` (1 dummy case, 1 provider)
- [ ] `data/processed/dev_set.sample.json` (5 synthetic cases, 2 tracks)
- [ ] `results/smoke-test.jsonl` matching EvalLog schema
- [ ] `doc/worklog/YYYY-MM-DD-phase-00-foundation.md` with what worked / what didn't

## Done criteria

- `promptfoo eval -c configs/promptfoo/smoke-test.yaml` runs to completion on `vsf`.
- `results/smoke-test.jsonl` validates against `eval_log.json` (use `jsonschema` CLI).
- The 3 schema files have at least one example each.
- The router returns deterministic output for 10 hand-crafted features.

## Blockers / risks

- **B1.** Held-out set (15 case/track) requires mentor-approved ground-truth. Cannot start
  Track 1 / Track 2 MVP until this exists.
- **B2.** Project 4's SDK may not be ready; if so, we write our own minimal client in
  Phase 0 and replace later.

## Reference

- `doc/plan/op5-llm-cost-latency-quality-plan.md` — the master plan; this phase implements
  §4.1–4.4 of that document.
