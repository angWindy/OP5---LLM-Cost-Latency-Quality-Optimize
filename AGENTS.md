# AGENTS.md — Instructions for AI agents working on OP5

> Read this first. The plan is in `doc/plan/op5-llm-cost-latency-quality-plan.md`. The
> current phase and recent worklog are in `doc/phases/INDEX.md` and `doc/worklog/`.

## TL;DR

- Side-project. **No company data, no company infra.** Demo providers only.
- **Conda env:** always `conda activate po5` before running anything Python.
- **Large folders are ignored by default** (see `.cursorignore`): `data/raw/`, `data/processed/`,
  `data/cache/`, `results/`. Don't try to `Read` or `Grep` them wholesale. Ask the user to
  fetch a slice if you need it.
- Every experiment writes JSONL paired by `case_id` (see master plan §4.4).
- Held-out set is **read-once-only** — do not run experiments on it during tuning.

## Workspace conventions

| Need to… | Use |
|---|---|
| Run an experiment | Write a script under `scripts/phase-XX/`, invoke from repo root |
| Add a schema | JSON Schema under `src/op5/schemas/`, reference from `doc/plan` §4.1 |
| Add a scorer | `src/op5/scorers/<name>.py`, expose `score(pred, ref) -> dict` |
| Log eval results | `results/<phase>-<desc>.jsonl` matching EvalLog schema |
| Document a decision | Append to `doc/worklog/YYYY-MM-DD-phase-XX-<topic>.md` |
| Start a new phase | Add `doc/phases/phase-XX-<slug>.md`, mark `active` in `INDEX.md` |

## What NOT to do

- Don't `Read` anything under `data/raw/` or `data/processed/` without the user
  requesting it explicitly. These can be gigabytes.
- Don't run anything on the **held-out set** until the phase that uses it says so.
- Don't commit `.env` or any file containing API keys.
- Don't change the deterministic router thresholds without a paired experiment on dev
  set + a worklog entry justifying the change.
- Don't start the next phase before closing out the current one (worklog written,
  INDEX.md updated).

## When you start a session

1. Skim `doc/phases/INDEX.md` — find the active phase.
2. Read its plan file.
3. Read the most recent worklog.
4. Confirm `po5` is available (`conda activate po5 && python --version`).
5. Confirm the user wants you to keep going on the active phase, or pivot.

## When you finish a session

1. Update the worklog with what actually happened, even if it's just "tried X, didn't work".
2. Update `doc/phases/INDEX.md` if a phase moved status.
3. Surface any blockers to the user explicitly.
