# Worklog — 2026-09-18 — Status check + Q2 resolved (po5 → vsf) + first runs

> **Phase:** [phase-01-llmlingua-poc.md](../phases/phase-01-llmlingua-poc.md)
> **Status when started:** active (scripts scaffolded, not executed)
> **Author:** —
> **Env (resolved):** conda `vsf` (Python 3.11.16). Project docs/script docstrings
> were updated today to match.

## What this session actually did

### 1. Orientation (read-only)
- `AGENTS.md`, `README.md`, `doc/phases/INDEX.md`.
- Master plan v2 (`doc/plan/op5-llm-cost-latency-quality-plan.md`).
- Phase 01 plan (`doc/phases/phase-01-llmlingua-poc.md`) and previous worklog
  (`doc/worklog/2026-09-17-phase-01-llmlingua-poc.md`).
- Phase 00 plan (`doc/phases/phase-00-foundation.md`).
- `scripts/phase-01/` (6 scripts + `_common.py` + `requirements.txt`) and
  `scripts/README.md`.
- `.env` (Gemini key + HF token already present).

### 2. Decision (Q2) — rename conda-env reference `po5` → `vsf`
- Reality on this host: only `base` and `vsf` exist; `po5` was never created.
- User instruction: *"Q2 : sửa doc dùng vsf"*.
- Change applied (textual rename only — no logic change, no schema change):
  - `AGENTS.md` (lines 9, 43)
  - `README.md` (line 46)
  - `src/README.md` (lines 20, 30)
  - `scripts/README.md` (lines 30, 35)
  - `scripts/phase-01/{smoke_gemini,smoke_huggingface,smoke_llmlingua_langchain,
    inspect_dataset,poc_track1,poc_track2}.py` (Usage docstring)
  - `doc/phases/phase-00-foundation.md` (lines 16, 30, 49)
  - `doc/phases/phase-01-llmlingua-poc.md` (lines 22, 78, 98, 107)
  - `doc/worklog/2026-09-17-phase-01-llmlingua-poc.md` (lines 6, 24)
  - `doc/worklog/2026-09-18-phase-01-status-check.md` (this file — Q2 marked resolved)
- Verified post-rename via `grep -E "\bpo5\b"` over the repo. Remaining matches
  are intentional historical references in this worklog only.

### 3. First runs — Phase 01 smoke chain
Will be appended below as each smoke step executes. Order from `scripts/README.md`:
`smoke_huggingface.py` → `smoke_gemini.py` → `smoke_llmlingua_langchain.py`
→ `inspect_dataset.py` → `poc_track1.py` → `poc_track2.py`.

## Reality check (post-rename)

| Item | Status now |
|---|---|
| Conda env name | `vsf` (Python 3.11.16). Both docs and host agree. |
| Phase 01 scripts | Written; Usage docstrings now point at `vsf`. |
| `results/` | To be created by the first script that writes JSONL. |
| EvalLog JSON Schema | Still missing (Phase 00 deliverable). PoC uses "EvalLog-style" loose fields — accepted for PoC. |
| Held-out set | Still blocked on mentor-approved ground-truth (master plan D3). |
| Git history | Single commit `73ee712` (initial scaffold); will become multi-commit once PoC produces output. |

## Blockers

- **B1 (resolved).** Conda env naming — docs and host now agree on `vsf`.
- **B2 (open, soft).** Phase 00 still not started. PoC can proceed without it.
- **B3 (open).** `scripts/phase-01/requirements.txt` deps may not be installed in
  `vsf` yet. First smoke step will reveal which `pip install` is needed.

## Open questions

- **Q1** (carry-over from 2026-09-17) — LongBench-v2 schema: `context`/`question`/
  `answer`/`choice` vs fallback `input`/`output`/`label`. To be answered by
  `inspect_dataset.py` output.
- **Q2 (resolved).** Conda env naming — adopted `vsf`.

## Next steps

1. Run the smoke chain in order, capture output:
   - `python scripts/phase-01/smoke_huggingface.py`
   - `python scripts/phase-01/smoke_gemini.py`
   - `python scripts/phase-01/smoke_llmlingua_langchain.py`
   - `python scripts/phase-01/inspect_dataset.py --n 3`
2. If smokes pass, run `poc_track1.py --n 15` then `poc_track2.py --n 15`.
3. Inspect `results/phase-01-track{1,2}-poc.jsonl`, compute ballpark metrics.
4. Write follow-up worklog with ship/iterate/drop decision per compressor.
5. Do **not** flip Phase 01 to `done` until both `poc_track{1,2}` succeed.
