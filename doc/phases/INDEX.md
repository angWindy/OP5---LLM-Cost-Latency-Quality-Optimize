# OP5 — Phase Index

Each phase has a plan file in this directory. The worklog (what actually happened) lives
in `../worklog/`. The master plan (GĐ 0 → GĐ 3) lives in `../../plan/op5-llm-cost-latency-quality-plan.md`.

| # | Title | Status | Plan | Latest worklog |
|---|---|---|---|---|
| 00 | Foundation: schemas, eval split, deterministic router, promptfoo scaffold | planned | [phase-00-foundation.md](phase-00-foundation.md) | — |
| 01 | **LLMLingua PoC** — test prompt-compression library (pip / LangChain), run on a HuggingFace dataset against the Gemini API | **active** | [phase-01-llmlingua-poc.md](phase-01-llmlingua-poc.md) | [2026-09-21-latency](../worklog/2026-09-21-phase-01-llm-call-latency-investigation.md) |
| 02 | (reserved) OCR provider bake-off — Surya / Marker / Docling / PaddleOCR | not started | — | — |
| 03 | (reserved) Track 1 MVP — 5 extraction configs | not started | — | — |
| 04 | (reserved) Track 2 MVP — 7 RAG configs (incl. B+C+D) | not started | — | — |

**Status legend:** `planned` → `active` → `done` (or `blocked`).

## Updating this index

When a phase starts: change status to `active`, fill in the plan link. When it ends:
add the worklog row, change status to `done` (or `blocked` with a note in the worklog).
