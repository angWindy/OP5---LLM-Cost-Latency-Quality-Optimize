# OP5 — Phase Index

Each phase has a plan file in this directory. The worklog (what actually happened) lives
in `../worklog/`. The master plan (GĐ 0 → GĐ 3) lives in `../../plan/op5-llm-cost-latency-quality-plan.md`.

| # | Title | Status | Plan | Latest worklog |
|---|---|---|---|---|
| 00 | Foundation: schemas, eval split, deterministic router, promptfoo scaffold | planned | [phase-00-foundation.md](phase-00-foundation.md) | — |
| 01 | **LLMLingua PoC** — test prompt-compression library (pip / LangChain), run on a HuggingFace dataset against the Gemini API | **active** | [phase-01-llmlingua-poc.md](phase-01-llmlingua-poc.md) | [2026-09-22-phase-01-model-selection](../worklog/2026-09-22-phase-01-model-selection.md) |
| 02 | **Baseline 200** — stratified LongBench dev set (196 cases, 7 QA tasks), Gemini 3.5 Flash-Lite cost/latency baseline. Dataset switched từ ZeroSCROLLS 2026-09-23. | **active** | (inline in `scripts/phase-02/baseline_200.py`) | [2026-09-23-phase-02-dataset-switch-longbench](../worklog/2026-09-23-phase-02-dataset-switch-longbench.md) |
| 02b | **LLMLingua-20 test** — compression pipeline test on 20 cases (ZeroSCROLLS, deprecated) | **done** | — | [2026-09-22-phase-02-llmlingua-20-test](../worklog/2026-09-22-phase-02-llmlingua-20-test.md) |
| 02c | **Compression benchmark scripts** — `run_longllmlingua.py` (LongLLMLingua, paper-faithful `rate` param) + `run_llmlingua_v2.py` (LLMLingua-2) + `build_longbench_stratified.py` (200-case sampler) | **active** | — | [2026-09-23-phase-02-dataset-switch-longbench](../worklog/2026-09-23-phase-02-dataset-switch-longbench.md) |
| 03 | OCR provider bake-off (lightweight, no-torch: PaddleOCR + Tesseract + EasyOCR) + Docker storage stack (ChromaDB + Postgres + MinIO/LocalStack + Redis) + Sensitive-PII redaction layer + custom RAG skeleton + Track 1/2 MVP + 9 docs | **done** | [phase-03-ocr-rag.md](phase-03-ocr-rag.md) | [2026-09-25-phase-03-ocr-rag](../worklog/2026-09-25-phase-03-ocr-rag.md) |
| 04 | (reserved) Track 1 MVP — 5 extraction configs | not started | — | — |
| 05 | (reserved) Track 2 MVP — 7 RAG configs (incl. B+C+D) | not started | — | — |

**Status legend:** `planned` → `active` → `done` (or `blocked`).

## Updating this index

When a phase starts: change status to `active`, fill in the plan link. When it ends:
add the worklog row, change status to `done` (or `blocked` with a note in the worklog).
