# Phase 03 Scripts — OCR Bake-off + RAG + Track 1/2 MVP

> English-only README per AGENTS.md. Vietnamese docs live in `doc/`.

## Install

```bash
conda activate vsf
pip install -r scripts/phase-03/requirements.txt

# Optional Docker stack (ChromaDB + Postgres + LocalStack + Redis):
docker compose -f docker/docker-compose.yml up -d
```

## Common environment

- `GOOGLE_API_KEY`: single key, or `GOOGLE_API_KEYS=k1,k2,k3` for rotation.
- `STORAGE_BACKEND=docker` (default) | `inprocess`
- Output: `results/phase-03-*.jsonl` (gitignored).

## Scripts (in execution order)

| #   | Script                          | Purpose                                                   |
| --- | ------------------------------- | --------------------------------------------------------- |
| 1   | `generate_synthetic_contracts.py` | (Pre-requisite) Build 15 SYNTH VinFast-style contracts. |
| 2   | `rasterize_for_ocr.py`            | Render PDFs to PNGs for OCR.                             |
| 3   | `verify_synthetic_contracts.py`   | Sanity-check the SYNTH corpus.                          |
| 4   | `run_ocr_bakeoff.py`              | 3 lightweight OCR providers × 15 scanned contracts.      |
| 5   | `score_ocr.py`                    | Text F1 + table TEDS per provider, pick winner.          |
| 6   | `verify_redaction.py`             | Sensitive-PII redaction smoke (precision/recall).        |
| 7   | `run_extraction_track1.py`        | 5 configs × 15 contracts (Track 1 extraction).           |
| 8   | `score_track1.py`                 | Track 1 scorers (field_f1, table_teds, schema, PII).     |
| 9   | `run_rag_track2.py`               | 7 configs × 30 mini-QA (Track 2 RAG).                   |
| 10  | `score_track2.py`                 | Track 2 scorers (EM, token_f1, refusal, citation, PII). |
| 11  | `pareto_plot.py`                  | Cost-vs-quality PNG plots into `doc/figs/`.            |

## Output JSONL schemas

See `../../src/op5/schemas/eval_log.json` (tracks: `track1 | track2 | ocr | redaction_smoke`).
Each row also carries `redacted: true` + `pii_spans[]` for redaction audit.

## Docker storage stack

See `../../docker/README.md` for ChromaDB / Postgres / LocalStack (S3) / Redis setup.
For offline dev with no Docker, set `STORAGE_BACKEND=inprocess` — the storage layer
falls back to Persistent Chroma, SQLite, local FS, and an in-memory dict.

## Smoke-test vs production

By default the Track 1 & Track 2 runners use a **stub LLM** that parses the
synthesized `key: value` corpus text. To switch to a live Gemini key, edit
`run_extraction_track1.py` / `run_rag_track2.py` and replace `_stub_llm(prompt)`
with `GeminiClient.generate(prompt)`. The cost/latency entries are still
illustrative until then.
