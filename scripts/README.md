# scripts/ — Ad-hoc runners

One-off Python scripts that are **not** part of the `op5` package. Each phase that needs
to run something gets a subfolder.

```
scripts/
├── README.md                       <- this file
├── phase-01/
│   ├── _common.py                  <- shared helpers (Gemini call, dataset, judge)
│   ├── smoke_huggingface.py        <- smoke test for HF_TOKEN + dataset access
│   ├── smoke_llmlingua_langchain.py <- LangChain + LLMLingua smoke test (1 sample)
│   ├── smoke_gemini.py              <- 1-request smoke test for GOOGLE_API_KEY
│   ├── inspect_dataset.py           <- explore LongBench-v2 schema
│   ├── poc_track1.py               <- Track 1 PoC: LLMLingua-2 vs baseline
│   └── poc_track2.py               <- Track 2 PoC: LongLLMLingua vs baseline
├── phase-00/
│   └── schema_smoke.py             <- Phase 0: validate JSONL against EvalLog schema
└── ...
```

## Conventions

- Each script is **invoked from the repo root**, not from inside `scripts/`. This is so
  relative paths to `data/`, `results/`, etc. work the same way for everyone.
- Scripts read API keys from `.env` at the repo root (via `python-dotenv`).
- Scripts write their outputs under `results/<phase>-<descr>.jsonl` or
  `results/<phase>-<descr>/` if there are many files.
- A script that takes more than 30 seconds should print progress every N records.
- **Always activate the env first:** `conda activate po5`.

## Example: running the Phase 1 PoC

```bash
conda activate po5

# 1) Smoke test the API keys (run all before the full PoC)
python scripts/phase-01/smoke_huggingface.py          # HF_TOKEN + dataset access
python scripts/phase-01/smoke_gemini.py               # GOOGLE_API_KEY + model
python scripts/phase-01/smoke_llmlingua_langchain.py  # LangChain + LLMLingua structure

# 2) Inspect dataset schema
python scripts/phase-01/inspect_dataset.py --n 3

# 3) Run Track 1 PoC (LLMLingua-2)
python scripts/phase-01/poc_track1.py --n 15

# 4) Run Track 2 PoC (LongLLMLingua)
python scripts/phase-01/poc_track2.py --n 15
```
