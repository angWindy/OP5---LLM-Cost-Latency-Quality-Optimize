# scripts/ — Ad-hoc runners

One-off Python scripts that are **not** part of the `op5` package. Each phase that needs
to run something gets a subfolder.

```
scripts/
├── README.md                          <- this file
├── _prompts.py                        <- shared prompt formatting helpers
└── phase-02/
    ├── download_longbench.py          <- Download LongBench from HuggingFace
    ├── convert_longbench.py           <- Convert raw LongBench to standard JSONL
    ├── build_longbench_stratified.py  <- Build stratified 200-case eval set
    ├── run_baseline.py                <- Baseline run (no compression, Gemini)
    ├── run_compressed.py              <- Compressed run (LongLLMLingua + Gemini)
    └── judge_196_concurrent.py        <- Concurrent LLM-as-judge across all 5 files
```

## Output convention

Results are organized by **layer** to keep the structure clean:

```
results/
├── runs/                              <- raw prediction outputs (no judge)
│   └── phase-02/
│       ├── phase-02-run-{slug}.jsonl           <- raw predictions
│       ├── phase-02-run-{slug}-summary.json
│       └── phase-02-run-{slug}.dedup-report.json
├── judges/                            <- judged outputs (run + judge verdict)
│   └── {profile}/phase-02/
│       ├── phase-02-judged-{slug}.jsonl         <- raw + judge fields merged
│       └── phase-02-judged-{slug}-summary.json
├── summaries/                         <- cross-run summaries
│   └── phase-02/
│       └── phase-02-{profile_key}-196-summary.json
└── _backups/                         <- timestamped backups before re-judge
```

Current judge profiles → subfolder:

| Profile | Subfolder |
|---|---|
| `deepseek` | `deepseek-flash` |
| `deepseek_pro` | `deepseek-v4-pro` |

## Conventions

- Each script is **invoked from the repo root**.
- Scripts read API keys from `.env` at the repo root (via `python-dotenv`).
- A script that takes more than 30 seconds prints progress every N records.
- **Always activate the env first:** `conda activate vsf`.

## Phase 02 workflow

```bash
conda activate vsf

# 1) One-time setup (if not done)
python scripts/phase-02/download_longbench.py
python scripts/phase-02/convert_longbench.py
python scripts/phase-02/build_longbench_stratified.py --n 200

# 2) Run predictions (raw outputs → results/runs/phase-02/)
python scripts/phase-02/run_baseline.py    --out-tag baseline_n196
python scripts/phase-02/run_compressed.py  --rate 0.4
python scripts/phase-02/run_compressed.py  --rate 0.5
python scripts/phase-02/run_compressed.py  --rate 0.6
python scripts/phase-02/run_compressed.py  --rate 0.7

# 3) Judge (read from runs/, write to judges/{profile}/phase-02/)
python scripts/phase-02/judge_196_concurrent.py                  # deepseek-flash
python scripts/phase-02/judge_196_concurrent.py --profile deepseek_pro  # deepseek-v4-pro

# 4) Summary appears at:
#    results/summaries/phase-02/phase-02-deepseek-flash-196-summary.json
#    results/summaries/phase-02/phase-02-deepseek-v4-pro-196-summary.json
```

## Dataset: LongBench

- **Eval set:** `data/processed/longbench_200_stratified.jsonl` (196 cases, 7 QA tasks)
- **Tasks:** hotpotqa, 2wikimqa, musique, narrativeqa, multifieldqa_en, qasper, triviaqa
- **Judge:** `deepseek-flash` (free) or `deepseek-v4-pro` (paid, stronger)
- **Eval model:** `gemini-3.5-flash-lite` (Google AI)

## Compression configs

| Slug | Rate | Token saving | Description |
|---|---|---|---|
| `baseline_n196` | 1.0 | 0% | No compression |
| `rate40` | 0.4 | 60% | Aggressive |
| `rate50` | 0.5 | 50% | LLMLingua default |
| `rate60` | 0.6 | 40% | Moderate |
| `rate70` | 0.7 | 30% | Conservative |
