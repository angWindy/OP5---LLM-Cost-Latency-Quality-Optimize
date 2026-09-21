# OP5 — LLM Cost / Latency / Quality Experiments

Side-project (no company data, no company infra) that mirrors the architecture of the real
OP5 pipeline: **Azure Document Intelligence v4.0 (OCR) + private company model (LLM)** for two
tracks — contract extraction and contract RAG — while running on public providers as a demo.

See:

- `doc/problem/OP5.md` — original brief (OP5 in 2026 intern projects).
- `doc/plan/op5-llm-cost-latency-quality-plan.md` — the **consolidated v2 plan** (phases, metrics,
  JSONL schema, deterministic router, stop/continue criteria, GĐ 3 integration checklist).

## Repo layout

```
OP5/
├── README.md                       <- you are here
├── CLAUDE.md / AGENTS.md           <- agent instructions (this project)
├── .cursorignore                   <- tells Cursor what NOT to read into context
├── .gitignore
├── doc/
│   ├── problem/                    <- source briefs (read-only)
│   │   ├── OP5.md
│   │   ├── FILE.md
│   │   └── FILE-vi.md
│   ├── plan/
│   │   └── op5-llm-cost-latency-quality-plan.md   <- the master plan (v2)
│   ├── phases/                     <- one file per phase (action plan)
│   │   ├── phase-00-foundation.md
│   │   ├── phase-01-llmlingua-poc.md
│   │   └── ...
│   ├── worklog/                    <- dated execution log (what was actually done)
│   │   └── YYYY-MM-DD-phase-NN-*.md
│   └── notes/                      <- ad-hoc notes, references, decisions
├── src/                            <- code (Python packages live here)
├── scripts/                        <- one-off scripts (eval runs, ad-hoc tools)
├── data/                           <- datasets (LARGE, gitignored, agent-ignored)
│   ├── raw/                        <- raw downloads from HuggingFace, etc.
│   ├── processed/                  <- cleaned / preprocessed
│   └── README.md
└── results/                        <- JSONL eval logs, plots, Pareto frontiers
```

## Conventions

- **Environment:** conda env `vsf` (Python 3.11). Activate with `conda activate vsf` before
  running anything.
- **LLM API for demos:** Gemini (Google AI Studio) — see `doc/phases/phase-01-llmlingua-poc.md`
  for API key setup.
- **Eval framework:** promptfoo (entry point) + Python scorers (TEDS, refusal accuracy,
  citation precision, etc.). JSONL output, paired by `case_id`.
- **No company data, no company infra.** Everything that touches a real contract must be
  sanitized or synthetic.

## Working in phases

The work is broken into phases. Each phase has:

1. A **plan file** in `doc/phases/` (what we're going to do, and why).
2. One or more **worklog files** in `doc/worklog/` (what actually happened, decisions made,
   blockers hit).

Phases are listed and tracked in `doc/phases/INDEX.md`. New phases get a new file; completed
phases move their summary into the worklog.

## Important: large folders are ignored by default

Cursor agents (including this one) will **not** read large directories like `data/raw/`,
`results/`, or anything that may contain large HuggingFace datasets. To inspect a slice of
such a folder, ask explicitly: *"Read the first 50 lines of `data/raw/<file>`"* or *"List
the first 10 files in `data/raw/`"*. This is set in `.cursorignore`.
