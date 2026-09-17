# data/ — Datasets (agent-ignored)

This folder is on the **Cursor ignore list** (see `/home/angwindy/Dev/VSF/OP5/.cursorignore`).
**Agents and the assistant will not read its contents by default.** If you need to inspect
something here, ask explicitly:

> *"Read the first 50 lines of `data/raw/squad-train-00000.parquet`"*
> *"List the first 10 files in `data/raw/`"*

Never load the whole tree into context — datasets can be gigabytes.

## Layout

```
data/
├── README.md                <- this file (always allowed)
├── raw/                     <- original downloads, untouched
│   └── README.md            <- provenance: who downloaded what, when, from where, license
├── processed/               <- cleaned / converted / split (dev, held-out)
│   └── README.md            <- transformation notes
└── cache/                   <- transient caches (HuggingFace downloads, OCR temp files)
    └── README.md
```

## Rules

1. **Never commit** anything under `data/`. All subdirs are gitignored.
2. **Provenance is mandatory.** Any file in `data/raw/` must have an entry in
   `data/raw/README.md` recording: source URL, download date (ISO 8601), license,
   hash if available.
3. **HuggingFace datasets** live here. Use streaming (`load_dataset(..., streaming=True)`)
   when you only need a slice. Persist only what you need to keep.
4. **Ground-truth** for OP5 evaluation lives in `data/processed/` under one of:
   - `dev_set/` — used to tune prompts, pick routing rules.
   - `held_out/` — locked, touched **exactly once** at the end (see master plan §4.3).
5. **No PII / no company data.** If you ever want to bring a real contract in, sanitize
   it first. Side-project rule from the master plan §1.

## Convention for filenames

`<source>_<split>_<yyyy-mm-dd>.<ext>`. Example: `squad_v1_train_2026-09-17.parquet`.
