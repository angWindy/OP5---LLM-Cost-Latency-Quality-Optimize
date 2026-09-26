# src/ — Code

Python package layout for OP5. Imports use `op5.<module>`.

```
src/
├── README.md                <- this file
└── op5/                     <- the package
    ├── __init__.py
    ├── router.py            <- deterministic router (Phase 0 deliverable)
    ├── schemas/             <- JSON Schema files (Phase 0)
    │   └── *.json
    ├── scorers/             <- Python scorers (Phase 0): TEDS, refusal, citation
    ├── llm/                 <- LLM client wrappers (Gemini, DeepSeek, OpenAI…)
    └── ocr/                 <- OCR adapter wrappers (Surya, Marker, Docling…)
```

## Conventions

- Python 3.11, conda env `vsf`.
- Type hints everywhere. `from __future__ import annotations`.
- Public functions get docstrings with input/output examples.
- No top-level side effects. Importing `op5` should not call any API.

## Installing for development

From the repo root:

```bash
conda activate vsf
pip install -e .
```

This makes `import op5` work from any working directory.
