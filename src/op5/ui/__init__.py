"""OP5 Streamlit UI (multipage) — talks to the FastAPI service.

This module is intentionally thin: no business logic lives here, only:
- HTTP client (`lib/api_client.py`) that wraps the FastAPI service
- 4 page files that render Streamlit widgets and call the client

Run with:
    conda activate vsf
    bash scripts/ui/run.sh
or:
    OP5_API_URL=http://localhost:8000 streamlit run src/op5/ui/Home.py
"""
