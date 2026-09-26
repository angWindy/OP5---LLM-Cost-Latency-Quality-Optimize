"""OP5 Home — landing page for the Streamlit UI.

Run with:
    conda activate vsf
    bash scripts/ui/run.sh
"""

from __future__ import annotations

import streamlit as st

from op5.ui.lib.api_client import OP5Api

st.set_page_config(
    page_title="OP5 — LLM Cost / Latency / Quality Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OP5 — End-to-end LLM Cost / Latency / Quality Demo")
st.markdown(
    """
    This UI wraps the Phase 03 OP5 pipeline. It calls a FastAPI service running
    on `OP5_API_URL` (default `http://localhost:8000`) which exposes Track 1
    (contract field extraction) and Track 2 (RAG question answering) with
    full cost / latency / quality instrumentation.
    """
)

api = OP5Api()
st.sidebar.header("Service status")

try:
    health = api.health()
    cols = st.sidebar.columns(2)
    with cols[0]:
        st.metric("Status", health.get("status", "?"))
        st.metric("Keys", health.get("keys_configured", 0))
    with cols[1]:
        st.metric("Backend", health.get("storage_backend", "?"))
        st.metric("Embedder", health.get("embedder_status", "?")[:30])
    st.sidebar.success(f"Connected to {api.base_url}")
except Exception as exc:
    st.sidebar.error(f"Cannot reach {api.base_url}: {exc}")
    st.warning(
        "Start the FastAPI service first:\n\n"
        "```\nconda activate vsf\nuvicorn scripts.api.serve:app --port 8000 --reload\n```"
    )

st.markdown("---")

st.subheader("What this demo shows")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """**Track 1 — Extraction**

        Contract field extractor with 5 configurations:
        - A: zero-shot, no redaction, cheap model
        - B: few-shot, no redaction
        - D: zero-shot + PII redaction
        - B+D, D-strong: compress + strong model

        → Navigate to the **Track 1 Demo** page to upload a PDF.
        """
    )

with col2:
    st.markdown(
        """**Track 2 — RAG Q&A**

        Question answering over 15 SYNTH VinFast contracts with 7 configurations
        combining prompt compression (B), BM25 rerank (C) and router (D).

        → Navigate to the **Track 2 RAG** page to ask questions.
        """
    )

with col3:
    st.markdown(
        """**Inspect + Pareto**

        Compare cost vs quality across configurations by reading the JSONL eval logs
        and rendering the live Pareto plots from Phase 03.

        → Navigate to the **Pareto / Inspect** page.
        """
    )

st.markdown("---")
st.subheader("Run end-to-end smoke test")
st.code(
    "python scripts/phase-03/smoke_e2e.py",
    language="bash",
)
st.caption("Runs 6 checks: health, extract, ask, inspect, cost > 0, Pareto plot.")
