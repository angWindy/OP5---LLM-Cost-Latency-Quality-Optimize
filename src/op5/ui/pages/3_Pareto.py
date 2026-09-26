"""Pareto / Inspect page — read JSONL eval logs and show cost-vs-quality trade-offs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from op5.ui.lib.api_client import OP5Api

st.set_page_config(page_title="OP5 — Pareto / Inspect", layout="wide")

st.title("Pareto / Inspect — Cost vs Quality")

api = OP5Api()

st.header("Inspect a specific case")

try:
    cases = api.track2_cases()
except Exception as exc:
    st.error(f"Cannot reach {api.base_url}: {exc}")
    cases = []

if not cases:
    st.info("No cases available yet.")
else:
    case_id = st.selectbox("Case", options=cases, index=0, key="inspect_case")
    if st.button("Inspect", key="btn_inspect"):
        try:
            ins = api.inspect(case_id)
        except Exception as exc:
            st.error(f"Inspect failed: {exc}")
            st.stop()

        st.markdown(f"**case_id:** `{ins.get('case_id')}`")
        s = ins.get("summary") or {}
        if s:
            st.markdown("##### Aggregated cost / latency per config")
            st.json(s)

        with st.expander("Show track1 rows"):
            st.json(ins.get("track1", []))
        with st.expander("Show track2 rows"):
            st.json(ins.get("track2", []))

st.markdown("---")

st.header("Phase 03 Pareto plots")

t1_path = Path("doc/figs/phase-03-track1.png")
t2_path = Path("doc/figs/phase-03-track2.png")
t1_live = Path("doc/figs/phase-03-track1-live.png")
t2_live = Path("doc/figs/phase-03-track2-live.png")

cols = st.columns(2)
with cols[0]:
    st.markdown("**Track 1 — Extraction (cost vs field_F1)**")
    if t1_live.exists():
        st.image(str(t1_live), caption="Live run (--llm gemini)", use_container_width=True)
    if t1_path.exists():
        st.image(str(t1_path), caption="Stub run", use_container_width=True)
    if not (t1_live.exists() or t1_path.exists()):
        st.info("Track 1 Pareto PNG not found. Run scripts/phase-03/pareto_plot.py to generate.")

with cols[1]:
    st.markdown("**Track 2 — RAG (cost vs token_F1)**")
    if t2_live.exists():
        st.image(str(t2_live), caption="Live run (--llm gemini)", use_container_width=True)
    if t2_path.exists():
        st.image(str(t2_path), caption="Stub run", use_container_width=True)
    if not (t2_live.exists() or t2_path.exists()):
        st.info("Track 2 Pareto PNG not found. Run scripts/phase-03/pareto_plot.py to generate.")

st.markdown("---")
st.caption(
    "Pareto plots are rendered by `scripts/phase-03/pareto_plot.py` from "
    "`results/phase-03-track*-scores.jsonl`. The Streamlit page only displays them — "
    "regenerate by re-running the score + pareto scripts."
)
