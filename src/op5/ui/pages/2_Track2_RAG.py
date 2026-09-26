"""Track 2 RAG demo — pick case + question + config, call /track2/ask."""

from __future__ import annotations

import streamlit as st

from op5.ui.lib.api_client import OP5Api

st.set_page_config(page_title="OP5 — Track 2 RAG", layout="wide")

st.title("Track 2 — RAG Question Answering")

api = OP5Api()

try:
    cases = api.track2_cases()
    configs = api.track2_configs()
except Exception as exc:
    st.error(f"Cannot reach {api.base_url}: {exc}")
    st.stop()

if not cases:
    st.warning("No SYNTH cases indexed yet. Run scripts/phase-03/run_rag_track2.py once to build the index.")
    st.stop()

col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("Inputs")
    case_id = st.selectbox("Case", options=cases, index=0)
    question = st.text_input(
        "Question",
        value="Hợp đồng có hiệu lực từ ngày nào?",
        help="In Vietnamese or English. Out-of-scope questions will trigger refusal.",
    )
    config = st.selectbox("Config", options=configs, index=configs.index("A") if "A" in configs else 0)
    top_k = st.slider("Top-k chunks", min_value=1, max_value=16, value=8)
    ask = st.button("Ask", type="primary", use_container_width=True)

with col_result:
    st.subheader("Result")
    if ask:
        try:
            r = api.track2_ask(case_id=case_id, question=question, config=config, top_k=top_k)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        if r.get("error"):
            st.warning(f"LLM returned error: {r['error']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost (USD)", f"${r.get('cost_usd', 0):.6f}")
        m2.metric("Latency (ms)", f"{r.get('latency_ms', 0):.0f}")
        m3.metric("Input tokens", r.get("input_tokens", 0))
        m4.metric("Output tokens", r.get("output_tokens", 0))

        st.markdown(f"**model:** `{r.get('model')}`")

        if r.get("refused"):
            st.warning(f"**Refused** ({r.get('refusal_reason') or 'unspecified'})")
        else:
            st.markdown("##### Answer")
            st.write(r.get("answer", "(empty)"))

        cites = r.get("citations") or []
        if cites:
            st.markdown("##### Citations")
            for c in cites:
                st.code(c, language="text")

        chunks = r.get("context_chunks") or []
        if chunks:
            with st.expander(f"Show context chunks ({len(chunks)})"):
                for c in chunks:
                    st.json(c)

        with st.expander("Show raw API response"):
            st.json(r)
    else:
        st.info("Pick a case + question on the left and click Ask.")

st.markdown("---")
st.caption(
    "Configs: A=zero-shot/cheap, B=few-shot, C=BM25 rerank, D=router, "
    "B+D, C+D, B+C+D — combinations of the three levers."
)
