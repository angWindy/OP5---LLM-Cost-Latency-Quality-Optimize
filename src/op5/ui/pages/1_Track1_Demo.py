"""Track 1 extraction demo — upload a PDF, choose config, call /track1/extract."""

from __future__ import annotations

import streamlit as st

from op5.ui.lib.api_client import OP5Api

st.set_page_config(page_title="OP5 — Track 1", layout="wide")

st.title("Track 1 — Contract Field Extraction")

api = OP5Api()

try:
    configs = api.track1_configs()
except Exception as exc:
    st.error(f"Cannot reach {api.base_url}: {exc}")
    st.stop()

col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("Inputs")
    pdf_path = st.text_input(
        "PDF path",
        value="data/Scan/scan_phase03/contract_synth-ctr-001.pdf",
        help="Absolute or repo-relative path to a scanned contract PDF.",
    )
    config = st.selectbox("Config", options=configs, index=configs.index("D") if "D" in configs else 0)
    use_ocr = st.checkbox("Run OCR (uncheck to use GT-derived text — smoke mode)", value=True)
    run = st.button("Extract", type="primary", use_container_width=True)

with col_result:
    st.subheader("Result")
    if run:
        try:
            t_resp = api.track1_extract(pdf_path=pdf_path, config=config, use_ocr=use_ocr)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        if t_resp.get("error"):
            st.warning(f"LLM returned error: {t_resp['error']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost (USD)", f"${t_resp.get('cost_usd', 0):.6f}")
        m2.metric("Latency (ms)", f"{t_resp.get('latency_ms', 0):.0f}")
        m3.metric("Input tokens", t_resp.get("input_tokens", 0))
        m4.metric("Output tokens", t_resp.get("output_tokens", 0))

        st.markdown(f"**case_id:** `{t_resp.get('case_id')}` &nbsp;&nbsp; **model:** `{t_resp.get('model')}`")

        st.markdown("##### Extracted fields")
        st.json(t_resp.get("extracted_fields", {}))

        with st.expander("Show redacted text (first 2000 chars)"):
            st.code(t_resp.get("redacted_text", ""), language="text")

        with st.expander("Show raw API response"):
            st.json(t_resp)
    else:
        st.info("Set inputs on the left and click Extract.")

st.markdown("---")
st.caption(
    "Configs from master plan §4.6: A=cheap/zero-shot/no-redact, B=cheap/few-shot/no-redact, "
    "D=cheap/zero-shot+redact, B+D=cheap/few-shot+redact, D-strong=strong/zero-shot+redact."
)
