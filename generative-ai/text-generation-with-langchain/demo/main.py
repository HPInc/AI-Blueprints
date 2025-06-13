# app/paper_to_script_demo.py
# -*- coding: utf-8 -*-
"""
Streamlit front-end for the **Paper-to-Script** MLflow service.

• Queries arXiv, extracts PDFs, summarises them and (optionally) produces a
  short presentation script.
• The back-end must expose the standard MLflow `/invocations` endpoint.
"""
from __future__ import annotations

import json
import os
import textwrap
import time

import requests
import streamlit as st

# ────────────────────────────── constants ──────────────────────────────
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

DEFAULT_SCRIPT_PROMPT = (
    "You are an academic writing assistant. Produce a short, well-structured "
    "presentation script covering:\n"
    "1. **Title** – concise and informative (add subtitle if helpful)\n"
    "2. **Introduction** – brief context, relevance and objectives\n"
    "3. **Methodology** – design, data and analysis used\n"
    "4. **Results** – key findings (mention figures/tables if relevant)\n"
    "5. **Conclusion** – main takeaway and implications\n"
    "6. **References** – properly formatted citations\n\n"
    "Write natural English prose; avoid numbered lists unless required. "
    "Return only the script – no extra commentary."
)

# ──────────────────────────── page setup ───────────────────────────────
st.set_page_config(
    page_title="Paper-to-Script 🎬",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .stTabs [data-baseweb=tab] { font-weight: 600; padding: 8px 18px; }

      .result-box {
        background: var(--secondary-background);
        border-left: 6px solid var(--primary-color);
        padding: 1rem;
        border-radius: .5rem;
        margin-top: .8rem;
        max-height: 28rem;
        overflow-y: auto;
      }
      .paper-card {
        background: var(--secondary-background);
        padding: .8rem 1rem;
        border: 1px solid var(--primary-color);
        border-radius: .5rem;
        margin-bottom: .8rem;
      }
      code { white-space: pre-wrap !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style='text-align:center;margin-bottom:0'>
      🎓 <span style='color:var(--primary-color)'>Paper-to-Script</span> Demo 🎬
    </h1>
    <p style='text-align:center'>
      Generate summaries or short presentation scripts from scientific papers
      hosted on <a href="https://arxiv.org" target="_blank">arXiv.org</a>.
    </p>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────── sidebar (settings) ─────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_url = st.text_input(
        "MLflow endpoint URL (must end with `/invocations`)",
        value="https://localhost:5000/invocations",
    )

    st.markdown("---")
    st.subheader("Steps")
    col_a, col_b = st.columns(2)
    with col_a:
        do_extract  = st.checkbox("📥 Extract",  value=True)
        do_generate = st.checkbox("🎬 Script",   value=False)
    with col_b:
        do_analyze  = st.checkbox("🧐 Analyze",  value=True)

    st.markdown("---")
    st.subheader("Prompts")
    analysis_prompt = st.text_input(
        "Analysis prompt",
        value="Summarise the content in English (≈150 words).",
    )
    generation_prompt = st.text_area(
        "Initial script prompt (leave blank to use the default)",
        value=DEFAULT_SCRIPT_PROMPT,
        height=180,
    )

    st.markdown("---")
    st.caption(
        "⚡ Tip: uncheck *Extract* to use cached PDFs, or check **only** *Extract* "
        "to download new PDFs without analysing them."
    )

# ───────────────────────────── main form ───────────────────────────────
st.subheader("🔍 Search articles on arXiv")
query = st.text_input("Search term", value="graph neural networks")
cols = st.columns(3)
max_results   = cols[0].number_input("Number of articles",   1, 10, 3)
chunk_size    = cols[1].number_input("Chunk size",           200, 2000, 1200, step=100)
chunk_overlap = cols[2].number_input("Chunk overlap",          0,  800,  400, step=50)

# ───────────────────────────── submission ──────────────────────────────
if st.button("🚀 Run"):
    if not api_url.lower().startswith(("http://", "https://")):
        st.error("Invalid URL – it must start with http:// or https://.")
        st.stop()

    # Use the default prompt only when the field is empty
    generation_prompt_final = generation_prompt.strip() or DEFAULT_SCRIPT_PROMPT

    payload = {
        "inputs": {
            "query":             [query],
            "max_results":       [max_results],
            "chunk_size":        [chunk_size],
            "chunk_overlap":     [chunk_overlap],
            "do_extract":        [do_extract],
            "do_analyze":        [do_analyze],
            "do_generate":       [do_generate],
            "analysis_prompt":   [analysis_prompt],
            "generation_prompt": [generation_prompt_final],
        },
        "params": {},
    }

    # ─────────────────────── HTTP request ────────────────────────
    try:
        t0 = time.perf_counter()
        with st.spinner("Processing…"):
            response = requests.post(api_url, json=payload, verify=False, timeout=600)
            response.raise_for_status()
            result = response.json()
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    records = (
        result.get("predictions")
        or result.get("dataframe_records")
        or result.get("data")
        or result
    )[0]

    papers_json = records.get("extracted_papers", "[]")
    main_output = records.get("script", "")

    papers = json.loads(papers_json)

    tab_art, tab_sum, tab_script = st.tabs(["📚 Articles", "📝 Summary", "🎬 Script"])

    # ─────────────────────────── tab: articles ──────────────────────────
    with tab_art:
        if not papers:
            st.info("No articles extracted or cached.")
        for idx, paper in enumerate(papers, 1):
            with st.expander(f"{idx}. {paper['title']}"):
                st.markdown(
                    f"<div class='paper-card'><code>"
                    f"{textwrap.shorten(paper['text'], width=800, placeholder=' …')}"
                    f"</code></div>",
                    unsafe_allow_html=True,
                )

    # ─────────────────────────── tab: summary ───────────────────────────
    with tab_sum:
        if not do_analyze:
            st.info("Analysis is turned off.")
        elif do_analyze and do_generate:
            st.info(
                "When *Script* is enabled the service returns only the final script. "
                "Disable *Script* if you are interested in the summary alone."
            )
        else:
            if main_output:
                st.markdown(f"<div class='result-box'>{main_output}</div>",
                            unsafe_allow_html=True)
            else:
                st.info("No summary returned.")

    # ─────────────────────────── tab: script ────────────────────────────
    with tab_script:
        if not do_generate:
            st.info("Generation is turned off.")
        else:
            if main_output:
                st.markdown(f"<div class='result-box'>{main_output}</div>",
                            unsafe_allow_html=True)
            else:
                st.info("No script generated.")

    st.success(f"✓ Completed in {elapsed:.1f}s")
