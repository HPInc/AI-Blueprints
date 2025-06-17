# app.py
import json
import io
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="⚙️📊🦙 Automated Evaluation with Structured Outputs",
    page_icon="🦙",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# 1 ▸ Sidebar – server settings & runtime params
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️  Model API Settings")

api_url = st.sidebar.text_input(
    "MLflow /invocations URL",
    value="https://localhost:5000/invocations",
    help="Endpoint where the MLflow model is served."
)

st.sidebar.markdown("---")
st.sidebar.header("📄 Runtime parameters")

key_column   = st.sidebar.text_input("Key column", value="BoothNumber")
eval_column  = st.sidebar.text_input("Text column", value="AbstractText")

criteria_default = [
    "Originality", "ScientificRigor", "Clarity",
    "Relevance", "Feasibility", "Brevity"
]
criteria_str = st.sidebar.text_area(
    "Criteria (JSON list)",
    value=json.dumps(criteria_default, indent=2),
    height=120,
)
batch_size = st.sidebar.number_input(
    "Batch size", min_value=1, max_value=100, value=5, step=1
)

# Validate criteria JSON
try:
    criteria_list = json.loads(criteria_str)
    assert isinstance(criteria_list, list) and all(isinstance(c, str) for c in criteria_list)
    crit_valid = True
except Exception as e:
    crit_valid = False
    st.sidebar.error(f"Invalid criteria JSON → {e}")

# ─────────────────────────────────────────────────────────────
# 2 ▸ Main – data input
# ─────────────────────────────────────────────────────────────
st.title("⚙️📊🦙 Automated Evaluation with Structured Outputs")

st.markdown(
"""
Upload a **CSV** (or paste table) containing at least the chosen  
*key column* and *text column*. Adjust the parameters in the sidebar, then press **Evaluate**.
"""
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
raw_text = st.text_area("…or paste CSV/TSV content here", height=150)

df: pd.DataFrame | None = None
if uploaded:
    df = pd.read_csv(uploaded)
elif raw_text.strip():
    try:
        df = pd.read_csv(io.StringIO(raw_text))
    except Exception as e:
        st.error(f"Could not parse pasted data: {e}")

if df is not None:
    st.subheader("Preview of input data")
    st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 3 ▸ Call the model
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Evaluate", disabled=df is None or not crit_valid):
    if df is None:
        st.error("Please upload or paste a dataset.")
    elif key_column not in df.columns or eval_column not in df.columns:
        st.error(f"Dataset must contain columns **{key_column}** and **{eval_column}**.")
    else:
        with st.spinner("Scoring with Llama…"):
            payload = {
                "dataframe_split": df.to_dict(orient="split"),
                "params": {
                    "key_column":  key_column,
                    "eval_column": eval_column,
                    "criteria":    json.dumps(criteria_list),
                    "batch_size":  batch_size
                }
            }
            try:
                res = requests.post(api_url, json=payload, timeout=600, verify=False)
                res.raise_for_status()
                # MLflow returns a JSON list of dicts by default

                raw = res.json()
                st.session_state["raw"] = raw

                # ── NEW robust unpacking ─────────────────────────────
                if isinstance(raw, dict) and "predictions" in raw:
                    records = raw["predictions"]            # MLflow's default wrapper
                elif isinstance(raw, list):
                    records = raw                           # already a list of dicts
                else:
                    st.error("Unexpected response format"); st.json(raw); st.stop()

                result_df = pd.json_normalize(records)      # flattens dicts into columns

                st.session_state["last_results_df"] = result_df
                # ─────────────────────────────────────────────────────

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed:\n```\n{e}\n```")
            except ValueError as e:
                st.error(f"Could not decode response:\n{e}")

# Offer download
if "last_results_df" in st.session_state:
    st.success("Scoring complete!")
    st.subheader("Results")
    st.dataframe(st.session_state["last_results_df"], use_container_width=True)

    with st.expander("🔍 Raw JSON response"):
        st.json(st.session_state["raw"], expanded=False)

    csv_bytes = st.session_state["last_results_df"].to_csv(index=False).encode()

    st.download_button(
        label="📥 Download CSV",
        data=csv_bytes,
        file_name="llamascore_results.csv",
        mime="text/csv",
        key="download-results"          # stable key prevents widget rebuild issues
    )



# ─────────────────────────────────────────────────────────────
# 4 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
"""
*⚙️📊🦙 Automated Evaluation with Structured Outputs © 2025* – local, private, reproducible text evaluation with LLaMA + MLflow.

---
> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
""",
unsafe_allow_html=True,
)