import json
import requests
import pandas as pd
import streamlit as st
import zipfile
from io import BytesIO
import difflib
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(page_title="Markdown Corrector AI", page_icon="🤖", layout="wide")
st.title("📚 Markdown Grammar Corrector")
st.markdown("Enter a public GitHub repository URL to scan and correct grammar in all `.md` files.")

# --- Sidebar for API Configuration ---
st.sidebar.header("⚙️ API Configuration")
api_url = st.sidebar.text_input(
    "MLflow Model /invocations URL",
    value="https://localhost:55919/invocations",
    help="The endpoint where your MLflow model is served."
)

# --- Main App Interface ---
repo_url = st.text_input("GitHub Repository URL", placeholder="e.g., https://github.com/HPInc/AI-Blueprints/tree/main")

if st.button("🚀 Correct Markdown", disabled=not repo_url):
    if not api_url:
        st.error("Please provide the MLflow API URL in the sidebar.")
        st.stop()

    # send repo_url to your model
    input_df = pd.DataFrame([{"repo_url": repo_url}])
    payload = {"dataframe_split": input_df.to_dict("split")}

    try:
        with st.spinner("Processing repository…"):
            res = requests.post(api_url, json=payload, timeout=None, verify=False)
            res.raise_for_status()
            preds = res.json().get("predictions", [])
            if not preds:
                st.error("No predictions returned.")
                st.stop()

            raw = preds[0]
            # if your model somehow still wrapped it in JSON string:
            if isinstance(raw, str):
                raw = json.loads(raw)

            if not isinstance(raw, dict) or "corrected" not in raw or "originals" not in raw:
                st.error("❌ Unexpected response structure. Expected keys `corrected` and `originals`.")
                st.json(raw)
                st.stop()

            st.session_state["corrected_files"] = raw["corrected"]
            st.session_state["original_files"]  = raw["originals"]
            st.session_state["last_repo_url"]  = repo_url

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        st.stop()

# --- Display Results ---
corrected_files = st.session_state.get("corrected_files", {})
original_files  = st.session_state.get("original_files", {})

if corrected_files:
    st.success(f"Successfully processed repository: {st.session_state['last_repo_url']}")
    st.subheader("📁 Corrected Markdown Files")

    file_names = sorted(corrected_files.keys())
    selected = st.selectbox("📄 Choose a Markdown File", file_names)

    if selected:
        orig = original_files.get(selected, "").splitlines()
        corr = corrected_files[selected].splitlines()
        differ = difflib.HtmlDiff(wrapcolumn=80)
        diff_html = differ.make_file(orig, corr, fromdesc="Original", todesc="Corrected", context=True, numlines=3)

        st.markdown("### ✨ Side‑by‑Side Comparison")
        components.html(diff_html, height=600, scrolling=True)

    # Zip & download corrected versions
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for path, txt in corrected_files.items():
            z.writestr(path, txt)
    zip_buf.seek(0)

    st.download_button(
        label="📦 Download Corrected Markdown",
        data=zip_buf,
        file_name="corrected_markdown.zip",
        mime="application/zip"
    )
