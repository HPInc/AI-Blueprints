"""
AI Learning Playground — Document Analyzer Demo

Focused Streamlit frontend for the AIStudio-EQ-Document model.
This app sends requests to the registered DocumentModel via the MLflow invocations endpoint.

Input sent to model:
    question   — What you want to know about the document
    input_text — The document content (from file upload or text paste)

To start the MLflow model server:
    mlflow models serve -m models:/AIStudio-EQ-Document/1 -p 5002 --no-conda

Then launch this app:
    python -m poetry run streamlit run main.py
"""

import json
from pathlib import Path

import requests
import streamlit as st

# ───────────────────────────── Page Configuration ─────────────────────────────
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide",
)

# ───────────────────────────── CSS Styling ─────────────────────────────────────
css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ───────────────────────────── Logo Bar ────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if Path("static/HP-logo.png").exists():
        st.image("static/HP-logo.png", width=100)
with col2:
    if Path("static/Z-logo.png").exists():
        st.image("static/Z-logo.png", width=100)
with col3:
    if Path("static/AIS-logo.png").exists():
        st.image("static/AIS-logo.png", width=100)

# ───────────────────────────── Header ──────────────────────────────────────────
st.markdown(
    '<div class="gradient-header">'
    "<h2>📄 Document Analyzer</h2>"
    "<p>Upload documents and ask questions — powered by chunk-based RAG with LlamaCpp</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("⚙️ Usage")
st.sidebar.markdown(
    """
**Instructions:**
1. Paste or type your document text into the input field.
2. Enter your question about the document.
3. Click **Analyze Document** to receive an AI-generated answer.
"""
)

endpoint_url = "http://localhost:5002/invocations"


# ───────────────────────────── Helper: API Call ────────────────────────────────


def call_model(question: str, input_text: str, timeout: int = 600) -> dict:
    """
    Send a POST request to the DocumentModel's invocations endpoint.

    Payload schema — only the columns DocumentModel expects:
        question   (str)
        input_text (str)
    """
    payload = {
        "dataframe_records": [{"question": question, "input_text": input_text}],
    }
    try:
        response = requests.post(
            endpoint_url.strip(),
            json=payload,
            verify=False,
            timeout=timeout,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()["predictions"][0]}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": (
                "Cannot connect to the model server.\n\n"
                "Start it with:\n"
                "```bash\n"
                "mlflow models serve -m models:/AIStudio-EQ-Document/1 -p 5002 --no-conda\n"
                "```"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out — document analysis can take several minutes for long documents.",
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


# ───────────────────────────── Document Form ───────────────────────────────────
st.markdown(
    "### 📋 Analyze Documents\n\n"
    "Upload one or more text files and ask a question. "
    "Or paste document text directly in the text area below."
)

with st.form("document_form"):
    question = st.text_area(
        "❓ Question",
        height=80,
        value="What are the main themes and key points discussed in this document?",
    )

    col_upload, col_paste = st.columns([1, 1])

    with col_upload:
        uploaded_files = st.file_uploader(
            "📁 Upload Documents",
            accept_multiple_files=True,
            type=["txt", "csv", "md"],
            help="Supported formats: .txt, .csv, .md",
        )

    with col_paste:
        pasted_text = st.text_area(
            "📝 Or paste document text",
            height=200,
            placeholder="Paste document content here...",
        )

    submitted = st.form_submit_button("🔍 Analyze Document", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Build input_text from uploaded files and/or pasted text
        input_parts = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    input_parts.append(f"=== {uploaded_file.name} ===\n{content}")
                except Exception as e:
                    st.warning(f"⚠️ Could not read {uploaded_file.name}: {e}")

        if pasted_text.strip():
            input_parts.append(pasted_text.strip())

        if not input_parts:
            st.warning("Please upload a document or paste text to analyze.")
        else:
            combined_text = "\n\n".join(input_parts)

            with st.spinner(f"Analyzing document ({len(combined_text):,} chars)..."):
                result = call_model(question, combined_text[:50000])  # Cap at 50k chars

            if result["success"]:
                output = result["data"]
                st.markdown("### 📈 Analysis Result")
                st.markdown(
                    f"<div class='result-box'>{output['answer']}</div>",
                    unsafe_allow_html=True,
                )
                st.divider()
                with st.expander("🔍 View Analysis Details"):
                    try:
                        st.json(json.loads(output.get("messages", "[]")))
                    except json.JSONDecodeError:
                        st.text(output.get("messages", ""))
            else:
                st.error(result["error"])
