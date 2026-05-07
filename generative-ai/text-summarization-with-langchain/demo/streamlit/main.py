# app.py
import json
import io
import requests
import streamlit as st
import pypdf
import docx
from typing import Optional, Union

st.set_page_config(
    page_title="📄✨ Text Summarization with AI Studio", page_icon="📄", layout="wide"
)

# ─────────────────────────────────────────────────────────────
# 1 ▸ Sidebar – runtime params
# ─────────────────────────────────────────────────────────────
# MLflow API Configuration
api_url = "http://localhost:5002/invocations"

st.sidebar.header("📋 Display Options")

show_original_text = st.sidebar.checkbox(
    "Show Original Text",
    value=False,
    help="Display the extracted text from uploaded documents",
)

show_api_details = st.sidebar.checkbox(
    "Show API Response Details",
    value=False,
    help="Display raw API response for debugging",
)

# ─────────────────────────────────────────────────────────────
# 2 ▸ Helper Functions
# ─────────────────────────────────────────────────────────────


def extract_text_from_file(uploaded_file) -> str:
    """Extract text content from various file formats."""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")

        elif uploaded_file.type == "application/pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text_content += f"Page {page_num}:\n{page_text}\n\n"
            return (
                text_content
                if text_content.strip()
                else "PDF text extraction complete. No text found or PDF contains images."
            )

        elif (
            uploaded_file.type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            doc = docx.Document(uploaded_file)
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            return (
                text_content
                if text_content.strip()
                else "No text could be extracted from the document."
            )

        else:
            raise Exception(f"Unsupported file format: {uploaded_file.type}")

    except Exception as e:
        raise Exception(f"Failed to extract text: {str(e)}")


# ─────────────────────────────────────────────────────────────
# 3 ▸ Main Interface
# ─────────────────────────────────────────────────────────────

st.title("📄✨ Text Summarization with AI Studio")

st.markdown("""
Upload a **document** (TXT, PDF, DOCX) or paste text directly to generate an AI-powered summary.
Adjust the parameters in the sidebar, then press **Summarize**.
""")

# Initialize session state
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = ""
if "api_response" not in st.session_state:
    st.session_state.api_response = None

# Input Methods
tab1, tab2 = st.tabs(["📎 File Upload", "✏️ Direct Text Input"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx"],
        help="Select a text file, PDF, or Word document to summarize",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("📖 Extracting text from file..."):
                extracted_text = extract_text_from_file(uploaded_file)
                st.session_state.extracted_text = extracted_text

            st.success(f"✅ Text extracted from {uploaded_file.name}")
            st.info(
                f"File: {uploaded_file.name} | Type: {uploaded_file.type} | Size: {uploaded_file.size / 1024:.1f} KB"
            )

        except Exception as e:
            st.error(f"❌ Error extracting text: {str(e)}")

with tab2:
    direct_text = st.text_area(
        "Enter text to summarize",
        height=200,
        placeholder="Paste your text here or type directly...",
    )

    if direct_text.strip():
        st.session_state.extracted_text = direct_text
        st.success("✅ Text ready for summarization")

# ─────────────────────────────────────────────────────────────
# 4 ▸ Original Text Display (if enabled)
# ─────────────────────────────────────────────────────────────

if show_original_text and st.session_state.extracted_text:
    st.subheader("📄 Original Text")
    st.text_area(
        "Extracted Text:",
        value=st.session_state.extracted_text,
        height=300,
        disabled=True,
        key="original_text_display",
    )

# ─────────────────────────────────────────────────────────────
# 5 ▸ Call the model
# ─────────────────────────────────────────────────────────────

if st.button("🚀 Summarize", disabled=not st.session_state.extracted_text.strip()):
    if not st.session_state.extracted_text.strip():
        st.error("❌ Please provide text to summarize")
    else:
        with st.spinner("🤖 Generating AI summary..."):
            payload = {
                "inputs": {"text": [st.session_state.extracted_text]},
                "params": {},
            }
            try:
                response = requests.post(
                    api_url, json=payload, timeout=300, verify=False
                )
                response.raise_for_status()

                api_response = response.json()
                st.session_state.api_response = api_response

                # Extract summary from response
                if "predictions" in api_response:
                    if (
                        isinstance(api_response["predictions"], list)
                        and len(api_response["predictions"]) > 0
                    ):
                        first_prediction = api_response["predictions"][0]
                        if (
                            isinstance(first_prediction, dict)
                            and "summary" in first_prediction
                        ):
                            summary = first_prediction["summary"]
                        else:
                            summary = str(first_prediction)
                    elif (
                        isinstance(api_response["predictions"], dict)
                        and "summary" in api_response["predictions"]
                    ):
                        summary = api_response["predictions"]["summary"]
                    else:
                        summary = "No summary provided by the model."
                else:
                    summary = "Invalid response format from model."

                st.session_state.summary_result = summary

            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Request failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")

# ─────────────────────────────────────────────────────────────
# 6 ▸ Results Display
# ─────────────────────────────────────────────────────────────

if st.session_state.summary_result:
    st.success("✅ Summary generated successfully!")
    st.subheader("✨ Generated Summary")

    st.text_area(
        "AI-Generated Summary:",
        value=st.session_state.summary_result,
        height=250,
        disabled=True,
        key="summary_display",
    )

    # Download button
    st.download_button(
        label="📥 Download Summary",
        data=st.session_state.summary_result,
        file_name="summary.txt",
        mime="text/plain",
    )

# API Response Details (if enabled)
if show_api_details and st.session_state.api_response:
    with st.expander("🔍 Raw JSON Response"):
        st.json(st.session_state.api_response, expanded=False)

# ─────────────────────────────────────────────────────────────
# 7 ▸ Footer
# ─────────────────────────────────────────────────────────────
st.warning(
    "Disclaimer: This application is provided for demonstration and illustrative purposes only. "
    "It does not represent a fully optimized or production-grade solution. "
    "Outputs may not be accurate, complete, or suitable for real-world decision-making. "
    "Results can often be improved by modifying the underlying code, models, data sources, and configuration."
)


st.markdown(
    """
*📄✨ Text Summarization with AI Studio © 2025* – Extract, process, and summarize documents with AI technology.

---
> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
""",
    unsafe_allow_html=True,
)
