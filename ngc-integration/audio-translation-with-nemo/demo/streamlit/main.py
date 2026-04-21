import streamlit as st
import requests

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nemo Text Translation",
    page_icon="📝",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Header styling */
.header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(90deg, #76B900, #3A5F00);
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.header h1 {
    color: white !important;
    margin: 0;
    font-size: 2.5rem;
}
.header p {
    color: #E0F2F1 !important;
    margin: 0;
    font-size: 1.1rem;
}

/* Button styling */
.stButton>button {
    background-color: #76B900;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #5A8A00;
}

/* Result box styling */
.result-box {
    background: #E8F5E9;
    border-left: 6px solid #76B900;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.5;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header">
        <h1>📝 Nemo Text Translation</h1>
        <p>Enter text below and get its translation via our MLflow model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar Instructions ────────────────────────────────────────────────────
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. The model endpoint is automatically configured for deployment.

    2. Type or paste the text to translate.

    3. Click **Translate** to see the result.
    """)

# ─── MLflow Endpoint Configuration ───────────────────────────────────────────
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"

# ─── Text Input ───────────────────────────────────────────────────────────────
text_to_translate = st.text_area(
    "✏️ Enter text to translate",
    height=200,
    placeholder="Type your source text here...",
)

# ─── Translate Button ─────────────────────────────────────────────────────────
translate = st.button("🚀 Translate")

if translate:
    # — Validate inputs —
    if not text_to_translate.strip():
        st.warning("Please enter some text to translate.")
    else:
        # — Build MLflow payload —
        payload = {
            "dataframe_records": [
                {"source_text": text_to_translate, "source_serialized_audio": ""}
            ],
            "parameters": {"use_audio": False},
        }

        try:
            # — Send request —
            with st.spinner("Translating…"):
                resp = requests.post(
                    MLFLOW_ENDPOINT, json=payload, verify=False, timeout=30
                )
                resp.raise_for_status()
                result = resp.json()

            # — Extract response record —
            rec = (
                result.get("predictions")
                or result.get("dataframe_records")
                or result.get("data")
            )
            if isinstance(rec, list):
                rec = rec[0]

            original = rec.get("original_text", "")
            translated = rec.get("translated_text", "")

            # — Display results —
            st.markdown(
                f"""
            <div class="result-box">
                <strong>🔍 Original:</strong><br>{original}
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="result-box">
                <strong>💡 Translation:</strong><br>{translated}
            </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Translation request failed: {e}")

# ────────────────  FOOTER  ─────────────────

st.warning(
    "Disclaimer: This application is provided for demonstration and illustrative purposes only. "
    "It does not represent a fully optimized or production-grade solution. "
    "Outputs may not be accurate, complete, or suitable for real-world decision-making. "
    "Results can often be improved by modifying the underlying code, models, data sources, and configuration."
)

st.write("Built with ❤️ using HP AI Studio")
