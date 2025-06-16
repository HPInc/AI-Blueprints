import streamlit as st
import requests
import base64

# ─── Page Config & Title ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vanilla RAG Chatbot Interface",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; color:#4B8BBE;'>🔮 Vanilla RAG Chatbot Service Interface</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#306998;'>Interact with your locally deployed MLflow Vanilla RAG Chatbot model.</p>",
    unsafe_allow_html=True
)

# ─── Sidebar with Instructions ────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown("""
    1. Ensure your MLflow model serving is running under HTTPS.
    2. Enter the full URL of the `/invocations` endpoint.
    3. (Optional) Upload a PDF to enrich the knowledge base.
    4. Type your query and an optional prompt, then Submit.
    
    **Note:** In **Z by HP AI Studio**, the **port number** for your MLflow API **changes with each deployment**, so always verify the correct URL and port before starting a session.
    """)

# ─── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ─── Main Form ────────────────────────────────────────────────────────────────
with st.form("chat_form"):
    api_url = st.text_input(
        "🔗 MLflow API URL",
        value="https://localhost:5000/invocations"
    )
    query = st.text_area(
        "💬 Your Query",
        height=150,
        placeholder="Type what you’d like to ask the chatbot about Z by HP AI Studio..."
    )
    # custom prompt moved under query
    prompt = st.text_area(
        "✏️ Custom Prompt (optional)",
        height=80,
        placeholder="E.g., “Answer in bullet points.”"
    )
    uploaded_file = st.file_uploader(
        "📄 Upload PDF (optional)",
        type="pdf"
    )
    # two buttons in separate columns
    col1, col2 = st.columns(2)
    submit = col1.form_submit_button("🚀 Submit Query")
    # there is a problem with reset functionality backend
    #reset = col2.form_submit_button("🔄 Reset Conversation")
    reset = None

# ─── Reset Logic ─────────────────────────────────────────────────────────────

if reset:
    try:
        payload = {
            "dataframe_records": [
                {"query": "", "prompt": "", "document": ""}
            ],
            "parameters": {"reset_history": True}
        }
        resp = requests.post(api_url, json=payload, verify=False)
        resp.raise_for_status()
        st.session_state.history = []
        st.session_state.chunks = []
        st.success("✅ Conversation history has been reset.")
    except Exception as e:
        st.error(f"❌ Reset failed: {e}")

# ─── Submission & Inference ───────────────────────────────────────────────────
if submit:
    # Basic validation
    if not api_url.lower().startswith(("http://", "https://")):
        st.error("Please enter a valid URL starting with http:// or https://")
    elif not query.strip() and not uploaded_file:
        st.warning("Enter a query or upload a PDF.")
    else:
        # Prepare document payload
        if uploaded_file:
            raw = uploaded_file.read()
            doc_b64 = base64.b64encode(raw).decode("utf-8")
            params = {"add_pdf": True}
        else:
            doc_b64 = ""
            params = {}

        payload = {
            "dataframe_records": [
                {"query": query, "prompt": prompt, "document": doc_b64}
            ]
        }
        if params:
            payload["parameters"] = params

        try:
            with st.spinner("Contacting model..."):
                resp = requests.post(api_url, json=payload, verify=False)
                resp.raise_for_status()
                result = resp.json()

            # extract the record
            rec = (result.get("predictions") or result.get("dataframe_records") or result.get("data"))[0]

            # update state
            st.session_state.history = rec.get("history", [])
            st.session_state.chunks = rec.get("chunks", [])

            # ─── Highlighted Answer ─────────────────────────
            st.markdown(
                "<h3 style='color:#00796B;'>🤖 Model Response</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:8px;
                    background: linear-gradient(90deg, #e0f7fa, #80deea);
                    color: #004d40;
                    font-size:16px;
                ">{rec.get("output", "No output returned.")}</div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            # ─── Relevant Chunks ──────────────────────────────
            if st.session_state.chunks:
                with st.expander("📚 Relevant Chunks", expanded=False):
                    for c in st.session_state.chunks:
                        st.write(c)

            # ─── Conversation History ────────────────────────
            if st.session_state.history:
                with st.expander("💬 Conversation History", expanded=False):
                    for msg in st.session_state.history:
                        st.write(msg)

        except Exception as e:
            st.error(f"❌ Inference error: {e}")
