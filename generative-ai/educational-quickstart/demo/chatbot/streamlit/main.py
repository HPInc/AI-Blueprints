"""
AI Learning Playground — Chatbot Demo

Focused Streamlit frontend for the AIStudio-EQ-Chatbot model.
This app sends requests to the registered ChatbotModel via the MLflow invocations endpoint.

Input sent to model:
    question      — The user's message
    system_prompt — LLM persona (editable in the sidebar)

To start the MLflow model server:
    mlflow models serve -m models:/AIStudio-EQ-Chatbot/1 -p 5002 --no-conda

Then launch this app:
    python -m poetry run streamlit run main.py
"""

import base64
import json
from pathlib import Path

import requests
import streamlit as st

# ───────────────────────────── Page Configuration ─────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="wide",
)

# ───────────────────────────── CSS Styling ─────────────────────────────────────
css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ───────────────────────────── Logo Bar ────────────────────────────────────────
_logo_paths = [
    ("static/HP-logo.png", "HP"),
    ("static/Z-logo.png", "Z by HP"),
    ("static/AIS-logo.png", "AI Studio"),
]
_logo_imgs = "".join(
    f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" alt="{label}">'
    for p, label in _logo_paths
    if Path(p).exists()
)
st.markdown(f'<div class="logo-bar">{_logo_imgs}</div>', unsafe_allow_html=True)

# ───────────────────────────── Header ──────────────────────────────────────────
st.markdown(
    '<div class="gradient-header">'
    "<h2>💬 AI Chatbot</h2>"
    "<p>Conversational Q&A powered by Meta-Llama-3.1-8B running locally via llama.cpp</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ─────────────────────────────────────────
st.sidebar.title("⚙️ Usage")
st.sidebar.markdown(
    """
**Instructions:**
1. (Optional) Customize the system prompt below.
2. Type your question and click **Send**.
3. The assistant will respond with an AI-generated answer.
"""
)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value=(
        "You are a helpful and friendly AI assistant specializing in explaining AI and "
        "machine learning concepts clearly. Use clear language and real-world analogies."
    ),
    height=150,
)

endpoint_url = "http://localhost:5002/invocations"


# ───────────────────────────── Helper: API Call ────────────────────────────────


def call_model(question: str, system_prompt: str, timeout: int = 600) -> dict:
    """
    Send a POST request to the ChatbotModel's invocations endpoint.

    Payload schema — only the columns ChatbotModel expects:
        question      (str)
        system_prompt (str)
    """
    payload = {
        "dataframe_records": [{"question": question, "system_prompt": system_prompt}],
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
                "mlflow models serve -m models:/AIStudio-EQ-Chatbot/1 -p 5002 --no-conda\n"
                "```"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out — the model may still be loading. Try again.",
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


# ───────────────────────────── Chat Form ───────────────────────────────────────
st.markdown(
    "### 💬 Ask the AI Tutor\n\n"
    "Type any question about data science, machine learning, or AI concepts."
)

with st.form("chatbot_form"):
    question = st.text_area(
        "❓ Your Question",
        height=120,
        placeholder=(
            "e.g., What is the difference between supervised and unsupervised learning?\n"
            "e.g., Explain backpropagation in simple terms.\n"
            "e.g., What are the most common activation functions and when should I use each?"
        ),
    )
    submitted = st.form_submit_button("💬 Ask", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = call_model(question, system_prompt)

        if result["success"]:
            output = result["data"]
            st.markdown("### 🤖 Response")
            st.markdown(
                f"<div class='result-box'>{output['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            with st.expander("🔍 View Conversation Details"):
                try:
                    st.json(json.loads(output.get("messages", "[]")))
                except json.JSONDecodeError:
                    st.text(output.get("messages", ""))
        else:
            st.error(result["error"])
