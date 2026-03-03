"""
AI Learning Playground — Chatbot Demo

Focused Streamlit frontend for the AIStudio-EQ-Chatbot model.
This app sends requests to the registered ChatbotModel via the MLflow invocations endpoint.

Features:
    - Persistent multi-turn conversation memory (saved to conversations/ folder)
    - Each conversation has its own ID and system prompt
    - Sidebar lists past conversations; "+ New Chat" creates a new session
    - System prompt is editable per conversation

Input sent to model:
    question      — The user's message
    system_prompt — LLM persona (editable in the sidebar)
    history       — JSON array of prior {role, content} messages for multi-turn context

To start the MLflow model server:
    mlflow models serve -m models:/AIStudio-EQ-Chatbot/1 -p 5002 --no-conda

Then launch this app:
    python -m poetry run streamlit run main.py
"""

import base64
import json
import uuid
from datetime import datetime
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
    "<p>Conversational Q&A powered by Zephyr 7B Beta running locally via llama.cpp</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ───────────────────────────── Conversation Storage ────────────────────────────
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.chmod(0o777)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly AI assistant specializing in explaining AI and "
    "machine learning concepts clearly. Use clear language and real-world analogies."
)


def _conv_path(conv_id: str) -> Path:
    return CONVERSATIONS_DIR / f"{conv_id}.json"


def load_all_conversations() -> dict:
    """Load all saved conversations from disk, keyed by ID, sorted newest first."""
    convs = {}
    for f in sorted(CONVERSATIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            convs[data["id"]] = data
        except Exception:
            pass
    return convs


def save_conversation(conv: dict) -> None:
    """Persist a conversation dict to disk."""
    _conv_path(conv["id"]).write_text(
        json.dumps(conv, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def new_conversation() -> dict:
    """Create and persist a fresh conversation."""
    conv = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    save_conversation(conv)
    return conv


def _derive_title(messages: list) -> str:
    """Derive a short title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"][:48].strip() + ("…" if len(msg["content"]) > 48 else "")
    return "New Chat"


# ───────────────────────────── Session State Bootstrap ─────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = load_all_conversations()

if "active_conv_id" not in st.session_state:
    if st.session_state.conversations:
        st.session_state.active_conv_id = next(iter(st.session_state.conversations))
    else:
        first = new_conversation()
        st.session_state.conversations[first["id"]] = first
        st.session_state.active_conv_id = first["id"]
    # Force-sync the radio widget on first load so it starts on the right conversation
    st.session_state["force_conv_sync"] = True

# ───────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    # ── Usage instructions ──
    st.title("⚙️ Usage")
    st.markdown(
        """
**Instructions:**
1. (Optional) Customize the system prompt below.
2. Type your question and press **Enter**.
3. The assistant will respond with an AI-generated answer.
"""
    )

    st.divider()

    # ── System prompt (editable, per conversation) ──
    active_conv = st.session_state.conversations.get(st.session_state.active_conv_id, {})
    new_system_prompt = st.text_area(
        "System Prompt",
        value=active_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        height=140,
        help="Edit the AI persona for this conversation. Changes apply to the next message.",
        key=f"sysprompt_{st.session_state.active_conv_id}",
    )
    # Persist system-prompt edits immediately
    if new_system_prompt != active_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT):
        active_conv["system_prompt"] = new_system_prompt
        save_conversation(active_conv)

    st.divider()

    # ── New Chat button ──
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        conv = new_conversation()
        st.session_state.conversations = {conv["id"]: conv, **st.session_state.conversations}
        st.session_state.active_conv_id = conv["id"]
        # Force-sync the radio so it doesn't snap back to the previously selected item
        st.session_state["force_conv_sync"] = True
        st.rerun()

    # ── Conversation list (radio styled as a plain list) ──
    st.markdown("**Conversations**")
    conv_ids = list(st.session_state.conversations.keys())
    if conv_ids:
        # Only pre-sync the radio key when code explicitly changed the active conversation
        # (New Chat or initial load). Doing it unconditionally would swallow user clicks.
        if st.session_state.get("force_conv_sync"):
            st.session_state["conv_radio"] = st.session_state.active_conv_id
            st.session_state["force_conv_sync"] = False
        current_index = conv_ids.index(st.session_state.active_conv_id) if st.session_state.active_conv_id in conv_ids else 0
        selected_id = st.radio(
            "",
            options=conv_ids,
            format_func=lambda cid: st.session_state.conversations[cid].get("title", "New Chat"),
            index=current_index,
            label_visibility="collapsed",
            key="conv_radio",
        )
        if selected_id != st.session_state.active_conv_id:
            st.session_state.active_conv_id = selected_id
            st.session_state["force_conv_sync"] = False
            st.rerun()

# ───────────────────────────── Resolve Active Conversation ─────────────────────
active_conv = st.session_state.conversations.get(st.session_state.active_conv_id)
if active_conv is None:
    active_conv = new_conversation()
    st.session_state.conversations[active_conv["id"]] = active_conv
    st.session_state.active_conv_id = active_conv["id"]

endpoint_url = "http://localhost:5002/invocations"


# ───────────────────────────── Helper: API Call ────────────────────────────────


def call_model(
    question: str,
    system_prompt: str,
    history: list,
    timeout: int = 600,
) -> dict:
    """
    Send a POST request to the ChatbotModel's invocations endpoint.

    Payload schema — columns ChatbotModel expects:
        question      (str) — The user's message
        system_prompt (str) — LLM persona
        history       (str) — JSON-serialized list of prior {role, content} messages
    """
    payload = {
        "dataframe_records": [
            {
                "question": question,
                "system_prompt": system_prompt,
                "history": json.dumps(history),
            }
        ],
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


# ───────────────────────────── Chat Area ───────────────────────────────────────
st.markdown("### 💬 Ask the AI Tutor")

# Render conversation history as chat bubbles
for msg in active_conv.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input — no button needed, Enter submits
user_input = st.chat_input(
    "Type your question about data science, machine learning, or AI concepts…"
)

if user_input and user_input.strip():
    question = user_input.strip()
    system_prompt = active_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    prior_messages = active_conv.get("messages", [])

    # Immediately show the user's message in the chat
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = call_model(question, system_prompt, prior_messages)

        if result["success"]:
            answer = result["data"]["answer"]
            st.markdown(answer)

            # Persist the new turn
            active_conv["messages"].append({"role": "user", "content": question})
            active_conv["messages"].append({"role": "assistant", "content": answer})
            active_conv["title"] = _derive_title(active_conv["messages"])
            save_conversation(active_conv)
            # Refresh sidebar conversation list and rerun to update titles
            st.session_state.conversations[active_conv["id"]] = active_conv
            st.rerun()
        else:
            st.error(result["error"])
