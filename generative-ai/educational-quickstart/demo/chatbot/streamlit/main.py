"""
AI Learning Playground — Chatbot Demo

Focused Streamlit frontend for the AIStudio-EQ-Chatbot model.
This app sends requests to the registered ChatbotModel via the MLflow invocations endpoint.

Features:
    - Persistent multi-turn conversation memory (stored in chatbot.db via SQLite)
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
from pathlib import Path

import requests
import streamlit as st

from db import (
    delete_all_conversations,
    delete_conversation,
    load_all_conversations,
    new_conversation,
    new_conversation_local,
    save_conversation,
)

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
    "<h2>🎨 AI Chatbot</h2>"
    "<p>Conversational AI powered by Zephyr 7B</p>"
    "</div>",
    unsafe_allow_html=True,
)
# ───────────────────────────── Conversation Storage ─────────────────────────────
# All persistence is handled by db.py (SQLite).  The three functions imported
# above — load_all_conversations, save_conversation, new_conversation — are the
# only storage calls made from this file.

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly AI assistant specializing in explaining AI and "
    "machine learning concepts clearly. Use clear language and real-world analogies."
)


def _derive_title(messages: list) -> str:
    """Derive a short title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"][:48].strip() + (
                "…" if len(msg["content"]) > 48 else ""
            )
    return "New Chat"


def _build_prompt_history_html(conv: dict) -> str:
    """Build an HTML block showing system prompt + all conversation turns."""
    system_prompt = conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    messages = conv.get("messages", [])

    entries = [
        f'<div class="ph-entry ph-system">'
        f'<span class="ph-role">system</span>'
        f'<div class="ph-content">{html.escape(system_prompt)}</div>'
        f"</div>"
    ]
    for msg in messages:
        role = msg.get("role", "user")
        content = html.escape(msg.get("content", ""))
        entries.append(
            f'<div class="ph-entry ph-{role}">'
            f'<span class="ph-role">{role}</span>'
            f'<div class="ph-content">{content}</div>'
            f"</div>"
        )

    body = (
        "".join(entries)
        if messages
        else (
            "".join(entries[:1])  # still show system prompt
            + '<p class="ph-empty">No messages yet.</p>'
        )
    )
    return f'<div class="prompt-history-panel">{body}</div>'


# ───────────────────────────── Session State Bootstrap ─────────────────────────
if "conversations" not in st.session_state:
    # Load from DB and immediately purge any empty conversations that were
    # left behind by prior crashes or interrupted sessions.
    _loaded = load_all_conversations()
    for _cid, _conv in list(_loaded.items()):
        if not _conv.get("messages"):
            delete_conversation(_cid)
            del _loaded[_cid]
    st.session_state.conversations = _loaded

if "active_conv_id" not in st.session_state:
    if st.session_state.conversations:
        st.session_state.active_conv_id = next(iter(st.session_state.conversations))
    else:
        # Create a local-only (unsaved) placeholder — it is persisted only when
        # the user actually sends their first message.
        first = new_conversation_local()
        st.session_state.conversations[first["id"]] = first
        st.session_state.active_conv_id = first["id"]

# Purge any in-memory stale empty conversations that aren't the active one.
_stale = [
    cid
    for cid, conv in list(st.session_state.conversations.items())
    if not conv.get("messages") and cid != st.session_state.active_conv_id
]
for _cid in _stale:
    delete_conversation(_cid)  # no-op if it was never persisted
    del st.session_state.conversations[_cid]

# ───────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    # ── Usage instructions ──
    st.title("⚙️ Usage")
    st.markdown("""
**Instructions:**
1. (Optional) Customize the system prompt below.
2. Type your question and press **Enter**.
3. The assistant will respond with an AI-generated answer.
""")

    st.divider()

    # ── System prompt (editable, per conversation) ──
    active_conv = st.session_state.conversations.get(
        st.session_state.active_conv_id, {}
    )
    new_system_prompt = st.text_area(
        "System Prompt",
        value=active_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        height=140,
        help="Edit the AI persona for this conversation. Changes apply to the next message.",
        key=f"sysprompt_{st.session_state.active_conv_id}",
    )
    # Persist system-prompt edits immediately (only if already saved to DB)
    if new_system_prompt != active_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT):
        active_conv["system_prompt"] = new_system_prompt
        if active_conv.get("messages"):
            save_conversation(active_conv)

    st.divider()

    # ── New Chat button ──
    if st.button(
        "＋  New Chat", use_container_width=True, key="new_chat_btn", type="primary"
    ):
        current = st.session_state.conversations.get(
            st.session_state.active_conv_id, {}
        )
        if not current.get("messages"):
            # Already on an empty chat — nothing to do
            st.rerun()
        else:
            conv = new_conversation_local()
            st.session_state.conversations = {
                conv["id"]: conv,
                **st.session_state.conversations,
            }
            st.session_state.active_conv_id = conv["id"]
            st.rerun()

    # ── Conversation list — radio styled as plain rows ──
    st.markdown("**Conversations**")
    conv_ids = list(st.session_state.conversations.keys())
    listed_ids = [
        cid for cid in conv_ids if st.session_state.conversations[cid].get("messages")
    ]
    if listed_ids:
        if st.session_state.active_conv_id not in listed_ids:
            # The active conversation is a new empty chat (not yet in the list).
            # Clear conv_radio so the radio widget doesn't retain a stale
            # selection that would snap back to a previous conversation and
            # swallow the user's first submitted message.
            st.session_state.pop("conv_radio", None)
            current_index = None
        else:
            current_index = listed_ids.index(st.session_state.active_conv_id)
            # Sync the widget key only when it holds a stale value (not in
            # the current option list).  When the user clicks a conversation,
            # Streamlit updates conv_radio to that valid ID BEFORE the script
            # reruns — overwriting it here would swallow the click.  A stale
            # value (None, deleted ID, etc.) means active_conv_id was changed
            # programmatically (e.g. first message on a new chat) and the
            # widget needs to catch up.
            if st.session_state.get("conv_radio") not in listed_ids:
                st.session_state["conv_radio"] = st.session_state.active_conv_id

        # Build unique labels: when two conversations share the same title,
        # Streamlit's radio can't distinguish them by displayed text and the
        # click gets swallowed.  Append a counter suffix to make them unique.
        from collections import Counter as _Counter

        _raw_labels = [
            st.session_state.conversations[cid].get("title") or "New Chat"
            for cid in listed_ids
        ]
        _label_count = _Counter(_raw_labels)
        _label_seen: dict[str, int] = {}
        _unique_label: dict[str, str] = {}
        for cid, label in zip(listed_ids, _raw_labels):
            if _label_count[label] > 1:
                _label_seen[label] = _label_seen.get(label, 0) + 1
                _unique_label[cid] = f"{label} ({_label_seen[label]})"
            else:
                _unique_label[cid] = label

        selected_id = st.radio(
            "",
            options=listed_ids,
            format_func=lambda cid: _unique_label[cid],
            index=current_index,
            label_visibility="collapsed",
            key="conv_radio",
        )
        if selected_id and selected_id != st.session_state.active_conv_id:
            st.session_state.active_conv_id = selected_id
            st.rerun()

    # ── Clear All History button (with inline confirmation) ──
    st.divider()
    st.markdown('<div class="clear-all-marker"></div>', unsafe_allow_html=True)
    if not st.session_state.get("confirm_clear_all"):
        if st.button(
            "🗑  Clear All History", use_container_width=True, key="clear_all_btn"
        ):
            st.session_state["confirm_clear_all"] = True
            st.rerun()
    else:
        st.warning("This will permanently delete all conversations.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button(
                "Delete all",
                key="confirm_yes",
                use_container_width=True,
                type="primary",
            ):
                delete_all_conversations()
                st.session_state.conversations = {}
                conv = new_conversation_local()
                st.session_state.conversations[conv["id"]] = conv
                st.session_state.active_conv_id = conv["id"]
                st.session_state["confirm_clear_all"] = False
                st.rerun()
        with col_no:
            if st.button("Cancel", key="confirm_no", use_container_width=True):
                st.session_state["confirm_clear_all"] = False
                st.rerun()

# ───────────────────────────── Resolve Active Conversation ─────────────────────
active_conv = st.session_state.conversations.get(st.session_state.active_conv_id)
if active_conv is None:
    active_conv = new_conversation_local()
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
st.markdown("### \U0001f4ac Ask the AI Tutor")

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

            # Persist the new turn (also handles first-time DB insert for new convs)
            active_conv["messages"].append({"role": "user", "content": question})
            active_conv["messages"].append({"role": "assistant", "content": answer})
            active_conv["title"] = _derive_title(active_conv["messages"])
            save_conversation(active_conv)
            # Refresh sidebar conversation list and rerun to update titles.
            # active_conv_id is already correct; the radio widget derives its
            # selected index from active_conv_id on the next rerun, so there
            # is no need (and it would be illegal) to set the widget-key
            # conv_radio after the widget has already been rendered.
            st.session_state.conversations[active_conv["id"]] = active_conv
            st.rerun()
        else:
            st.error(result["error"])

# -------------------------FOOTER-----------------------------------------------------
st.warning(
    "Disclaimer: This application is provided for demonstration and illustrative purposes only. "
    "It does not represent a fully optimized or production-grade solution. "
    "Outputs may not be accurate, complete, or suitable for real-world decision-making. "
    "Results can often be improved by modifying the underlying code, models, data sources, and configuration."
)

st.write("Built with ❤️ using HP AI Studio")
