"""
db.py — SQLite persistence layer for the AI Chatbot.

Schema
------
conversations
    id            TEXT  PRIMARY KEY
    title         TEXT  NOT NULL
    system_prompt TEXT  NOT NULL
    created_at    TEXT  NOT NULL          -- ISO-8601 string
    messages      TEXT  NOT NULL          -- JSON array of {role, content} dicts
    updated_at    TEXT  NOT NULL          -- ISO-8601 string, updated on every save

The database file is created automatically next to this module (chatbot.db).
No external dependencies — only Python's built-in sqlite3.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# ── Database location ─────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "chatbot.db"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and friendly AI assistant specializing in explaining AI and "
    "machine learning concepts clearly. Use clear language and real-world analogies."
)

# ── Schema ────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    messages      TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    """Open (or create) the database and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the conversations table if it does not already exist."""
    with _connect() as conn:
        conn.executescript(_DDL)


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_conversations() -> dict:
    """
    Return all conversations as an ordered dict keyed by id, newest first.

    Each value is a plain Python dict with keys:
        id, title, system_prompt, created_at, messages (list)
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, system_prompt, created_at, messages "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return {
        row["id"]: {
            "id": row["id"],
            "title": row["title"],
            "system_prompt": row["system_prompt"],
            "created_at": row["created_at"],
            "messages": json.loads(row["messages"]),
        }
        for row in rows
    }


def save_conversation(conv: dict) -> None:
    """
    Insert or update a conversation in the database.

    Accepts the same dict shape used throughout main.py:
        {id, title, system_prompt, created_at, messages}
    """
    init_db()
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, system_prompt, created_at, updated_at, messages)
            VALUES (:id, :title, :system_prompt, :created_at, :updated_at, :messages)
            ON CONFLICT(id) DO UPDATE SET
                title         = excluded.title,
                system_prompt = excluded.system_prompt,
                updated_at    = excluded.updated_at,
                messages      = excluded.messages
            """,
            {
                "id": conv["id"],
                "title": conv.get("title", "New Chat"),
                "system_prompt": conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                "created_at": conv.get("created_at", now),
                "updated_at": now,
                "messages": json.dumps(conv.get("messages", []), ensure_ascii=False),
            },
        )


def new_conversation() -> dict:
    """Create, persist, and return a fresh conversation dict."""
    conv = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    save_conversation(conv)
    return conv


def new_conversation_local() -> dict:
    """
    Create and return a fresh conversation dict WITHOUT persisting it to the DB.

    Use this when starting a new chat session.  The conversation is saved to the
    database only when the first message is sent (via save_conversation).
    """
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


def delete_conversation(conv_id: str) -> None:
    """Delete a single conversation by ID. No-op if the ID does not exist."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def delete_all_conversations() -> None:
    """Delete every conversation from the database."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM conversations")
