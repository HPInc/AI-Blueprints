# ─────── Standard Library Imports ───────
import json  # JSON parsing and serialization
import logging  # Logging utilities
from pathlib import Path  # Object-oriented filesystem paths
from typing import (
    Any,
    Dict,
    Optional,
)  # Type annotations for mappings and optional values
from threading import Lock


class SimpleKVMemory:
    """
    Very small persistent key-value store (JSON on disk).
    Thread-safe for the single-process runtime used by LangGraph nodes.

    • Keys   : arbitrary strings – in this project we use
               "file_id :: question" (lower-cased, stripped).
    • Values : any JSON-serialisable object, typically
               {"answer": "...", "evidence": [...]}

    """

    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self._lock: Lock = Lock()
        self._store: Dict[str, Any] = self._load()

    # ---------- public ----------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        """Return answer if present, else None."""
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        """Save answer and flush to disk."""
        with self._lock:
            self._store[key] = value
            self._dump()

    def clear(self) -> None:
        """Drop all cached entries (useful in tests)."""
        with self._lock:
            self._store.clear()
            self._dump()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)

    # ---------- private ---------------------------------------------------
    def _load(self) -> Dict[str, str]:
        if self.file_path.exists():
            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logging.warning(
                        "Memory file is not a valid JSON object. Starting fresh."
                    )
            except Exception as exc:
                logging.warning("Failed to load memory (%s). Starting fresh.", exc)
        return {}

    def _dump(self) -> None:
        """Flush the current store to disk."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)


# ---------- module-level helpers (used by graph/pyfunc) -------------------
def _mem_get(mem: Any, key: str) -> Optional[Any]:
    """
    Polymorphic get:
      - dict-like -> dict.get
      - object with `.get(key)` -> use it
      - otherwise None
    """
    if mem is None:
        return None
    if isinstance(mem, dict):
        return mem.get(key)
    get = getattr(mem, "get", None)
    if callable(get):
        return get(key)
    return None


def _mem_put(mem, key, value):
    """
    Polymorphic put:
      - dict-like -> mem[key] = value
      - object with `.set(key, value)` -> use it
      - object with `.put(key, value)` -> use it
    No locking here – SimpleKVMemory.set uses an internal lock.
    """
    if mem is None:
        raise RuntimeError("No memory object provided")
    if isinstance(mem, dict):
        mem[key] = value
        return
    set_fn = getattr(mem, "set", None)
    if callable(set_fn):
        set_fn(key, value)
        return
    put_fn = getattr(mem, "put", None)
    if callable(put_fn):
        put_fn(key, value)
        return
    raise RuntimeError("Unsupported memory object: missing set/put methods")
