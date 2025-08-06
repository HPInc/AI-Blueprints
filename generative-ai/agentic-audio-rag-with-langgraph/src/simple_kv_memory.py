# ─────── Standard Library Imports ───────
import json  # JSON parsing and serialization
import logging  # Logging utilities
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Dict, Optional  # Type annotations for mappings and optional values
from threading import Lock

class SimpleKVMemory:
    """
    Very small persistent key-value store (JSON on disk).
    Thread-safe for the single-process runtime used by LangGraph nodes.

    • Keys   : arbitrary strings – in this project we use  
               "file_id :: question" (lower-cased, stripped).  
    • Values : any JSON-serialisable object, typically  
               {"answer": <markdown>, "snippets": [...]}
    
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
                    return json.load(f)
            except Exception as exc:  
                logging.warning("Failed to load memory (%s). Starting fresh.", exc)
        return {}

    def _dump(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)