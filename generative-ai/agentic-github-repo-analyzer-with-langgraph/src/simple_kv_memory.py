# ─────── Standard Library Imports ───────
import json  # JSON parsing and serialization
import logging  # Logging utilities
from pathlib import Path  # Object-oriented filesystem paths
from typing import Dict, Optional  # Type annotations for mappings and optional values



class SimpleKVMemory:
    """Very small persistent key-value store (JSON on disk)."""

    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self._store: Dict[str, str] = self._load()

    # ---------- public ----------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        """Return answer if present, else None."""
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        """Save answer and flush to disk."""
        self._store[key] = value
        self._dump()

    # ---------- private ---------------------------------------------------
    def _load(self) -> Dict[str, str]:
        if self.file_path.exists():
            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  
                logger.warning("Failed to load memory (%s). Starting fresh.", exc)
        return {}

    def _dump(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)