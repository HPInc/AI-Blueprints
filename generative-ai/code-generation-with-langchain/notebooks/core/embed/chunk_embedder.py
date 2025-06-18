

from __future__ import annotations

from typing import List, Dict, Any, Sequence, Callable

import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkEmbedder:
    """
    Parameters
    ----------
    model_name : str
        Any Sentence-Transformers model ID (default: all-MiniLM-L6-v2).
    batch_size : int
        Batch size for embedding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        model = SentenceTransformer(model_name)
        self._encode: Callable[[Sequence[str]], List[List[float]]] = lambda batch: model.encode(
            list(batch),
            batch_size=batch_size,
            show_progress_bar=False,
        )

    # ─────────────────────  Public API  ───────────────────── #

    def embed_repo_chunks(
        self, repo_chunks: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        repo_chunks  →  [
            {file_path, chunk_idx, text, embedding(np.ndarray)}, ...
        ]
        """
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for path, chunks in repo_chunks.items():
            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                metas.append({"file_path": path, "chunk_idx": idx, "text": chunk})

        emb_matrix = np.asarray(self._encode(texts), dtype="float32")

        return [
            {**metas[i], "embedding": emb_matrix[i]} for i in range(len(texts))
        ]


