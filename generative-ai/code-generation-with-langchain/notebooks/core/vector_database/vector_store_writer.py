# vector_store_writer.py
# ------------------------------------------------------------
# Escreve embeddings gerados localmente (ex.: ChunkEmbedder)
# no ChromaDB, aceitando **duas** entradas:
#
#   1. Uma lista de dicionários (ex.: `chunk_embeddings`)
#      [
#         {
#             "file_path": "...",
#             "chunk_idx": 0,
#             "text": "...",
#             "embedding": np.ndarray | list[float]
#         },
#         ...
#      ]
#
#   2. Um pandas.DataFrame que possua colunas:
#        ["ids", "code", "metadatas", "embeddings"]
#
# Não há dependência de OpenAI; tudo roda localmente.
# ------------------------------------------------------------

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Sequence

import chromadb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreWriter:
    """Persist chunk embeddings in a local ChromaDB instance."""

    def __init__(
        self,
        collection_name: str = "my_collection",
        persist_dir: str | Path = "./chroma_db",
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=str(persist_dir))
            logger.info(f"ChromaDB client using storage at: {persist_dir}")
        except Exception as e:
            logger.warning(f"Persistent client failed ({e}); using in-memory DB.")
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Ready to upsert into collection: '{collection_name}'")

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def upsert_chunk_embeddings(self, chunk_embeddings: Sequence[Dict[str, Any]]) -> None:
        """
        Accepts list[dict] 
        Each dict *must* contain keys: file_path, chunk_idx, text, embedding.
        """
        ids, docs, metas, embeds = [], [], [], []

        for rec in chunk_embeddings:
            try:
                fp = rec["file_path"]
                idx = rec["chunk_idx"]
                text = rec["text"]
                emb = rec["embedding"]

                uid = f"{fp}::{idx}"
                if isinstance(emb, np.ndarray):
                    emb = emb.astype(float).tolist()

                ids.append(uid)
                docs.append(text)
                metas.append({"file_path": fp, "chunk_idx": idx})
                embeds.append(emb)
            except KeyError as e:
                logger.error(f"Record missing key {e}: {rec}")
                continue

        self._upsert(ids, docs, metas, embeds)

    def upsert_dataframe(self, df: pd.DataFrame) -> None:
        """Maintains compatibility with DataFrame in the old format."""
        required = ["ids", "code", "metadatas", "embeddings"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame precisa das colunas: {required}")

        ids = df["ids"].astype(str).tolist()
        docs = df["code"].tolist()
        metas = df["metadatas"].tolist()
        embeds = [
            (row.tolist() if isinstance(row, np.ndarray) else row)
            for row in df["embeddings"]
        ]

        self._upsert(ids, docs, metas, embeds)

 
    def _upsert(
        self,
        ids: List[str],
        docs: List[str],
        metas: List[Dict[str, Any]],
        embeds: List[List[float]],
    ) -> None:
        if self.verbose:
            logger.info(f"Upserting {len(ids)} register…")

        try:
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeds,
            )
            logger.info("✅ Upsert completed with embeddings.")
        except Exception as e:
            logger.warning(f"Falhou com embeddings ({e}); tentando sem embedding.")
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info("✅ Upsert completed (Chroma generated embeddings).")




