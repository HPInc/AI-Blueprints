# src/components.py

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from typing import Any, Dict, List, Optional

# LangChain and vectorstore imports
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Transformer imports for SigLIP
from transformers import SiglipModel, SiglipProcessor
import logging
logger = logging.getLogger("multimodal_rag_register_notebook")


class SemanticCache:
    """
    A semantic cache using a Chroma vector store to find similar queries.
    """
    def __init__(self, persist_directory: Path, embedding_function: Embeddings, collection_name: str = "multimodal_cache"):
        self.embedding_function = embedding_function
        self._vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=self.embedding_function
        )

    def get(self, query: str, threshold: float = 0.90) -> Optional[Dict[str, Any]]:
        """
        Searches for a semantically similar query in the cache.
        """
        if self._vectorstore._collection.count() == 0:
            return None # Cache is empty

        results = self._vectorstore.similarity_search_with_score(query, k=1)
        if not results:
            return None

        most_similar_doc, score = results[0]
        # Chroma's 'score' is a distance metric (L2), so we convert it to similarity
        similarity = 1.0 - score

        logger.info(f"Most similar cached query: '{most_similar_doc.page_content}' (Similarity: {similarity:.4f})")

        if similarity >= threshold:
            cached_result = most_similar_doc.metadata
            # Deserialize the JSON string for used_images
            cached_result['used_images'] = json.loads(cached_result.get('used_images', '[]'))
            return cached_result

        return None

    def set(self, query: str, result_dict: Dict[str, Any]) -> None:
        """
        Adds a new query and its result to the cache.
        """
        # Ensure used_images is stored as a JSON string
        metadata_to_store = {
            'reply': result_dict.get('reply', ''),
            'used_images': json.dumps(result_dict.get('used_images', []))
        }

        doc = Document(page_content=query, metadata=metadata_to_store)
        self._vectorstore.add_documents([doc])
        logger.info(f"Added query to semantic cache: '{query}'")

    def delete(self, query: str) -> None:
        """
        Finds and deletes a query and its cached response from the vector store.
        """
        # Note: ChromaDB's `get` with `where` is the intended way to find documents by content.
        # This operation might be slow on large collections without proper indexing on metadata.
        results = self._vectorstore.get(where={"page_content": query})
        if results and 'ids' in results and results['ids']:
            doc_id_to_delete = results['ids'][0]
            self._vectorstore._collection.delete(ids=[doc_id_to_delete])
            logger.info(f"Cleared old cache for query: '{query}'")


class SiglipEmbeddings(Embeddings):
    """LangChain compatible wrapper for SigLIP image/text embeddings."""
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.model = SiglipModel.from_pretrained(model_id).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(model_id)

    def _embed_text(self, txts: List[str]) -> np.ndarray:
        inp = self.processor(text=txts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inp).cpu().numpy()

    def _embed_imgs(self, paths: List[str]) -> np.ndarray:
        imgs = [PILImage.open(p).convert("RGB") for p in paths]
        inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inp).cpu().numpy()

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self._embed_imgs(docs).tolist()

    def embed_query(self, txt: str) -> List[float]:
        return self._embed_text([txt])[0].tolist()