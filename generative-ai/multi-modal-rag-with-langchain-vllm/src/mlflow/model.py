"""
Standalone Model class.

Business Logic Layer
- Handles multimodal RAG-based question answering with document and image retrieval
- Manages model initialization, embeddings, vector database, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import gc
import json
import base64
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import pandas as pd
from PIL import Image as PILImage
from rank_bm25 import BM25Okapi

# LangChain and vectorstore imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Transformer imports for SigLIP and Qwen
from transformers import AutoImageProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

# Project-specific imports
from src.components import SiglipEmbeddings
from src.wiki_pages_clone import orchestrate_wiki_clone
from src.local_genai_judge import LocalGenAIJudge
from src.utils import load_mm_docs_clean

import logging

logger = logging.getLogger("multimodal_rag_model")


class QwenVLMM:
    """Minimal, self-contained multimodal QA wrapper."""

    def __init__(
        self,
        llm: LLM,
        tok: AutoTokenizer,
        image_processor: AutoImageProcessor,
        device: str,
        text_db: Chroma,
        image_db: Chroma,
        bm25_index: Optional[BM25Okapi],
        doc_map: dict,
    ):
        self.llm = llm
        self.tok = tok
        self.image_processor = image_processor
        self.device = device
        self.text_db = text_db
        self.image_db = image_db
        self.bm25_index = bm25_index
        self.doc_map = doc_map

    # Retreval
    @staticmethod
    def _reciprocal_rank_fusion(
        results: list[list[Document]], k: int = 60
    ) -> list[tuple[Document, float]]:
        """Performs Reciprocal Rank Fusion on multiple ranked lists of documents."""
        ranked_lists = [
            {doc.page_content: (doc, i + 1) for i, doc in enumerate(res)}
            for res in results
        ]
        rrf_scores = defaultdict(float)
        all_docs = {}

        for ranked_list in ranked_lists:
            # Iterate through each ranked list and calculate RRF scores
            for content, (doc, rank) in ranked_list.items():
                rrf_scores[content] += 1 / (k + rank)
                if content not in all_docs:
                    all_docs[content] = doc

        fused_results = [
            (all_docs[content], rrf_scores[content])
            for content in sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        ]
        return fused_results

    # Retrieval
    def _retrieve_mm(
        self, query: str, k_text: int = 3, k_img: int = 2, recall_k: int = 20
    ) -> dict[str, any]:
        """Retrieves relevant documents and images based on the query using both dense and sparse retrieval methods."""
        dense_hits = self.text_db.similarity_search(query, k=recall_k)

        # If no dense hits, try sparse retrieval with BM25
        sparse_hits = []
        if self.bm25_index and list(self.doc_map.keys()):
            tokenized_query = query.lower().split(" ")
            sparse_texts = self.bm25_index.get_top_n(
                tokenized_query, list(self.doc_map.keys()), n=recall_k
            )
            sparse_hits = [self.doc_map[text] for text in sparse_texts]

        # Combine and rerank using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion([dense_hits, sparse_hits])
        selected_docs = [doc for doc, score in fused_results[:k_text]]

        # Image retrieval using SigLIP embeddings
        image_hits = self.image_db.similarity_search(query, k=k_img)
        selected_images = [hit.page_content for hit in image_hits] if image_hits else []

        return {
            "documents": selected_docs,
            "images": selected_images,
            "text_sources": [
                doc.metadata.get("source", "Unknown") for doc in selected_docs
            ],
            "image_sources": (
                [hit.metadata.get("source", "Unknown") for hit in image_hits]
                if image_hits
                else []
            ),
        }

    def generate(self, query: str) -> dict[str, any]:
        """Generates a response using the multimodal RAG pipeline."""
        start_gen_time = time.time()

        # Retrieve relevant documents and images
        retrieval_results = self._retrieve_mm(query)
        documents = retrieval_results["documents"]
        images = retrieval_results["images"]
        referenced_sources = list(
            set(retrieval_results["text_sources"] + retrieval_results["image_sources"])
        )

        # Prepare context from retrieved documents
        context_str = (
            "\n\n".join([doc.page_content for doc in documents])
            if documents
            else "No relevant context found."
        )

        # System prompt for the multimodal RAG task
        system_prompt = """You are an AI assistant that helps answer questions using provided context and images. Please provide accurate and helpful responses based on the given information.

When answering:
1. Use the provided context and images to answer the question
2. Be concise but comprehensive
3. If the context doesn't contain relevant information, say so clearly
4. Reference specific details from the context when applicable"""

        # Prepare user content with image tokens if images are available
        if images:
            image_tokens = ""
            for i in range(len(images)):
                image_tokens += f"<|vision_start|><|image_pad|><|vision_end|>"

            user_content = f"""{image_tokens}

<context>
{context_str}
</context>

<user_query>
{query}
</user_query>"""
        else:
            user_content = f"""<context>
{context_str}
</context>

<user_query>
{query}
</user_query>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_string = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            self._clear_cuda()
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=2048
            )

            # Process images if available
            if images:
                pil_images = []
                for i, img_path in enumerate(images):
                    try:
                        img = PILImage.open(img_path).convert("RGB")
                        # Resize large images while preserving aspect ratio
                        if img.size[0] > 512 or img.size[1] > 512:
                            img.thumbnail((512, 512), PILImage.Resampling.LANCZOS)
                        pil_images.append(img)
                        logger.info(
                            f"Processed image {i+1}: {img_path} -> new size {img.size}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_path}: {e}")
                        continue

                # If no images were successfully processed, proceed with text-only request
                if not pil_images:
                    logger.warning(
                        "No images successfully processed, proceeding text-only"
                    )
                    request_payload = {"prompt": prompt_string}
                else:
                    request_payload = {
                        "prompt": prompt_string,
                        "multi_modal_data": {"image": pil_images},
                    }
            else:
                request_payload = {"prompt": prompt_string}

            # Generate the response using the LLM
            output_list = self.llm.generate(
                request_payload, sampling_params=sampling_params
            )
            reply = (
                output_list[0].outputs[0].text.strip()
                if output_list and output_list[0].outputs
                else "Error: no output from LLM."
            )

            self._clear_cuda()
            end_gen_time = time.time()

            # If the reply is empty or indicates no relevant context, handle it gracefully
            if (
                reply
                == "The provided context does not contain relevant information to answer the query."
            ):
                images = []
                referenced_sources = []

            return {
                "reply": reply,
                "used_images": images,
                "referenced_sources": referenced_sources,
                "generation_time_seconds": end_gen_time - start_gen_time,
            }

        except RuntimeError as e:
            logger.error("Qwen-VL generation failed: %s", e)
            return {
                "reply": f"Error during generation: {e}",
                "used_images": images,
                "referenced_sources": referenced_sources,
                "generation_time_seconds": 0.0,
            }

    def _clear_cuda(self):
        """Clears CUDA memory."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class Model:
    """
    Standalone model class containing all multimodal RAG business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(
        self,
        local_model_dir: str,
        e5_model_dir: str,
        siglip_model_dir: str,
        config: dict,
    ):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.
        """
        logger.info("--- Initializing Stateless MultimodalRAG Service (Qwen-VL) ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        logger.info("--- Service initialized with a single locked collection. ---")
        model_path = Path(local_model_dir).resolve()

        logger.info("Loading text embedding model (E5)...")
        self.text_embed_model = HuggingFaceEmbeddings(
            model_name=e5_model_dir, model_kwargs={"device": self.device}
        )

        logger.info("Loading image embedding model (SigLIP)...")
        self.siglip_embed_model = SiglipEmbeddings(
            model_id=siglip_model_dir, device=self.device
        )

        logger.info("Loading main LLM (Qwen-VL via vLLM)...")
        base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.tok = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.image_processor = AutoImageProcessor.from_pretrained(
            base_model_name, trust_remote_code=True, use_fast=True
        )

        if self.device == "cuda":
            self.llm = LLM(
                model=str(model_path),
                quantization="gptq",
                gpu_memory_utilization=0.60,
                max_model_len=4096,
                enforce_eager=True,
                limit_mm_per_prompt={"image": 2},
                disable_custom_all_reduce=True,
                tensor_parallel_size=1,
                dtype="float16",
            )
            logger.info("Initializing LocalGenAIJudge for self-evaluation...")
            self.judge = LocalGenAIJudge(llm=self.llm, tokenizer=self.tok)
        else:
            self.llm = None
            self.judge = None  # Judge is None if no CUDA LLM
            logger.error(
                "Qwen-VL with vLLM requires a CUDA device. LLM and Judge not loaded."
            )

        self.db_lock = threading.Lock()
        self.text_collection_name = "rag_text_collection"
        self.image_collection_name = "rag_image_collection"

        # Use a persistent client that lives with the model
        self.chroma_client = chromadb.Client()
        self.text_vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.text_collection_name,
            embedding_function=self.text_embed_model,
        )
        self.image_vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.image_collection_name,
            embedding_function=self.siglip_embed_model,
        )

        # Persistent KB state
        self.bm25_index = None
        self.doc_map = None
        self.kb_dir = None
        self.rag_pipeline = None

        logger.info(
            "--- Service initialized with all models loaded. KB will build on first sync. ---"
        )


    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.
        """
        query = model_input["query"].iloc[0]
        payload = json.loads(model_input["payload"].iloc[0])

        # Build or refresh the kb
        if query == "update_kb":
            with self.db_lock:
                try:
                    self._initialize_kb(
                        config=payload["config"], secrets=payload["secrets"]
                    )

                    # Warmup query to trigger Triton JIT compilation
                    logger.info("Running warmup query to pre-compile Triton kernels...")
                    self.rag_pipeline.generate("What is this wiki about?")
                    logger.info("Warmup complete.")

                    return pd.DataFrame(
                        [{"status": "success", "message": "Knowledge base loaded."}]
                    )
                except Exception as e:
                    logger.error(f"KB initialization failed: {e}", exc_info=True)
                    return pd.DataFrame([{"status": "error", "message": str(e)}])

        # Process the query
        with self.db_lock:
            logger.info("Received new query. Lock acquired.")
            pipeline_start_time = time.time()
            actual_query = payload.get("query", query)

            # Validate input DataFrame
            if self.rag_pipeline is None:
                return pd.DataFrame([{
                    "status": "error",
                    "message": "Knowledge base not loaded. Please sync first.",
                }])
            if not self.llm:
                return pd.DataFrame([{"status": "error", "message": "LLM not loaded."}])

            try:
                response_dict = self.rag_pipeline.generate(actual_query)

                logger.info("Performing self-evaluation with LocalGenAIJudge...")
                if self.judge:
                    context_str = "\n\n".join(
                        d.page_content
                        for d in self.text_vector_store.similarity_search(actual_query, k=3)
                    )
                    eval_df = pd.DataFrame([{
                        "questions": actual_query,
                        "result": response_dict["reply"],
                        "source_documents": context_str,
                    }])
                    response_dict["faithfulness"] = self.judge.evaluate_faithfulness(eval_df).iloc[0]
                    response_dict["relevance"] = self.judge.evaluate_relevance(eval_df).iloc[0]
                    response_dict["conciseness"] = self.judge.evaluate_conciseness(eval_df).iloc[0]
                else:
                    response_dict["faithfulness"] = -1.0
                    response_dict["relevance"] = -1.0
                    response_dict["conciseness"] = -1.0

                base64_images = []
                for path in response_dict.get("used_images", []):
                    try:
                        with open(path, "rb") as img_file:
                            base64_images.append(
                                base64.b64encode(img_file.read()).decode("utf-8")
                            )
                    except FileNotFoundError:
                        logger.warning(f"Image file not found: {path}")
                response_dict["used_images"] = json.dumps(base64_images)
                response_dict["referenced_sources"] = json.dumps(
                    response_dict.get("referenced_sources", [])
                )
                response_dict["total_pipeline_time_seconds"] = (
                    time.time() - pipeline_start_time
                )

                logger.info("Request finished. Releasing lock.")
                return pd.DataFrame([response_dict])

            except Exception as e:
                logger.error(f"RAG pipeline failed: {e}", exc_info=True)
                return pd.DataFrame([{"status": "error", "message": str(e)}])

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


    def _collect_image_vectors(self, mm_raw_docs: List[Document], image_dir: Path):
        """Scans raw docs and returns paths, IDs, and metadata for unique images."""
        img_paths, img_ids, img_meta = [], [], []
        seen = set()

        # Ensure the image directory exists, process all images in the directory
        for doc in mm_raw_docs:
            src = doc.metadata["source"]
            for name in doc.metadata.get("images", []):
                img_id = f"{src}::{name}"
                if img_id in seen:
                    continue
                seen.add(img_id)
                img_path = image_dir / name
                if img_path.is_file():
                    img_paths.append(str(img_path))
                    img_ids.append(img_id)
                    img_meta.append({"source": src, "image": name})
        return img_paths, img_ids, img_meta


    def _chunk_docs(self, docs: List[Document]) -> List[Document]:
        """Takes a list of raw docs and performs chunking with unique IDs per doc."""
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")]
        )
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        all_chunks: list[Document] = []

        # Process each document, splitting by headers and then recursively splitting sections
        for doc in docs:
            page_title = Path(doc.metadata["source"]).stem.replace("-", " ")
            section_docs = header_splitter.split_text(doc.page_content)
            doc_chunk_counter = 0
            for section in section_docs:
                tiny_texts = recursive_splitter.split_text(section.page_content)
                for tiny in tiny_texts:
                    chunk_metadata = {
                        "title": page_title,
                        "source": doc.metadata["source"],
                        "section_header": section.metadata.get("header", ""),
                        "chunk_id": doc_chunk_counter,
                    }
                    all_chunks.append(
                        Document(
                            page_content=f"{page_title}\n\n{tiny.strip()}",
                            metadata=chunk_metadata,
                        )
                    )
                    doc_chunk_counter += 1
        return all_chunks


    def _initialize_kb(self, config: dict, secrets: dict) -> None:
        """Builds the knowledge base once and stores it persistently on the instance."""
        new_kb_dir = Path(tempfile.mkdtemp(prefix="kb_"))
        logger.info(f"Building persistent KB in {new_kb_dir}...")

        orchestrate_wiki_clone(
            pat=secrets["AIS_ADO_TOKEN"], config=config, output_dir=new_kb_dir
        )

        image_dir = new_kb_dir / "images"
        wiki_metadata_path = new_kb_dir / "wiki_flat_structure.json"
        if not wiki_metadata_path.exists():
            raise FileNotFoundError("Cloning failed: 'wiki_flat_structure.json' not found.")

        all_raw_docs = load_mm_docs_clean(wiki_metadata_path, image_dir)
        all_chunks = self._chunk_docs(all_raw_docs)
        logger.info(f"Loaded {len(all_chunks)} text chunks")

        # Wipe + repopulate text collection (makes update_kb a true refresh)
        existing_text_ids = self.text_vector_store._collection.get(include=[])["ids"]
        if existing_text_ids:
            self.text_vector_store._collection.delete(ids=existing_text_ids)
        if all_chunks:
            self.text_vector_store.add_documents(documents=all_chunks)

        # Wipe + repopulate image collection
        img_paths, img_ids, img_meta = self._collect_image_vectors(all_raw_docs, image_dir)
        existing_image_ids = self.image_vector_store._collection.get(include=[])["ids"]
        if existing_image_ids:
            self.image_vector_store._collection.delete(ids=existing_image_ids)
        if img_paths:
            self.image_vector_store.add_texts(texts=img_paths, metadatas=img_meta, ids=img_ids)

        # BM25 + doc map live on the instance now
        unique_splits = list({doc.page_content: doc for doc in all_chunks}.values())
        corpus = [doc.page_content for doc in unique_splits]
        self.bm25_index = BM25Okapi([d.split(" ") for d in corpus]) if corpus else None
        self.doc_map = {doc.page_content: doc for doc in unique_splits}

        # Build the RAG pipeline once, not per request
        self.rag_pipeline = QwenVLMM(
            llm=self.llm,
            tok=self.tok,
            image_processor=self.image_processor,
            device=self.device,
            text_db=self.text_vector_store,
            image_db=self.image_vector_store,
            bm25_index=self.bm25_index,
            doc_map=self.doc_map,
        )

        # Swap in new KB dir, clean up the old one on refresh
        if self.kb_dir and self.kb_dir.exists():
            shutil.rmtree(self.kb_dir, ignore_errors=True)
        self.kb_dir = new_kb_dir
        logger.info("Knowledge base initialization complete.")
