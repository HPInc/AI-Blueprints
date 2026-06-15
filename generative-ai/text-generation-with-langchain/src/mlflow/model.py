"""
Standalone Model class.

Business Logic Layer
- Handles arXiv paper extraction, analysis, and script generation pipeline
- Manages model initialization, vectorDB creation, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


class Model:
    """
    Standalone model class containing all business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        docs_path: str = None,
        model_path: str = None,
        secrets: Dict[str, Any] = None,
    ):
        """
        Universal constructor compatible with MLflow models-from-code loader.
        Initializes LLM internally based on config and model_path.

        Args:
            config: Model configuration dictionary
            docs_path: Path to documents directory (unused in text-generation)
            model_path: Path to model file (for LLM initialization)
            secrets: Secrets dictionary (unused in text-generation)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets
        self.LOCAL_LOGGING_ACTIVE = False  # Default value

        # Initialize logging
        self._setup_logging()

        # Initialize LLM based on config and model_path
        self.llm = self._initialize_llm()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_llm(self):
        """Initialize LLM based on config and model_path."""
        from src.utils import configure_hf_cache, configure_proxy
        from langchain_core.callbacks import CallbackManager
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain_community.llms import LlamaCpp
        import glob
        import os

        if hasattr(LlamaCpp, "model_rebuild"):
            LlamaCpp.model_rebuild()

        # Use provided model_path or find *.gguf file
        if self.model_path and os.path.exists(self.model_path):
            model_file_path = self.model_path
        else:
            # Fallback: look for *.gguf files in various locations
            search_paths = []
            if self.model_path:
                search_paths.append(os.path.dirname(self.model_path))
            if hasattr(self, "docs_path") and self.docs_path:
                models_dir = os.path.join(os.path.dirname(self.docs_path), "models")
                search_paths.append(models_dir)

            model_file_path = None
            for search_path in search_paths:
                if os.path.exists(search_path):
                    model_files = glob.glob(os.path.join(search_path, "*.gguf"))
                    if model_files:
                        model_file_path = model_files[0]
                        break

            if not model_file_path:
                raise RuntimeError(
                    f"No *.gguf model file found. Searched paths: {search_paths}"
                )

        self.logger.info(f"Using model file: {model_file_path}")

        configure_hf_cache()
        configure_proxy(self.config)

        start = time.perf_counter()
        llm = LlamaCpp(
            model_path=model_file_path,
            n_gpu_layers=-1,  # 0 → CPU-only
            n_batch=256,
            n_ctx=4096,
            max_tokens=1024,
            f16_kv=True,
            temperature=0,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            streaming=False,
            use_mmap=False,
        )
        self.logger.info("🔹 LlamaCpp loaded in %.1fs", time.perf_counter() - start)
        return llm

    @staticmethod
    def _create_arxiv_searcher(query: str, max_results: int, download: bool):
        """Create ArxivSearcher instance."""
        from core.extract_text.arxiv_search import ArxivSearcher

        # The download parameter is ignored since ArxivSearcher doesn't use it
        return ArxivSearcher(
            query=query,
            max_results=max_results,
            logging_enabled=True,  # Enable logging for better debugging
        )

    def _build_vectordb(self, papers: List[dict], chunk: int, overlap: int):
        """Build vector database from papers."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        uid = hashlib.md5(
            ("|".join(sorted(p["title"] for p in papers)) + str(chunk)).encode()
        ).hexdigest()[:10]
        path = Path(".vectordb") / uid
        path.mkdir(parents=True, exist_ok=True)

        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings()
        except ImportError:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. Please ensure sentence-transformers "
                "is installed with: pip install sentence-transformers"
            )

        if any(path.iterdir()):
            return Chroma(persist_directory=str(path), embedding_function=embeddings)

        docs = [
            Document(page_content=p["text"], metadata={"title": p["title"]})
            for p in papers
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk, chunk_overlap=overlap
        )
        chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=str(path))
        db.persist()
        return db

    def _summarise(self, papers, prompt, chunk, overlap):
        """Summarize papers using scientific paper analyzer."""
        from core.analyzer.scientific_paper_analyzer import ScientificPaperAnalyzer

        vectordb = self._build_vectordb(papers, chunk, overlap)
        analyzer = ScientificPaperAnalyzer(vectordb.as_retriever(), self.llm)
        return analyzer.analyze(prompt), analyzer.get_chain()

    def _generate_script(self, chain, prompt):
        """Generate script using script generator."""
        from core.generator.script_generator import ScriptGenerator

        generator = ScriptGenerator(
            chain=chain, use_local_logging=self.LOCAL_LOGGING_ACTIVE
        )
        generator.add_section(name="user_prompt", prompt=prompt)

        stdin_backup, builtins.input = builtins.input, lambda *_a, **_kw: "y"
        try:
            generator.run()
        finally:
            builtins.input = stdin_backup

        return generator.get_final_script()

    def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.
        """
        DEFAULT_SCRIPT_PROMPT = (
            "You are an academic writing assistant. Produce a short, well-structured "
            "presentation script covering:\n"
            "1. **Title** – concise and informative (add subtitle if helpful)\n"
            "2. **Introduction** – brief context, relevance and objectives\n"
            "3. **Methodology** – design, data and analysis used\n"
            "4. **Results** – key findings (mention figures/tables if relevant)\n"
            "5. **Conclusion** – main takeaway and implications\n"
            "6. **References** – properly formatted citations\n\n"
            "Write natural English prose; avoid numbered lists unless required. "
            "Return only the script – no extra commentary."
        )

        results: List[dict] = []

        for idx, row in model_input.iterrows():
            do_extract = bool(row.get("do_extract", True))
            do_analyse = bool(row.get("do_analyze", True))
            do_generate = bool(row.get("do_generate", True))

            query = row["query"]
            k = int(row.get("max_results", 3))
            chunk = int(row.get("chunk_size", 1200))
            overlap = int(row.get("chunk_overlap", 400))
            analysis_prompt = row.get(
                "analysis_prompt", "Summarise the content in ≈150 Portuguese words."
            )
            generation_prompt = (
                row.get("generation_prompt") or DEFAULT_SCRIPT_PROMPT
            ).strip()

            logging.info(
                "(row %d) extract=%s | analyse=%s | generate=%s — %s",
                idx,
                do_extract,
                do_analyse,
                do_generate,
                query,
            )

            papers = self._create_arxiv_searcher(
                query, k, do_extract
            ).search_and_extract()

            if do_extract and not (do_analyse or do_generate):
                results.append(
                    {
                        "extracted_papers": json.dumps(papers, ensure_ascii=False),
                        "script": "",
                    }
                )
                continue

            summary, chain = ("", None)
            if do_analyse or do_generate:
                summary, chain = self._summarise(
                    papers, analysis_prompt, chunk, overlap
                )

            if do_analyse and not do_generate:
                results.append(
                    {
                        "extracted_papers": json.dumps(papers, ensure_ascii=False),
                        "script": summary,
                    }
                )
                continue

            script = (
                self._generate_script(chain, generation_prompt)
                if do_generate and chain
                else ""
            )
            results.append(
                {
                    "extracted_papers": json.dumps(papers, ensure_ascii=False),
                    "script": script or summary,
                }
            )

        return pd.DataFrame(results)
