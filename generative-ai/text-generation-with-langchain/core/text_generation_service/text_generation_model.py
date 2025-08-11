"""
Standalone TextGenerationModel class.

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


class TextGenerationModel:
    """
    Standalone model class containing all business logic.
    NO MLflow inheritance - pure domain functionality.
    """

    def __init__(self, llm, config: Dict[str, Any]):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.
        """
        self.llm = llm
        self.config = config
        self.LOCAL_LOGGING_ACTIVE = False  # Default value
        
        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _create_arxiv_searcher(query: str, max_results: int, download: bool):
        """Create ArxivSearcher instance."""
        import inspect
        from core.extract_text.arxiv_search import ArxivSearcher

        kwargs: Dict[str, Any] = {"query": query, "max_results": max_results}
        sig = inspect.signature(ArxivSearcher)  
        if "cache_only" in sig.parameters:
            kwargs["cache_only"] = not download
        elif "download" in sig.parameters:
            kwargs["download"] = download
        return ArxivSearcher(**kwargs)

    def _build_vectordb(self, papers: List[dict], chunk: int, overlap: int):
        """Build vector database from papers."""
        from langchain.schema import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        uid = hashlib.md5(
            ("|".join(sorted(p["title"] for p in papers)) + str(chunk)).encode()
        ).hexdigest()[:10]
        path = Path(".vectordb") / uid
        path.mkdir(parents=True, exist_ok=True)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings()
        except ImportError:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. Please ensure sentence-transformers "
                "is installed with: pip install sentence-transformers"
            )

        if any(path.iterdir()):  
            return Chroma(
                persist_directory=str(path), embedding_function=embeddings
            )

        docs = [
            Document(page_content=p["text"], metadata={"title": p["title"]})
            for p in papers
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk, chunk_overlap=overlap
        )
        chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(
            chunks, embeddings, persist_directory=str(path)
        )
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

        generator = ScriptGenerator(chain=chain, use_local_logging=self.LOCAL_LOGGING_ACTIVE)
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
            generation_prompt = (row.get("generation_prompt") or DEFAULT_SCRIPT_PROMPT).strip()

            logging.info(
                "(row %d) extract=%s | analyse=%s | generate=%s — %s",
                idx,
                do_extract,
                do_analyse,
                do_generate,
                query,
            )

            papers = (
                self._create_arxiv_searcher(query, k, do_extract)
                .search_and_extract()
            )

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
                summary, chain = self._summarise(papers, analysis_prompt, chunk, overlap)

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
