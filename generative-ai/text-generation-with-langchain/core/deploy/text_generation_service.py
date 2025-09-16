# core/deploy/text_generation_service.py
# -*- coding: utf-8 -*-
"""
End-to-end pipeline exposed as an MLflow **pyfunc**:

    arXiv → paper extraction → summarisation → slide-style script.

MLflow-compatible service for text generation from arXiv papers.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import tempfile
import multiprocessing


import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema



ROOT_DIR = Path(__file__).resolve().parent.parent
LOGLEVEL_FILE = Path(__file__).with_suffix(".loglevel")
DEFAULT_LOG_LEVEL = LOGLEVEL_FILE.read_text().strip() if LOGLEVEL_FILE.exists() else "INFO"

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

LOCAL_LOGGING_ACTIVE: bool = False

logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def _add_project_to_syspath() -> Tuple[Path, Path | None]:
    """
    Ensure <repo>/core and an optional <repo>/src are on ``sys.path`` so that
    imports work when the model is loaded inside the MLflow scoring server.
    """
    core_path = ROOT_DIR
    (core_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(core_path))

    src_path = next(
        (p / "src" for p in [core_path, *core_path.parents] if (p / "src").is_dir()),
        None,
    )
    if src_path:
        sys.path.insert(0, str(src_path))

    sys.path.insert(0, str(core_path.parent))
    return core_path, src_path


def _load_llm(artifacts: Dict[str, str]):
    """
    Load the LlamaCpp model.
    """
    from src.utils import (
        configure_hf_cache,
        configure_proxy,
        load_config
    )
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    if hasattr(LlamaCpp, "model_rebuild"):
        LlamaCpp.model_rebuild()

    cfg_dir = Path(artifacts["config"]).parent
    cfg = load_config(
        cfg_dir / "config.yaml"
    )

    # External logging integration disabled
    global LOCAL_LOGGING_ACTIVE
    LOCAL_LOGGING_ACTIVE = False

    model_path = artifacts.get("llm") or ""
    if not model_path:
        raise RuntimeError("Missing *.gguf artefact for the LLM.")

    configure_hf_cache()
    configure_proxy(cfg)

    start = time.perf_counter()
    llm = LlamaCpp(
        model_path=model_path,
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
    logging.info("🔹 LlamaCpp loaded in %.1fs", time.perf_counter() - start)
    return llm


class TextGenerationService(mlflow.pyfunc.PythonModel):
    """arXiv → summary → slide-script."""

    def load_context(self, context):
        _add_project_to_syspath()
        self.llm = _load_llm(context.artifacts)

    @staticmethod
    def _create_arxiv_searcher(query: str, max_results: int, download: bool):
        from core.extract_text.arxiv_search import ArxivSearcher

        kwargs: Dict[str, Any] = {"query": query, "max_results": max_results}
        sig = inspect.signature(ArxivSearcher)  
        if "cache_only" in sig.parameters:
            kwargs["cache_only"] = not download
        elif "download" in sig.parameters:
            kwargs["download"] = download
        return ArxivSearcher(**kwargs)  

    def _build_vectordb(self, papers: List[dict], chunk: int, overlap: int):
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
        from core.analyzer.scientific_paper_analyzer import ScientificPaperAnalyzer

        vectordb = self._build_vectordb(papers, chunk, overlap)
        analyzer = ScientificPaperAnalyzer(vectordb.as_retriever(), self.llm)
        return analyzer.analyze(prompt), analyzer.get_chain()

    def _generate_script(self, chain, prompt):
        from core.generator.script_generator import ScriptGenerator

        generator = ScriptGenerator(chain=chain, use_local_logging=LOCAL_LOGGING_ACTIVE)
        generator.add_section(name="user_prompt", prompt=prompt)

        stdin_backup, builtins.input = builtins.input, lambda *_a, **_kw: "y"
        try:
            generator.run()
        finally:
            builtins.input = stdin_backup

        return generator.get_final_script()

    def predict(self, _: Any, df: pd.DataFrame) -> pd.DataFrame:
        results: List[dict] = []

        for idx, row in df.iterrows():
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

    @classmethod
    def log_model(
        cls,
        *,
        artifact_path: str = "script_generation_model",
        llm_artifact: str = "models/",
        config_path: str = "configs/config.yaml",
        secrets_dict: Dict = None,
        demo_folder: str = None,

    ):
        """
        Log the model to MLflow.
        
        Args:
            artifact_path: Path to store the model artifacts
            llm_artifact: Path to the llm model
            config_path: Path to the configuration file
            secrets_dict: Dict with secrets to persist as YAML (optional)
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
             
        core, src = _add_project_to_syspath()
        
        artifacts = {
            "config": str(Path(config_path).resolve()),
            "llm": llm_artifact,
        }

        if secrets_dict:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            yaml.safe_dump(secrets_dict, tmp)
            tmp.close()
            artifacts["secrets"] = tmp.name
            logging.info(f"Secrets artifact written to temporary file {tmp.name}")
        
        # Add demo folder to artifacts if provided
        if demo_folder:
            artifacts["demo"] = str(Path(demo_folder).resolve())
        
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=ModelSignature(
                inputs=Schema(
                    [
                        ColSpec("string", "query"),
                        ColSpec("integer", "max_results"),
                        ColSpec("integer", "chunk_size"),
                        ColSpec("integer", "chunk_overlap"),
                        ColSpec("boolean", "do_extract"),
                        ColSpec("boolean", "do_analyze"),
                        ColSpec("boolean", "do_generate"),
                        ColSpec("string", "analysis_prompt"),
                        ColSpec("string", "generation_prompt"),
                    ]
                ),
                outputs=Schema(
                    [
                        ColSpec("string", "extracted_papers"),
                        ColSpec("string", "script"),
                    ]
                ),
            ),
            pip_requirements="../requirements.txt",
            code_paths=[str(core)] + ([str(src)] if src else []),
        )


