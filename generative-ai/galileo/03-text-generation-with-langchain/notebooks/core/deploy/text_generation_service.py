# core/deploy/text_generation_service.py
# -*- coding: utf-8 -*-
"""
End-to-end pipeline exposed as an MLflow **pyfunc**:

    arXiv → paper extraction → summarisation → slide-style script.

Optional integration with **Galileo Prompt-Quality** (promptquality):
activate it by setting the environment variable

    GALILEO_PQ=ON        # (ON | 1 | TRUE are accepted, case-insensitive)

and provide an API-key either in `secrets.yaml` (key: `GALILEO_API_KEY`)
or as `GALILEO_API_KEY` in the environment.  
`config.yaml` may optionally define `galileo_console_url`.
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

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

ROOT_DIR = Path(__file__).resolve().parent.parent
ENABLE_GALILEO_FLAG = os.getenv("GALILEO_PQ", "OFF").upper() in {"ON", "1", "TRUE"}
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

GALILEO_ACTIVE: bool = False

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


def _patch_promptquality(pq_module) -> None:
    """Replace `GalileoPromptCallback` with a no-op stub everywhere."""

    class _Stub: 
        def __init__(self, *_, **__): ...
        def __call__(self, *_, **__): return self
        def finish(self, *_, **__): ...

    pq_module.GalileoPromptCallback = _Stub  

    for submodule in ("promptquality.callback", "promptquality.set_config_module"):
        try:
            mod = importlib.import_module(submodule)
            if hasattr(mod, "GalileoPromptCallback"):
                mod.GalileoPromptCallback = _Stub  
            if hasattr(mod, "set_config"):
                mod.set_config = lambda *_, **__: None  
        except ModuleNotFoundError:
            pass

    try:
        sg = importlib.import_module("core.generator.script_generator")
        sg.pq = pq_module
    except ModuleNotFoundError:
        pass

    if hasattr(pq_module, "disable"):
        pq_module.disable()


def _initialise_promptquality(api_key: str | None, console_url_cfg: str | None) -> bool:
    """
    Try to enable Galileo Prompt-Quality.  
    Returns **True** if fully enabled *and login succeeded*, otherwise patches
    prompt-quality with stubs so that it never raises at runtime.
    """
    try:
        pq = importlib.import_module("promptquality")
    except ModuleNotFoundError:
        logging.info("promptquality not installed – Galileo disabled.")
        return False

    console_url = (
        console_url_cfg
        or os.getenv("GALILEO_CONSOLE_URL")
        or "https://console.hp.galileocloud.io/"
    ).rstrip("/") + "/"

    # If the global flag is OFF or key is missing, disable gracefully
    if not (ENABLE_GALILEO_FLAG and api_key):
        reason = "flag OFF" if not ENABLE_GALILEO_FLAG else "API-key missing"
        logging.info("🔸 Galileo disabled – %s.", reason)
        _patch_promptquality(pq)
        return False

    # Set environment vars expected by prompt-quality
    os.environ["GALILEO_API_KEY"] = api_key
    os.environ["GALILEO_CONSOLE_URL"] = console_url

    try:
        pq.login(console_url)
        logging.info("🔹 Galileo enabled – console: %s", console_url)
        return True
    except Exception as exc:  
        logging.warning("Galileo login failed (%s). Falling back to stub.", exc)
        _patch_promptquality(pq)
        return False


def _load_llm(artifacts: Dict[str, str]):
    """
    Load the LlamaCpp model and configure Prompt-Quality if requested.
    """
    from src.utils import (
        configure_hf_cache,
        configure_proxy,
        load_config_and_secrets,
    )
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    if hasattr(LlamaCpp, "model_rebuild"): 
        LlamaCpp.model_rebuild()

    cfg_dir = Path(artifacts["config"]).parent
    cfg, secrets = load_config_and_secrets(
        cfg_dir / "config.yaml", cfg_dir / "secrets.yaml"
    )

    global GALILEO_ACTIVE
    GALILEO_ACTIVE = _initialise_promptquality(
        api_key=secrets.get("GALILEO_API_KEY") or os.getenv("GALILEO_API_KEY"),
        console_url_cfg=cfg.get("galileo_console_url"),
    )

    model_path = artifacts.get("llm") or ""
    if not model_path:
        raise RuntimeError("Missing *.gguf artefact for the LLM.")

    configure_hf_cache()
    configure_proxy(cfg)

    start = time.perf_counter()
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=int(cfg.get("n_gpu_layers", 1)),  # 0 → CPU-only
        n_batch=256,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,
        temperature=0.2,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=False,
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
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        uid = hashlib.md5(
            ("|".join(sorted(p["title"] for p in papers)) + str(chunk)).encode()
        ).hexdigest()[:10]
        path = Path(".vectordb") / uid
        path.mkdir(parents=True, exist_ok=True)

        if any(path.iterdir()):  
            return Chroma(
                persist_directory=str(path), embedding_function=HuggingFaceEmbeddings()
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
            chunks, HuggingFaceEmbeddings(), persist_directory=str(path)
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

        generator = ScriptGenerator(chain=chain, use_galileo=GALILEO_ACTIVE)
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
        config_yaml: str = "configs/config.yaml",
        secrets_yaml: str = "configs/secrets.yaml",
    ):
        core, src = _add_project_to_syspath()
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts={
                "config": str(Path(config_yaml).resolve()),
                "secrets": str(Path(secrets_yaml).resolve()),
                "llm": llm_artifact,
            },
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
            pip_requirements=["PyYAML", "requests", "pymupdf"],
            code_paths=[str(core)] + ([str(src)] if src else []),
        )


