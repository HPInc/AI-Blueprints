

# === Built-in ======================================================================
import os, sys, logging, traceback, hashlib
from typing import Dict, Any, List, Optional

# === Third-party ===================================================================
import pandas as pd
import chromadb
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Path bootstrap ================================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# === Internal ======================================================================
from src.service.base_service import BaseGenerativeService
from src.utils import get_context_window, clean_code, initialize_llm
from core.extract_text.rag_utils import process_repository_question
from core.prompt_templates import (
    get_code_description_prompt,
    get_code_generation_prompt,
    get_specialized_prompt,
)
from core.dataflow.dataflow import DataFrameConverter, EmbeddingUpdater
from core.vector_database.vector_store_writer import VectorStoreWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CodeGenerationService(BaseGenerativeService):
    # ---------------------------------------------------------------- INIT
    def __init__(self, delay_async_init: bool = False):
        super().__init__()
        self.delay_async_init = delay_async_init
        self.embedding_fn: Optional[HuggingFaceEmbeddings] = None
        self.chroma_embedding_fn = None
        self.collection_name = "my_collection"
        self.collection = self.vector_store = None
        self.code_description_prompt = get_code_description_prompt()
        self.code_generation_prompt = get_code_generation_prompt()
        self.context_window: Optional[int] = None

    # ---------------------------------------------------------------- Helpers
    @staticmethod
    def _wrap(txt: str) -> pd.DataFrame:
        return pd.DataFrame([{"result": clean_code(txt)}])

    # ---------------------------------------------------------------- Embeddings
    def _init_embeddings(self, local_path: Optional[str] = None):
        model_name = local_path or "all-MiniLM-L6-v2"
        self.embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
        from core.chroma_embedding_adapter import ChromaEmbeddingAdapter
        self.chroma_embedding_fn = ChromaEmbeddingAdapter(self.embedding_fn)

    # ---------------------------------------------------------------- Vector store
    def _ensure_vector_store(self, persist="./chroma_db"):
        os.makedirs(persist, exist_ok=True)
        client = chromadb.PersistentClient(path=persist)
        self.collection = client.get_or_create_collection(self.collection_name)
        self.vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.chroma_embedding_fn,
        )

    # ---------------------------------------------------------------- Model
    def load_model(self, context):
        cfg = self.model_config
        source = cfg.get("model_source", "local")
        if source == "local":
            self.llm = initialize_llm("local", {}, context.artifacts.get("models"))
        else:
            self.llm = initialize_llm(source, cfg.get("secrets", {}), None)
        self.context_window = get_context_window(self.llm)

    # ---------------------------------------------------------------- Chains
    def _make_chains(self):
        def _extract(inp):
            q = inp.get("question", "")
            res = process_repository_question(
                query=q,
                collection=self.collection,
                context_window=self.context_window,
                top_n=8,
            )
            if res["document_count"] == 0:
                return {"prompt": self.code_description_prompt,
                        "question": q, "context": "No relevant documents"}
            return {
                "prompt": get_specialized_prompt(res["question_types"]) or self.code_description_prompt,
                "question": q,
                "context": res["context"],
            }

        self.repo_chain = (
            _extract
            | (lambda d: d["prompt"].format(question=d["question"], context=d["context"]))
            | self.llm
            | StrOutputParser()
        )

        self.direct_chain = (
            {"question": RunnablePassthrough(), "context": lambda _: ""}
            | self.code_generation_prompt
            | self.llm
            | StrOutputParser()
        )

    # ---------------------------------------------------------------- Predict
    def predict(self, context, model_input: Dict[str, Any]) -> pd.DataFrame:
        try:
            req = model_input.get("inputs", model_input)
            q = req.get("question", "")
            repo = req.get("repository_url")
            fast = bool(req.get("metadata_only", False))

            if not q:
                return self._wrap("# Error: ‘question’ é obrigatório")

            if not repo:
                return self._wrap(self.direct_chain.invoke({"question": q}))

            if fast:
                return self._wrap("# Repository fast-path: volte depois.")

            return self._wrap(self.repo_chain.invoke({"question": q}))
        except Exception as e:
            logger.error("Erro inferência: %s", e)
            logger.debug(traceback.format_exc())
            return self._wrap(f"# Error: {e}")

    # ---------------------------------------------------------------- load_context
    def load_context(self, context):
        self._init_embeddings(context.artifacts.get("embedding_model"))
        self._ensure_vector_store()
        super().load_context(context)
        self._make_chains()

    # ---------------------------------------------------------------- log_model
    @classmethod
    def log_model(
        cls,
        secrets_path: str,
        config_path: str,
        model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        delay_async_init: bool = True,
        experiment_name: str = "Code-Generation-Experiment",
        run_name: str = "Code-Generation-Run",
    ):
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        mlflow.set_experiment(experiment_name)

        signature = ModelSignature(
            inputs=Schema([
                ColSpec("string", "question"),
                ColSpec("string", "repository_url", required=False),
                ColSpec("boolean", "metadata_only", required=False),
            ]),
            outputs=Schema([ColSpec("string", "result")]),
        )

        # --- artefatos ------------------------------------------------------
        artifacts = {"secrets": secrets_path, "config": config_path}
        if model_path:
            artifacts["models"] = model_path
        if embedding_model_path and os.path.exists(embedding_model_path):
            artifacts["embedding_model"] = embedding_model_path

        # --- code_paths  ------------------------------------------
        this_dir = os.path.abspath(os.path.dirname(__file__))      # notebooks/core
        src_dir  = os.path.abspath(os.path.join(this_dir, "..", "..", "src"))
        paths = [p for p in (this_dir, src_dir) if os.path.isdir(p)]
        if not paths:                                              # fallback 
            paths = None

        # --- log -----------------------------------------------------------
        with mlflow.start_run(run_name=run_name):
            mlflow.pyfunc.log_model(
                artifact_path="code_generation_service",
                python_model=cls(delay_async_init=delay_async_init),
                artifacts=artifacts,
                signature=signature,
                code_paths=paths,
                pip_requirements=[
                    "langchain",
                    "chromadb",
                    "sentence-transformers",
                    "langchain_huggingface",
                    "langchain_community",
                ],
            )
            logger.info("✅Registered model (run_id=%s)",
                        mlflow.active_run().info.run_id)
