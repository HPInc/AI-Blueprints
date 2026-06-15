"""
Standalone Model class for Agentic RAG with TensorRT-LLM.

Business Logic Layer
- Handles complex agentic RAG workflow using LangGraph state machine
- Manages TensorRT-LLM integration, embeddings, vector database, and memory systems
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import json
import logging
import os
import sys
import shutil
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import tensorrt_llm
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.trt_llm_langchain import TensorRTLangchain

# Import MLflow for PythonModel base class
try:
    import mlflow.pyfunc
except ImportError:
    # Fallback if MLflow is not available during development
    class PythonModel:
        pass

    mlflow = type(
        "mlflow", (), {"pyfunc": type("pyfunc", (), {"PythonModel": PythonModel})()}
    )()

# Suppress verbose warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set up logger
logger = logging.getLogger(__name__)


class Model(mlflow.pyfunc.PythonModel):
    """
    Standalone model class for Agentic RAG with TensorRT-LLM.
    Handles complex agentic RAG workflow using LangGraph state machine.
    """

    TOPIC: str = "AI Studio"
    CONTEXT_DIR: Path = Path("../data/context")
    CHROMA_DIR: Path = Path("../data/chroma_store")
    MEMORY_PATH: Path = Path("../data/memory/memory.json")
    MANIFEST_PATH: Path = CHROMA_DIR / "manifest.json"

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

    class RAGState(TypedDict, total=False):
        topic: str
        query: str
        is_relevant: Optional[bool]
        rewritten_query: Optional[str]
        retrieved_chunks: List[str]
        answer: Optional[str]
        from_memory: Optional[bool]
        messages: List[Dict[str, Any]]  # full conversation with LLM

    def __init__(
        self, config: dict, docs_path: str, secrets: dict = None, model_path: str = None
    ):
        """
        Initialize the Model with vanilla-rag compatible interface.
        Internally maps to TensorRT-LLM and RAG-specific requirements.

        Args:
            config: Model configuration dictionary containing model paths and settings
            docs_path: Path to documents directory for vector database and memory storage
            secrets: Secrets dictionary for environment variables (optional)
            model_path: Single model path (fallback for model resolution)
        """

        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.default_llm_model = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
        self.model_config = config
        self.docs_path = docs_path
        self.secrets = secrets
        self.model_path = model_path
        self.TOPIC = Model.TOPIC

        # Resolve model paths from artifacts or configuration
        self.resolved_model_path = self._resolve_model_path()
        self.model_dir = os.path.dirname(docs_path) if docs_path else ""

        # Model components
        self._embed_model = None
        self._vectorstore = None
        self._llm = None
        self._memory = None
        self._compiled_graph = None

        # Configuration

        # Setup environment and load components
        try:
            self._setup_environment()
            self._load_models()
            # self._setup_memory()
            self._build_state_graph()
            logger.info("Agentic RAG Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agentic RAG Model: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def _resolve_model_path(self) -> str:
        """
        Resolve model path from either artifacts or configuration.

        Returns:
            Resolved model path for TensorRT-LLM
        """
        # First, check if we're in artifact context
        if self.docs_path:
            artifact_dir = os.path.dirname(self.docs_path)  # This is the data_path

            # Try models subdirectory first (where vanilla-rag puts model_path contents)
            models_subdir = os.path.join(artifact_dir, "models")
            if os.path.exists(models_subdir):
                # Look for TensorRT-LLM model files or directories
                for item in os.listdir(models_subdir):
                    item_path = os.path.join(models_subdir, item)
                    if os.path.isdir(item_path) or item.endswith((".engine", ".plan")):
                        logger.info(
                            f"Using TensorRT-LLM model from artifacts: {item_path}"
                        )
                        return item_path

        # Check if model_path was provided and exists
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Using provided model path: {self.model_path}")
            return self.model_path

        # Get model path from config
        config_model_path = self.model_config.get("model_path")
        if config_model_path:
            # Check if it looks like a HuggingFace repo ID (contains '/' but not absolute path)
            if "/" in config_model_path and not os.path.isabs(config_model_path):
                logger.info(
                    f"Using HuggingFace model repo from config: {config_model_path}"
                )
                return config_model_path
            elif os.path.exists(config_model_path):
                logger.info(f"Using local model path from config: {config_model_path}")
                return config_model_path

        # Last fallback - default HF repo
        logger.info(f"Using default HuggingFace model: {self.default_llm_model}")
        return self.default_llm_model

    def _setup_environment(self) -> None:
        """Configure environment variables and suppress verbose logs."""
        try:
            # Load secrets into environment if provided
            if self.secrets:
                for key, value in self.secrets.items():
                    os.environ[key] = str(value)
                logger.info("Secrets loaded into environment")

            # Configure proxy if specified in config
            if "proxy" in self.model_config and self.model_config["proxy"]:
                logger.info(f"Setting up proxy: {self.model_config['proxy']}")
                os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
                os.environ["HTTP_PROXY"] = self.model_config["proxy"]
            else:
                logger.info("No proxy configuration found")

        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function

    def _load_models(self) -> None:
        """Load all required models for the RAG pipeline."""
        try:
            self.TOPIC = Model.TOPIC
            self._logger = logging.getLogger("Model")

            # 1. Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embed_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )

            # 2. Set up Chroma vectorstore directory based on docs_path
            # Use chroma_store to match existing artifacts and class constants
            if self.docs_path:
                chroma_dir = os.path.join(self.docs_path, "chroma_store")
            else:
                chroma_dir = "../data/chroma_store"
            chroma_dir_path = Path(chroma_dir)

            # Initialize or create the vector database
            self._vectorstore = self._initialize_vectorstore(chroma_dir_path)

            # 3. Load LLM via TensorRTLangchain
            sampling_params = tensorrt_llm.SamplingParams(
                temperature=0.0,
                top_k=1,
                repetition_penalty=1.2,
                stop_token_ids=[128009],
            )

            logger.info(
                f"Initializing TensorRT LLM with model path: {self.resolved_model_path}"
            )
            self._llm = TensorRTLangchain(
                model_path=self.resolved_model_path,
                sampling_params=sampling_params,
            )

            # 3. Initialize memory - use memory subdirectory to match class constants
            if self.docs_path:
                memory_path = os.path.join(self.docs_path, "memory", "memory.json")
            else:
                memory_path = "../data/memory/memory.json"
            memory_path_obj = Path(memory_path)
            memory_path_obj.parent.mkdir(parents=True, exist_ok=True)
            if not memory_path_obj.exists():
                memory_path_obj.write_text("{}", encoding="utf-8")
            self._memory = Model.SimpleKVMemory(memory_path_obj)
            self._LLMResponse = namedtuple("Response", ["content"])

            self._build_state_graph()

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _initialize_vectorstore(self, chroma_dir_path: Path) -> Chroma:
        """Initialize or create the Chroma vector database."""
        from langchain_core.documents import Document
        from langchain_community.document_loaders import (
            UnstructuredMarkdownLoader,
            TextLoader,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import json
        import shutil

        collection = "-".join(self.TOPIC.split())

        # Check if context directory exists
        if self.docs_path:
            context_dir = Path(os.path.join(self.docs_path, "context"))
        else:
            context_dir = Path("../data/context")

        manifest_path = chroma_dir_path / "manifest.json"

        def _load_markdown(path: Path) -> List[Document]:
            """Load markdown files with fallback to text loader."""
            try:
                return UnstructuredMarkdownLoader(str(path)).load()
            except Exception:
                return TextLoader(str(path), encoding="utf-8").load()

        def _current_manifest() -> List[str]:
            """Compute a sorted list of all Markdown file paths."""
            if not context_dir.exists():
                logger.warning(f"Context directory does not exist: {context_dir}")
                return []
            return sorted(str(p.resolve()) for p in context_dir.rglob("*.md"))

        def _needs_rebuild() -> bool:
            """Check whether we need to rebuild the vector database."""
            if not chroma_dir_path.exists() or not manifest_path.exists():
                return True
            try:
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return True
            return old != _current_manifest()

        def _save_manifest(manifest: List[str]) -> None:
            """Save the current manifest so future runs can compare."""
            chroma_dir_path.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # If the manifest has changed, wipe & rebuild
        if _needs_rebuild():
            if chroma_dir_path.exists():
                shutil.rmtree(chroma_dir_path)
            logger.info("Building new Chroma index from Markdown files…")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=128, add_start_index=True
            )
            docs: List[Document] = []

            if context_dir.exists():
                for md_file in context_dir.rglob("*.md"):
                    try:
                        for page in _load_markdown(md_file):
                            for chunk in splitter.split_documents([page]):
                                chunk.metadata.setdefault("source", md_file.name)
                                docs.append(chunk)
                    except Exception as e:
                        logger.warning(f"Error processing {md_file}: {e}")

            if docs:
                chroma = Chroma.from_documents(
                    docs,
                    embedding=self._embed_model,
                    collection_name=collection,
                    persist_directory=str(chroma_dir_path),
                )
                _save_manifest(_current_manifest())
                logger.info(f"Chroma index rebuilt with {len(docs)} chunks.")
            else:
                logger.warning(
                    "No documents found to index. Creating empty vector store."
                )
                chroma = Chroma(
                    collection_name=collection,
                    persist_directory=str(chroma_dir_path),
                    embedding_function=self._embed_model,
                )
                _save_manifest([])
            return chroma

        # Otherwise, load the existing, up-to-date index
        logger.info(f"Loading existing Chroma index from {chroma_dir_path}")
        return Chroma(
            collection_name=collection,
            persist_directory=str(chroma_dir_path),
            embedding_function=self._embed_model,
        )

    # ----------------------------------------
    # Node Functions (each mirrors the notebook)
    # ----------------------------------------
    def ingest_query(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Log the incoming user query and record it in the message history.
        """
        user_query = state["query"]
        logger.info("Received user query: %s", user_query)
        previous_messages = state.get("messages", [])
        new_messages = previous_messages + [{"role": "user", "content": user_query}]
        return {"messages": new_messages}

    def check_relevance(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Ask the LLM whether the query relates to our topic.
        If not relevant, include a default apology answer.
        """
        topic = state["topic"]
        user_query = state["query"]

        system_prompt = (
            'You are a strict classifier. Only respond with either "yes" or "no". '
            "Do not include any additional words, explanations, or punctuation. "
            "Answer based solely on whether the user's query is about the specified topic."
        )
        user_prompt = (
            f'The topic is: "{topic}"\n\n'
            f'User query: "{user_query}"\n\n'
            "Is this query related to the topic above? Respond with only 'yes' or 'no'."
            "Answer: "
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt)
        is_relevant = "yes" in resp.strip().lower()
        logger.info("Relevance check result: %s", is_relevant)

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Relevance check result:"},
            {"role": "assistant", "content": resp},
        ]
        result: Dict[str, Any] = {"is_relevant": is_relevant, "messages": messages}
        if not is_relevant:
            result["answer"] = f"Sorry, I can only answer questions related to {topic}."
        return result

    def check_memory(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Look up the exact user query in memory and return the cached answer if found.
        """
        raw_query = state["query"]
        key = raw_query.strip().lower()
        cached_answer = self._memory.get(key)
        if cached_answer is not None:
            logger.info("Cache hit for query: %s", raw_query)
            return {"answer": cached_answer, "from_memory": True}
        logger.info("Cache miss for query: %s", raw_query)
        return {"from_memory": False}

    def rewrite_query(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Correct any grammar in the question and rewrite it as a clear statement
        without altering its meaning, to improve retrieval.
        """
        original = state["query"]
        system_prompt = (
            "You are a rewriting assistant. Your only task is to convert a question into a "
            "grammatically correct statement. Do not change its meaning. "
            "Output only the corrected statement—no explanations or extra text."
        )
        user_prompt = (
            "Convert the following question into a grammatically correct statement "
            "that preserves the original meaning exactly:\n\n"
            "Note: Output only the corrected statement—no explanations or extra text.\n"
            f'Question: "{original}"\n\n'
            "Corrected Statement:"
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt).strip()
        logger.info("Rewritten query: %s", resp)

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Rewritten query:"},
            {"role": "assistant", "content": resp},
        ]
        return {"rewritten_query": resp, "messages": messages}

    def retrieve_chunks(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Fetch the top-k most relevant chunks for the rewritten query.
        """
        statement = state["rewritten_query"]
        docs = self._vectorstore.similarity_search(statement, k=5)
        chunks = [doc.page_content for doc in docs]
        logger.info("Retrieved %d chunks for query.", len(chunks))
        return {"retrieved_chunks": chunks}

    def generate_answer(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Use the LLM to generate an answer based solely on retrieved context.
        """
        topic = state["topic"]
        user_query = state["query"]
        context = "\n\n---\n\n".join(state["retrieved_chunks"])

        system_prompt = (
            f"You are a knowledgeable assistant specialized in {topic}. Your task is to answer "
            "the user query using only the information found within the <context> block. "
            'Ignore any external knowledge. If the context does not contain the answer, reply exactly with: "I don\'t know." '
            "Do not assume, infer, or add any extra information. "
            "Respond with only the answer—do not include any introductory or explanatory text."
        )
        user_prompt = (
            f"<context>\n{context}\n</context>\n\n"
            f'User query: "{user_query}"\n\n'
            "Based only on the context above, provide the exact answer to the query. "
            'If the context does not contain the answer, respond exactly with: "I don\'t know." '
            "Give only the answer—do not include any intro phrases such as 'The answer is' or 'Here it is'."
            "Answer: "
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt).strip()
        logger.info("Generated answer (%d chars)", len(resp))

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Generated answer:"},
            {"role": "assistant", "content": resp},
        ]
        return {"answer": resp, "messages": messages}

    def update_memory(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Store new query-answer pairs in memory for faster future lookup.
        """
        if state.get("from_memory"):
            return {}
        raw_query = state["query"]
        key = raw_query.strip().lower()
        answer = state["answer"]
        if answer is not None:
            self._memory.set(key, answer)
            logger.info("Stored query-answer in memory for key: %s", key)
        return {}

    def output_answer(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        The final node. We do not print to STDOUT when serving via MLflow.
        Just return an empty dict as this node does not add new state.
        """
        return {}

    # ----------------------------------------
    # Helper Methods
    # ----------------------------------------
    def _get_response_from_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Wrap the LLM call into the meta-prompt format and return the .content string.
        """
        meta_llama_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        raw = self._llm.invoke(meta_llama_prompt)
        # TensorRTLangchain returns a raw string; we can wrap into Response if needed
        return raw

    def _route_relevance(self, state: "Model.RAGState") -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _route_memory(self, state: "Model.RAGState") -> str:
        return "cached" if state.get("from_memory") else "not_cached"

    def _build_state_graph(self) -> None:
        """
        Construct and compile the LangGraph state graph exactly as in the notebook.
        """
        rag_graph = StateGraph(Model.RAGState)

        # Add nodes
        rag_graph.add_node("ingest_query", self.ingest_query)
        rag_graph.add_node("check_relevance", self.check_relevance)
        rag_graph.add_node("rewrite_query", self.rewrite_query)
        rag_graph.add_node("check_memory", self.check_memory)
        rag_graph.add_node("retrieve_chunks", self.retrieve_chunks)
        rag_graph.add_node("generate_answer", self.generate_answer)
        rag_graph.add_node("update_memory", self.update_memory)
        rag_graph.add_node("output_answer", self.output_answer)

        # Add edges
        rag_graph.add_edge(START, "ingest_query")
        rag_graph.add_edge("ingest_query", "check_relevance")

        rag_graph.add_conditional_edges(
            "check_relevance",
            self._route_relevance,
            {
                "irrelevant": "output_answer",
                "relevant": "check_memory",
            },
        )

        rag_graph.add_conditional_edges(
            "check_memory",
            self._route_memory,
            {
                "cached": "output_answer",
                "not_cached": "rewrite_query",
            },
        )

        rag_graph.add_edge("rewrite_query", "retrieve_chunks")
        rag_graph.add_edge("retrieve_chunks", "generate_answer")
        rag_graph.add_edge("generate_answer", "update_memory")
        rag_graph.add_edge("update_memory", "output_answer")
        rag_graph.add_edge("output_answer", END)

        # Compile graph
        self._compiled_graph = rag_graph.compile()

    def predict(self, model_input, params=None):
        """
        Process inputs and generate responses.
        Performs end-to-end agentic RAG pipeline.

        Args:
            model_input: Input data containing query (pandas.DataFrame or dict)
            params: Optional parameters for model prediction

        Returns:
            Dict with answer, retrieved_chunks, and conversation messages
        """
        try:
            if params is None:
                params = {}
            # Handle pandas DataFrame input
            if isinstance(model_input, pd.DataFrame):
                if "query" not in model_input.columns:
                    raise Exception("DataFrame must contain a 'query' column.")
                # Take the first record in that column
                raw_query = model_input["query"].iloc[0]
            else:
                # Could be a plain dict or something else
                if not isinstance(model_input, dict):
                    raise Exception(
                        f"Unexpected input type: {type(model_input)}. "
                        "Expected pandas.DataFrame or dict with 'query'."
                    )
                # If it's a dict, accept either string or single-element list
                if "query" not in model_input:
                    raise Exception("Input dict must contain key 'query'.")

                query_value = model_input["query"]
                # Handle list format from Streamlit (e.g., [query_string])
                if isinstance(query_value, list):
                    if len(query_value) == 0:
                        raise Exception("Query list cannot be empty.")
                    raw_query = query_value[0]
                else:
                    # Handle direct string
                    raw_query = query_value

            # Initialize state with topic, query, and empty messages
            initial_state: Model.RAGState = {
                "topic": self.TOPIC,
                "query": raw_query.strip(),
                "messages": [],
            }

            # Invoke the compiled LangGraph
            final_state = self._compiled_graph.invoke(input=initial_state)

            # Extract elements to return
            answer = final_state.get("answer", "")
            retrieved_chunks = final_state.get("retrieved_chunks", [])
            messages = final_state.get("messages", [])

            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "messages": messages,
            }

        except Exception as e:
            import traceback

            logger.error(f"Error in predict: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Return a meaningful error response instead of None
            return {
                "answer": f"Error processing query: {str(e)}",
                "retrieved_chunks": [],
                "messages": [{"role": "error", "content": str(e)}],
            }

    def get_onnx_export_config(self) -> List:
        """
        Get configuration for ONNX export.
        Returns the configuration needed for ONNX model export.

        Returns:
            List of ModelExportConfig objects for ONNX conversion
        """
        try:
            # Import here to avoid circular imports
            from src.onnx_utils import ModelExportConfig
            from transformers import AutoTokenizer, AutoModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            # Create sample inputs for ONNX export - Embedding model
            embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name, torch_dtype=dtype
            ).to(device)
            embedding_model.eval()

            embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name
            )
            embedding_inputs = embedding_tokenizer(
                "What is AI Studio?",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            embedding_input_sample = (
                embedding_inputs["input_ids"].to(device),
                embedding_inputs["attention_mask"].to(device),
            )

            model_configs = [
                ModelExportConfig(
                    model=embedding_model,  # Pre-loaded embedding model!
                    model_name="embedding_model",  # ONNX file naming
                    input_sample=embedding_input_sample,
                    task="feature-extraction",
                    opset=17,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                    },
                )
            ]
            logger.info(
                "Added embedding model to ONNX export configuration with model_name: embedding_model"
            )

            # Try to add TensorRT-LLM model for ONNX export
            logger.info("Attempting to add TensorRT-LLM model for ONNX export...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                class TorchWrapper(torch.nn.Module):
                    """Wrapper to make TensorRT model ONNX-compatible."""

                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, input_ids, attention_mask):
                        # Remove use_cache for ONNX compatibility
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        return outputs.logits if hasattr(outputs, "logits") else outputs

                # Create sample inputs for LLM model
                tokenizer = AutoTokenizer.from_pretrained(self.default_llm_model)
                tokenizer.pad_token = tokenizer.eos_token
                llm_inputs = tokenizer(
                    "Hello",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=8,
                )

                llm_input_sample = (
                    llm_inputs["input_ids"].to(device),
                    llm_inputs["attention_mask"].to(device),
                )

                # Try to load a PyTorch version for ONNX export
                logger.info(
                    f"Attempting to load PyTorch model from: {self.resolved_model_path}"
                )
                try:
                    # Use the resolved model path (could be HF cache or local path)
                    model_path_for_loading = self.resolved_model_path

                    # If it's a local directory path from HF cache, use it directly
                    # Otherwise fall back to the HF model ID
                    if os.path.exists(model_path_for_loading) and os.path.isdir(
                        model_path_for_loading
                    ):
                        logger.info(
                            f"Loading PyTorch model from local cache: {model_path_for_loading}"
                        )
                        pytorch_model = AutoModelForCausalLM.from_pretrained(
                            model_path_for_loading,
                            torch_dtype=dtype,
                            local_files_only=True,
                        ).to(device)
                    else:
                        logger.info(
                            f"Loading PyTorch model from HF hub: {self.default_llm_model}"
                        )
                        pytorch_model = AutoModelForCausalLM.from_pretrained(
                            self.default_llm_model, torch_dtype=dtype
                        ).to(device)

                    pytorch_model.eval()
                    wrapped_model = TorchWrapper(pytorch_model)

                    model_configs.append(
                        ModelExportConfig(
                            model=wrapped_model,
                            model_name="nemotron_model",  # Changed to match expected name
                            input_sample=llm_input_sample,
                            task="text-generation",
                            input_names=["input_ids", "attention_mask"],
                            output_names=["logits"],
                            opset=17,
                            dynamic_axes={
                                "input_ids": {0: "batch_size", 1: "sequence_length"},
                                "attention_mask": {
                                    0: "batch_size",
                                    1: "sequence_length",
                                },
                                "logits": {0: "batch_size", 1: "sequence_length"},
                            },
                        )
                    )
                    logger.info(
                        "✅ Successfully added TensorRT-LLM model to ONNX export configuration with model_name: nemotron_model"
                    )
                except Exception as model_load_error:
                    logger.error(
                        f"❌ Could not load PyTorch version of LLM for ONNX export: {model_load_error}"
                    )
                    import traceback

                    logger.error(f"Full traceback: {traceback.format_exc()}")

            except Exception as llm_export_error:
                logger.error(
                    f"❌ Could not add TensorRT-LLM model to ONNX export: {llm_export_error}"
                )
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")

            logger.info("ONNX export configuration created successfully")
            logger.info(
                f"Total models configured for ONNX export: {len(model_configs)}"
            )
            for i, config in enumerate(model_configs):
                logger.info(f"Model {i+1}: {config.model_name}")
            return model_configs

        except Exception as e:
            logger.error(f"Error creating ONNX export configuration: {str(e)}")
            raise RuntimeError(
                f"Failed to create ONNX export configuration: {str(e)}"
            ) from e

    def copy_model_artifacts_to_directory(self, target_dir: str) -> None:
        """
        Copy model artifacts to a target directory.

        Args:
            target_dir: Directory path where to copy the model files
        """
        try:
            os.makedirs(target_dir, exist_ok=True)

            # Copy TensorRT-LLM model artifacts if they exist locally
            if os.path.exists(self.resolved_model_path) and os.path.isdir(
                self.resolved_model_path
            ):
                target_model_dir = os.path.join(target_dir, "tensorrt_llm")
                shutil.copytree(
                    self.resolved_model_path, target_model_dir, dirs_exist_ok=True
                )
                logger.info(f"Copied TensorRT-LLM model to {target_model_dir}")
            elif os.path.exists(self.resolved_model_path) and os.path.isfile(
                self.resolved_model_path
            ):
                target_model_file = os.path.join(
                    target_dir, os.path.basename(self.resolved_model_path)
                )
                shutil.copyfile(self.resolved_model_path, target_model_file)
                logger.info(f"Copied TensorRT-LLM model file to {target_model_file}")
            else:
                logger.info(
                    f"Model path {self.resolved_model_path} is not a local file/directory - skipping copy"
                )

            # Copy vector database if it exists
            if self.docs_path:
                chroma_dir = os.path.join(self.docs_path, "chroma_db")
                if os.path.exists(chroma_dir):
                    target_chroma_dir = os.path.join(target_dir, "chroma_db")
                    shutil.copytree(chroma_dir, target_chroma_dir, dirs_exist_ok=True)
                    logger.info(f"Copied Chroma database to {target_chroma_dir}")

                # Copy memory file if it exists
                memory_file = os.path.join(self.docs_path, "memory.json")
                if os.path.exists(memory_file):
                    target_memory_file = os.path.join(target_dir, "memory.json")
                    shutil.copyfile(memory_file, target_memory_file)
                    logger.info(f"Copied memory file to {target_memory_file}")

            logger.info(f"Model artifacts copied to directory: {target_dir}")

        except Exception as e:
            logger.error(f"Error copying model artifacts: {str(e)}")
            raise RuntimeError(f"Failed to copy model artifacts: {str(e)}") from e
