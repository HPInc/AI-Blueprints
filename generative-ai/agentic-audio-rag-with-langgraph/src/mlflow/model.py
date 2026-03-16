"""
Standalone Model class for Agentic Audio RAG.

Business Logic Layer
- Manages model initialization, embeddings, vector database, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

# ─────── Standard Library Imports ───────
from __future__ import annotations  # Future-proofing for type annotations
import json  # JSON parsing and serialization
import re  # Regular expressions
import logging  # Logging utilities
import os  # Operating system interaction
from pathlib import Path  # Object-oriented filesystem paths
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    TypedDict,
    Annotated,
    Optional,
)  # Static typing support
import numpy as np  # Numerical computing with arrays
import soundfile as sf  # Audio file reading and writing
from types import SimpleNamespace  # Simple object for attribute access
import operator  # Functional programming utilities
import uuid
import sys
import pandas as pd
import yaml

# ─────── Third-Party Package Imports ───────
from langchain_core.documents import (
    Document,
)  # Core document abstraction for LangChain
from langgraph.graph import StateGraph, END  # LangGraph for stateful agent workflows
from pydantic import BaseModel  # Data validation and model parsing
import torch  # PyTorch for deep learning
import torchaudio
import faiss
from transformers import (
    ClapProcessor,
    ClapModel,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from qwen_omni_utils import process_mm_info

# ─────── Local Application-Specific Imports ───────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.agentic_workflow import (
    build_audio_agentic_graph,
)  # Custom LangGraph construction logic
from src.qwen_agent import QwenOmniAgent
from src.simple_kv_memory import (
    SimpleKVMemory,
    _mem_get,
    _mem_put,
)  # In-memory key-value store for agent state
from src.utils import setup_model_environment  # Project-wide configured logger
from src.segment_audio_embeddings import (
    AudioIndex,
    ensure_wav,
    segment_audio_embeddings,
)

# Set up logger
logger = logging.getLogger(__name__)

MEMORY_FILENAME = "kv_memory.json"
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

RELEVANCE_THRESHOLD = 0.18
FETCH_K = 24  # breadth for stage-1
TOP_K = 6  # final segments


class Model:
    """
    Standalone agentic audio RAG model class with no MLflow inheritance.
    Handles RAG-based question answering with audio files.
      - CLAP for retrieval + MMR reranking
      - Qwen2.5 Omni Thinker for audio reasoning
      - LangGraph for memory → retrieve → generate → memoize
    """

    def __init__(
        self,
        context,
        config: dict,
        model_path: str = None,
        secrets: dict = None,
    ):
        """
        Initialize the Model with configuration and artifacts.

        Args:
            context: MLflow context with artifacts paths
            config: Model configuration dictionary
            model_path: Path to local model file (optional)
            secrets: Dictionary containing secrets (optional)
        """
        self.model_config = config
        self.model_path = model_path
        self.secrets = secrets

        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.chain = None
        self.prompt = None
        self.prompt_str = ""
        self.memory = []
        self.callback_manager = None

        # File-based index storage
        self.index_base_dir = None
        self.file_indexes = {}  # In-memory cache: {file_id: AudioIndex}

        # Setup environment and load components
        try:
            self._setup_environment()

            # --- Artifacts ---
            config_path = Path(context.artifacts.get("config_path", "config.yaml"))
            memory_dir = Path(context.artifacts.get("memory_dir", "memory"))
            memory_dir.mkdir(parents=True, exist_ok=True)

            # Index storage directory
            self.index_base_dir = Path(context.artifacts.get("index_dir", "indexes"))
            self.index_base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Index storage directory: {self.index_base_dir}")

            # Load or create runtime config
            if config_path.exists():
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {
                    "relevance_threshold": 0.18,
                    "fetch_k": 24,
                    "top_k": 6,
                }
                with open(config_path, "w") as f:
                    yaml.safe_dump(cfg, f, indent=2)

            self.relevance_threshold = float(cfg.get("relevance_threshold", 0.18))
            self.fetch_k = int(cfg.get("fetch_k", 24))
            self.top_k = int(cfg.get("top_k", 6))

            # --- CLAP (CPU to avoid OOM) ---
            clap_local_path = cfg.get("clap_model_path")

            # Check for packaged model in artifacts first (deployment), then datafabric (development)
            clap_artifacts_path = None
            if "models" in context.artifacts:
                models_dir = Path(context.artifacts["models"])
                if models_dir.exists():
                    clap_artifacts_path = models_dir / "clap"

            if clap_artifacts_path and clap_artifacts_path.exists():
                logger.info(
                    f"🔄 Loading CLAP model from artifacts: {clap_artifacts_path}"
                )
                self.clap_processor = ClapProcessor.from_pretrained(
                    str(clap_artifacts_path)
                )
                self.clap_model = ClapModel.from_pretrained(
                    str(clap_artifacts_path)
                ).eval()
            elif clap_local_path and Path(clap_local_path).exists():
                logger.info(f"🔄 Loading CLAP model from datafabric: {clap_local_path}")
                self.clap_processor = ClapProcessor.from_pretrained(clap_local_path)
                self.clap_model = ClapModel.from_pretrained(clap_local_path).eval()
            else:
                # Fallback to default repo with cache-only loading
                clap_repo = "laion/clap-htsat-unfused"
                logger.info(f"🔄 Loading CLAP model from cache: {clap_repo}")
                self.clap_processor = ClapProcessor.from_pretrained(
                    clap_repo, local_files_only=True, force_download=False
                )
                self.clap_model = ClapModel.from_pretrained(
                    clap_repo, local_files_only=True, force_download=False
                ).eval()

            try:
                self.clap_model.to("cpu")
            except Exception:
                pass

            # --- Memory ---
            self.memory = SimpleKVMemory(memory_dir / MEMORY_FILENAME)

            # --- Qwen Omni (audio agent) ---
            qwen_local_path = cfg.get("qwen_model_path")

            # Check for packaged model in artifacts first (deployment), then datafabric (development)
            qwen_artifacts_path = None
            if "models" in context.artifacts:
                models_dir = Path(context.artifacts["models"])
                if models_dir.exists():
                    qwen_artifacts_path = models_dir / "qwen"

            if qwen_artifacts_path and qwen_artifacts_path.exists():
                logger.info(
                    f"🔄 Loading Qwen model from artifacts: {qwen_artifacts_path}"
                )
                self.q_processor = Qwen2_5OmniProcessor.from_pretrained(
                    str(qwen_artifacts_path), trust_remote_code=True
                )
                self.q_model = (
                    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        str(qwen_artifacts_path),
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    ).eval()
                )
            elif qwen_local_path and Path(qwen_local_path).exists():
                logger.info(f"🔄 Loading Qwen model from datafabric: {qwen_local_path}")
                self.q_processor = Qwen2_5OmniProcessor.from_pretrained(
                    qwen_local_path, trust_remote_code=True
                )
                self.q_model = (
                    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        qwen_local_path,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    ).eval()
                )
            else:
                # Fallback to default repo with cache-only loading
                audio_llm_id = "Qwen/Qwen2.5-Omni-7B"
                logger.info(f"🔄 Loading Qwen model from cache: {audio_llm_id}")
                self.q_processor = Qwen2_5OmniProcessor.from_pretrained(
                    audio_llm_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    force_download=False,
                )
                self.q_model = (
                    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        audio_llm_id,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                        force_download=False,
                    ).eval()
                )

            self.audio_llm = QwenOmniAgent(self.q_processor, self.q_model)

            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Model: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def _setup_environment(self) -> None:
        """Configure environment variables based on configuration and secrets."""
        setup_model_environment()
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

            # Configure environment for optimal model loading
            logger.info("🚀 Configuring model loading environment")

            # Only set offline mode if we detect we'll be using HuggingFace cache
            # (datafabric models don't need offline restrictions)
            qwen_local_path = self.model_config.get("qwen_model_path")
            clap_local_path = self.model_config.get("clap_model_path")

            using_datafabric = (
                qwen_local_path
                and Path(qwen_local_path).exists()
                and clap_local_path
                and Path(clap_local_path).exists()
            )

            if not using_datafabric:
                logger.info("Setting offline mode for HuggingFace cache loading")
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            else:
                logger.info("Using datafabric models - no offline restrictions needed")

            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function

    def _sanitize_file_id(self, file_id: str) -> str:
        """Sanitize file_id for safe filesystem usage."""
        # Replace unsafe characters with underscores
        safe_id = re.sub(r'[<>:"/\\|?*]', "_", file_id)
        return safe_id

    def _get_index_dir(self, file_id: str) -> Path:
        """Get the directory path for storing a file's index."""
        safe_id = self._sanitize_file_id(file_id)
        return self.index_base_dir / safe_id

    def _save_index(self, file_id: str, audio_index: AudioIndex) -> None:
        """Save AudioIndex to disk using FAISS binary format + JSON metadata."""
        try:
            index_dir = self._get_index_dir(file_id)
            index_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_file = index_dir / "faiss.index"
            faiss.write_index(audio_index.index, str(index_file))

            # Save metadata (without embedding vectors to save space)
            meta_file = index_dir / "metadata.json"
            # Remove 'vec' from metadata if present (it's in FAISS index)
            clean_meta = [
                {k: v for k, v in m.items() if k != "vec"} for m in audio_index.meta
            ]
            with open(meta_file, "w") as f:
                json.dump(clean_meta, f, indent=2)

            # Save index info
            info_file = index_dir / "info.json"
            with open(info_file, "w") as f:
                json.dump(
                    {
                        "file_id": file_id,
                        "num_segments": len(audio_index.meta),
                        "dim": audio_index.index.d,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Saved index for {file_id}: {len(audio_index.meta)} segments")
        except Exception as e:
            logger.error(f"Error saving index for {file_id}: {e}")
            raise

    def _load_index(self, file_id: str) -> Optional[AudioIndex]:
        """Load AudioIndex from disk if it exists."""
        try:
            index_dir = self._get_index_dir(file_id)
            index_file = index_dir / "faiss.index"
            meta_file = index_dir / "metadata.json"

            if not (index_file.exists() and meta_file.exists()):
                return None

            # Load FAISS index
            faiss_index = faiss.read_index(str(index_file))

            # Load metadata
            with open(meta_file, "r") as f:
                metadata = json.load(f)

            # Reconstruct AudioIndex
            audio_index = AudioIndex(dim=faiss_index.d)
            audio_index.index = faiss_index
            audio_index.meta = metadata

            logger.info(f"Loaded index for {file_id}: {len(metadata)} segments")
            return audio_index

        except Exception as e:
            logger.warning(f"Could not load index for {file_id}: {e}")
            return None

    def process_audio_file(self, audio_path: str, file_id: str = None) -> str:
        """
        Process a single audio file on-demand using the same pattern as run-workflow.ipynb:
        1. Check if already indexed (load from disk cache)
        2. Copy temp file to persistent storage
        3. Call segment_audio_embeddings() exactly like the notebook does
        4. Save index to disk for persistence
        5. Cache in memory for fast access

        Args:
            audio_path: Path to audio/video file
            file_id: Optional identifier for this file (defaults to filename)

        Returns:
            file_id: Identifier to use in subsequent queries
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Use filename as file_id if not provided
            if file_id is None:
                file_id = audio_path.name

            # Create persistent audio storage directory
            persistent_audio_dir = self.index_base_dir / "audio_files"
            persistent_audio_dir.mkdir(parents=True, exist_ok=True)

            # Copy temp file to persistent storage if it's in a temp location
            safe_id = self._sanitize_file_id(file_id)
            persistent_path = persistent_audio_dir / f"{safe_id}{audio_path.suffix}"

            if audio_path.parent.name == "tmp" or "tmp" in str(audio_path):
                # This is a temp file, copy it to persistent storage
                import shutil

                shutil.copy2(audio_path, persistent_path)
                logger.info(
                    f"Copied temp file to persistent storage: {persistent_path}"
                )
                # Use persistent path for processing
                audio_path = persistent_path

            # Check in-memory cache first
            if file_id in self.file_indexes:
                logger.info(f"File {file_id} already in memory cache")
                return file_id

            # Try loading from disk
            cached_index = self._load_index(file_id)
            if cached_index is not None:
                self.file_indexes[file_id] = cached_index
                logger.info(
                    f"Loaded {file_id} from disk cache: {len(cached_index.meta)} segments"
                )
                return file_id

            # Not cached - process using notebook's exact pattern
            logger.info(f"Processing new audio file: {file_id}")

            # Call segment_audio_embeddings exactly like run-workflow.ipynb does
            audio_index, media_paths = segment_audio_embeddings(
                self.clap_processor,
                self.clap_model,
                audio_path,  # Pass the file path directly
                MEDIA_EXTS,
                AUDIO_EXTS,
                VIDEO_EXTS,
            )

            if not audio_index.meta:
                raise ValueError(f"No audio segments extracted from {audio_path}")

            # Save to disk
            self._save_index(file_id, audio_index)

            # Cache in memory
            self.file_indexes[file_id] = audio_index

            logger.info(
                f"Successfully processed {file_id}: {len(audio_index.meta)} segments indexed"
            )

            return file_id

        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            raise

    # def _load_model(self) -> None:
    #     """Load the appropriate model based on configuration."""
    #     try:
    #         model_source = self.model_config.get("model_source", "local")
    #         logger.info(f"Loading model with source: {model_source}")

    #         from src.utils import initialize_llm, DEFAULT_MODELS

    #         # Extract secrets and model path based on configuration
    #         secrets = self.secrets if self.secrets else {}
    #         # Use model_path from notebook if provided, otherwise fall back to default
    #         local_model_path = self.model_path if self.model_path else DEFAULT_MODELS["local"]
    #         logger.info(f"Using local_model_path: {local_model_path}")

    #         hf_repo_id = self.model_config.get("hf_repo_id", "")

    #         # Use the shared initialize_llm function
    #         self.llm = initialize_llm(
    #             model_source=model_source,
    #             secrets=secrets,
    #             local_model_path=local_model_path,
    #             hf_repo_id=hf_repo_id
    #         )

    #         if self.llm is None:
    #             logger.error("Model failed to initialize - llm is None after loading")
    #             raise RuntimeError("Model initialization failed - llm is None")

    #         logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")

    #     except Exception as e:
    #         logger.error(f"Error loading model: {str(e)}")
    #         raise

    # Wrapper used by mlflow
    def _invoke(self, question: str, file_id: str = "global") -> dict:
        """
        Execute the agentic workflow for a question against a specific file index.
        Builds the graph with file-specific AudioIndex to match notebook behavior.
        """
        # Get or load the file-specific index
        audio_index = self.file_indexes.get(file_id)
        if audio_index is None:
            # Try loading from disk
            audio_index = self._load_index(file_id)
            if audio_index is not None:
                self.file_indexes[file_id] = audio_index
            else:
                # No index for this file_id - create empty one
                audio_index = AudioIndex(dim=512)
                logger.warning(
                    f"No index found for file_id '{file_id}', using empty index"
                )

        # Build graph with file-specific index (exactly like run-workflow does)
        graph = build_audio_agentic_graph(
            relevance_threshold=self.relevance_threshold,
            fetch_k=self.fetch_k,
            top_k=self.top_k,
            vecs=np.array([]).reshape(0, 512).astype(np.float32),  # Not used by graph
            metas=[],  # Not used by graph
            audio_index=audio_index,
            clap_processor=self.clap_processor,
            clap_model=self.clap_model,
        )

        # Execute with file-namespaced state (matching notebook pattern)
        return graph.invoke(
            {
                "question": question,
                "file_id": file_id,
                "index": audio_index,
                "memory": self.memory,
                "audio_llm": self.audio_llm,
                "messages": [],
            }
        )

    def predict(self, model_input):
        """
        Make predictions on input data.

        Input format:
        - DataFrame or list of dicts with columns/keys:
          - 'question': str (required) - The question to ask
          - 'file_id': str (optional) - Identifier for previously processed audio
          - 'audio_path': str (optional) - Path to audio file (for first-time processing)

        Returns:
        - List of prediction dictionaries containing:
          - 'question': str - The input question
          - 'file_id': str - File identifier
          - 'answer': str - Generated answer
          - 'evidence': list - Evidence segments with timestamps
          - 'from_memory': bool - Whether answer came from cache
        """
        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, list):
            records = model_input
        else:
            raise ValueError(
                "Pass a list[dict] or pandas DataFrame with 'question' and optional 'file_id' or 'audio_path'."
            )

        out = []
        for r in records:
            q = (r.get("question") or "").strip()
            audio_path = r.get("audio_path")
            fid = r.get("file_id", "").strip()

            try:
                # If audio_path provided, process it first
                if audio_path:
                    # Use audio filename as file_id if not provided
                    if not fid:
                        fid = Path(audio_path).name

                    # Process the audio file (will skip if already processed)
                    fid = self.process_audio_file(audio_path, fid)
                    logger.info(f"Audio file processed: {fid}")

                # Default to "global" if no file_id specified
                if not fid:
                    fid = "global"

                # Execute query
                s = self._invoke(q, fid)
                out.append(
                    {
                        "question": q,
                        "file_id": fid,
                        "answer": s.get("answer", ""),
                        "evidence": s.get("evidence", []),
                        "from_memory": s.get("from_memory", False),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing question '{q}': {e}")
                out.append(
                    {
                        "question": q,
                        "file_id": fid or "error",
                        "answer": "",
                        "evidence": [],
                        "from_memory": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
        return out
