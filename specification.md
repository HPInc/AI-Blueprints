# 🎓 Educational Quickstart Blueprint — Implementation Specification

**Document Type:** Implementation Specification
**Target Repository:** `AI-Blueprints/generative-ai/educational-quickstart/`
**Derived From:** `raw_specification.md` (SCAD Applied AI Design & Development)
**Date:** February 2026
**Version:** 2.0

---

## Table of Contents

1. [Objective & Scope](#1-objective--scope)
2. [Blueprint Location & Naming](#2-blueprint-location--naming)
3. [Project Structure](#3-project-structure)
4. [Config File](#4-config-file)
5. [Requirements File](#5-requirements-file)
6. [README.md](#6-readmemd)
7. [Source Modules (`src/`)](#7-source-modules-src)
8. [Notebooks (`notebooks/`)](#8-notebooks)
9. [Streamlit Demo App (`demo/streamlit/`)](#9-streamlit-demo-app-demostreamlit)
10. [Documentation (`docs/`)](#10-documentation-docs)
11. [Data Directory (`data/`)](#11-data-directory-data)
12. [Plotly Interactive Dashboard](#12-plotly-interactive-dashboard)
13. [GPU Optimization & Blackwell Support](#13-gpu-optimization--blackwell-support)
14. [MLflow Integration & Deployment Pipeline](#14-mlflow-integration--deployment-pipeline)
15. [Validated Model Registry](#15-validated-model-registry)
16. [Student Experience Requirements](#16-student-experience-requirements)
17. [Error Handling & Recovery](#17-error-handling--recovery)
18. [Acceptance Testing Criteria](#18-acceptance-testing-criteria)
19. [Implementation Checklist](#19-implementation-checklist)

---

## 1. Objective & Scope

### 1.1 — Goal

Create **one** new blueprint under `generative-ai/` named `educational-quickstart` that serves as a pre-configured, multi-capability AI development environment for SCAD (Savannah College of Art and Design) students enrolled in Applied AI Design & Development.

The blueprint must eliminate the 2–4 hour manual environment setup currently required, enabling creative students **with no prior coding experience** to begin building AI projects within 20 minutes of launching their workspace.

### 1.2 — What This Blueprint Delivers

A single blueprint that, once launched:

- Installs all dependencies and configures GPU access
- Authenticates Hugging Face
- Sets up MLflow tracking
- Provides **four starter project notebooks** that students can customize for:
  1. **Chatbot / Text Generation** (conversational AI with streaming)
  2. **Image Generation** (text-to-image with diffusion models)
  3. **Document Analysis** (PDF/Markdown upload with Q&A)
  4. **Voice Assistant** (speech-to-text + text-to-speech pipeline)
- Includes a centralized **Project Setup notebook** as the primary entry point
- Ships a **Streamlit-based UI** for interactive model inference (post-deployment)
- Embeds an **interactive Plotly monitoring dashboard** directly inside the notebooks

### 1.3 — Key Constraints

| # | Constraint |
|---|---|
| 1 | All four capabilities live inside a **single** blueprint: `generative-ai/educational-quickstart/` |
| 2 | The frontend framework is **Streamlit** (not Gradio). Streamlit is used in `demo/streamlit/main.py` exactly like every other blueprint in the repo. |
| 3 | Models are stored in `/home/jovyan/datafabric/` (the standard AI Studio datafabric path), **not** `data/models/`. |
| 4 | **Plotly dashboards** must be rendered inline inside notebooks (using `plotly.graph_objects` / `plotly.express` with `fig.show()`). No separate port-forwarded dashboard server. |
| 5 | No Trello integration of any kind. The Voice Assistant project focuses on general voice command processing. |
| 6 | Follow the **exact** repository pattern for folder structure, README format, config.yaml, requirements.txt, notebooks, demo/, docs/, and src/. |
| 7 | This specification is written for an **implementation agent** — every detail must be explicit and actionable. |

---

## 2. Blueprint Location & Naming

```
AI-Blueprints/
└── generative-ai/
    └── educational-quickstart/       ← NEW BLUEPRINT
```

After creation, update `generative-ai/README.md` to include the new blueprint in the listing, incrementing the count to **8 blueprint projects**. Use this entry:

```markdown
### 🎓 Educational Quickstart

**Educational Quickstart** is a pre-configured, multi-capability AI development environment designed for SCAD students in Applied AI Design & Development. It provides four ready-to-use starter notebooks covering text generation (chatbot), image generation (diffusion pipelines), document analysis (PDF/Markdown Q&A), and voice assistant (Whisper + TTS) — all pre-wired with GPU optimization, MLflow tracking, and a Streamlit frontend. Students with no coding experience can produce their first AI output within 20 minutes.
```

---

## 3. Project Structure

The blueprint **must** use the exact folder hierarchy below. This matches the pattern used by all existing generative-ai blueprints.

```bash
educational-quickstart/
├── configs/
│   └── config.yaml                          # Blueprint configuration (model paths, UI mode, ports, service)
├── data/
│   └── input/                               # Sample input data (feedback docs, sample PDFs, sample audio)
│       ├── sample_feedback.txt              # Sample text for document analyzer
│       └── sample_prompts.txt               # Sample prompts for chatbot/image gen
├── demo/
│   └── streamlit/
│       ├── main.py                          # Streamlit web app (calls MLflow /invocations endpoint)
│       ├── pyproject.toml                   # Poetry-based dependency spec
│       ├── README.md                        # Instructions for deploying Streamlit via AI Studio
│       ├── assets/
│       │   └── styles.css                   # Custom CSS for Streamlit app
│       └── static/
│           ├── HP-logo.png                  # HP branding logo
│           ├── Z-logo.png                   # Z by HP branding logo
│           └── AIS-logo.png                 # AI Studio branding logo
├── docs/
│   ├── streamlit-ss.png                     # Streamlit UI screenshot
│   └── architecture-diagram.png             # Blueprint architecture diagram
├── notebooks/
│   ├── run-workflow.ipynb                   # Main entry-point: Project Setup + GPU test + HF auth + all starters
│   ├── register-model.ipynb                 # MLflow model registration + deployment
│   ├── chatbot-starter.ipynb                # Starter: Conversational AI with streaming
│   ├── image-gen-starter.ipynb              # Starter: Text-to-image with diffusion
│   ├── document-analyzer-starter.ipynb      # Starter: PDF/Markdown Q&A
│   └── voice-assistant-starter.ipynb        # Starter: Speech-to-text + TTS
├── src/
│   ├── __init__.py                          # Empty init file
│   ├── utils.py                             # Shared utilities (logger, config loading, model path helpers)
│   ├── gpu_monitor.py                       # Plotly-based GPU monitoring utilities
│   ├── model_manager.py                     # Model loading, quantization, and download helpers
│   └── mlflow/                              # MLflow 3-layer architecture (universal pattern)
│       ├── __init__.py                      # Lazy-loading exports: Model, Logger
│       ├── model.py                         # Business Logic Layer (standalone Model class, no MLflow inheritance)
│       ├── loader.py                        # MLflow Loader Layer (_load_pyfunc entry point)
│       └── logger.py                        # MLflow Registration Layer (Logger.log_model classmethod)
├── requirements.txt                         # All required packages
└── README.md                                # Project documentation
```

---

## 4. Config File

**File:** `configs/config.yaml`

Must follow the **exact** v2.0.0 config format used across all blueprints. The config has top-level model settings, UI configuration, port configuration, and service configuration:

```yaml
# Blueprint Configuration
# This file configures the UI mode and ports for the model service

# Path to the local model file (used for LlamaCpp initialization)
model_path: "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

# Context window size for the LLM (default: 8192)
context_window: 8192

# Proxy is used to set the HTTPS_PROXY environment variable when necessary.
# For example, if you need to access external services from a restricted network,
# you should specify the proxy in this config.yaml file.
# proxy: "http://web-proxy.austin.hp.com:8080"

# UI Configuration
ui:
  # UI mode: gradio, streamlit, or static
  mode: "streamlit"

# Port Configuration
ports:
  # External port exposed by Envoy proxy
  external: 5000

  # Internal port mappings for different UI types
  internal:
    gradio: 7860
    streamlit: 8501
    static: 5001

# Service Configuration
service:
  # MLflow model server timeout (seconds)
  mlflow_timeout: 600

  # Health check timeout for service startup (seconds)
  health_check_timeout: 600

  # Number of health check retries
  health_check_retries: 5
```

**Key notes on v2.0.0 config format:**
- `model_path` is a **top-level** key (not nested under `models:`). The Model class reads it via `config.get("model_path")`.
- `context_window` is a **top-level** key used for LLM initialization.
- Environment variables (`PYTORCH_CUDA_ALLOC_CONF`, etc.) are **not** in config.yaml — they are set directly in notebook cells.
- There is no `models:` section — just a single `model_path` at the top level. Students who want additional models should add additional top-level keys (e.g., `image_model_path`, `stt_model_path`).

---

## 5. Requirements File

**File:** `requirements.txt`

Must follow the existing header convention and pin `mlflow==3.1.0`. The `Local GenAI` base image already includes PyTorch, CUDA libraries, and basic Python packages. Only list packages that are **not** in the base image.

```
# Create the workspace using the Local GenAI image.
# This image includes all the required libraries to run the sample project, except for the ones explicitly listed below.

# ─── Core ML/AI Frameworks ───
mlflow==3.1.0
transformers>=4.35.0
diffusers>=0.24.0
accelerate>=0.25.0
bitsandbytes>=0.41.3
safetensors>=0.4.1
sentencepiece>=0.1.99
xformers>=0.0.23

# ─── Interface & Deployment ───
huggingface_hub>=0.19.4
datasets
fastapi
uvicorn
pydantic>=2.5.0

# ─── Document & Media Processing ───
pypdf>=4.0.0
pymupdf
pdfplumber
python-docx>=1.1.0
markdown>=3.5.1
pillow
opencv-python

# ─── Audio & Voice Processing ───
sounddevice
librosa
soundfile
openai-whisper

# ─── Agentic AI & RAG ───
langchain>=0.3.26
langchain-community>=0.3.27
chromadb
faiss-cpu
sentence-transformers

# ─── Monitoring & Visualization ───
plotly>=5.18.0
psutil
gputil
pyyaml
python-dotenv
```

> **Note:** `faiss-cpu` is listed instead of `faiss-gpu` since the base image may handle GPU FAISS separately. If the base image includes `faiss-gpu`, replace accordingly. `vllm` is omitted from requirements because it is typically available in the base image or installed separately due to complex CUDA build dependencies.

---

## 6. README.md

**File:** `README.md`

Must follow the **exact** format of existing blueprints. Use the agentic-feedback-analyzer README as the canonical template. Below is the complete README:

````markdown
# 🎓 Educational Quickstart

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Model_Deployment-orange.svg?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend_App-ff4b4b.svg?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-red.svg?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg?logo=jupyter)

</div>

---

## 📚 Contents

* [🧠 Overview](#🧠-overview)
* [📁 Project Structure](#📁-project-structure)
* [⚙️ Setup](#⚙️-setup)
* [🚀 Usage](#🚀-usage)
* [📞 Contact & Support](#📞-contact--support)

---

## 🧠 Overview

The **Educational Quickstart** is a pre-configured, multi-capability AI development environment designed for SCAD students enrolled in Applied AI Design & Development.

It provides:

* 🤖 **Chatbot Starter** — Conversational AI using LLMs with streaming, system prompt support, and conversation memory
* 🎨 **Image Generation Starter** — Text-to-image generation using diffusion models with parameter controls
* 📄 **Document Analyzer Starter** — PDF/Markdown upload and analysis with question-answering capability
* 🎙️ **Voice Assistant Starter** — Speech-to-text input, command processing, and text-to-speech response
* 📊 **Interactive GPU Monitoring** — Real-time Plotly dashboards for GPU utilization, memory, and performance tracking
* 📦 **MLflow Deployment** — Full model packaging, registration, and REST API deployment pipeline
* 🌐 **Streamlit UI** — Interactive web interface for deployed model inference

This blueprint eliminates the 2–4 hour manual environment setup, enabling students with no prior coding experience to produce their first AI output within 20 minutes.

---

## 📁 Project Structure

```bash
educational-quickstart/
├── configs/                                 # Configuration files
│   └── config.yaml
├── data/                                    # Sample input data
│   └── input/
├── demo/                                    # UI frontend code (Streamlit)
│   └── streamlit/
├── docs/                                    # Screenshots and architecture diagrams
│   ├── streamlit-ss.png
│   └── architecture-diagram.png
├── notebooks/                               # Workflow and starter notebooks
│   ├── run-workflow.ipynb                   # Main entry point: setup + GPU test + auth
│   ├── register-model.ipynb                 # MLflow model registration
│   ├── chatbot-starter.ipynb               # Conversational AI starter
│   ├── image-gen-starter.ipynb             # Image generation starter
│   ├── document-analyzer-starter.ipynb     # Document analysis starter
│   └── voice-assistant-starter.ipynb       # Voice assistant starter
├── requirements.txt                         # All required packages
├── README.md                                # Project documentation
└── src/                                     # Core Python modules
    ├── __init__.py
    ├── utils.py                             # Shared utilities
    ├── gpu_monitor.py                       # Plotly GPU monitoring
    ├── model_manager.py                     # Model loading helpers
    └── mlflow/                              # MLflow 3-layer architecture
        ├── __init__.py                      # Lazy-loading: Model, Logger
        ├── model.py                         # Business logic (standalone Model)
        ├── loader.py                        # MLflow loader (_load_pyfunc)
        └── logger.py                        # MLflow registration (Logger.log_model)
```

---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements

* ✅ **GPU**: NVIDIA GPU with 8 GB VRAM (48 GB+ recommended for image generation and large models)
* ✅ **RAM**: 32–64 GB system memory
* ✅ **Disk**: ≥ 20 GB free space

### Step 1: Create an AI Studio Project

1. Go to [HP AI Studio](https://hp.com/ai-studio) and create a new project.
2. Use the base image: `Local GenAI`

### Step 2: Add Required Assets

Download the following model(s) via the Models tab:

**Required (Text Generation):**
- **Model Name**: `meta-llama3.1-8b-Q8`
- **Model Source**: `AWS S3`
- **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
- **Resource Type**: `public`
- **Bucket Region**: `us-west-2`

**Optional (Image Generation):**
- **Model Name**: `sdxl-turbo`
- Download from Hugging Face: `stabilityai/sdxl-turbo`

**Optional (Speech-to-Text):**
- **Model Name**: `whisper-large-v3`
- Download from Hugging Face: `openai/whisper-large-v3`

Make sure that each model is in the `datafabric` folder inside your workspace. If a model does not appear after downloading, please restart your workspace.

### Step 3: Configure Secrets (if needed)

If you need Hugging Face access for gated models, add your token:
- Go to **Project Settings → Secrets**
- Add a secret named `HF_TOKEN` with your Hugging Face access token

### Step 4: Configuration

Review and update `configs/config.yaml` if you need to change model paths or proxy settings.

---

## 🚀 Usage

### 🧪 Step 1: Run Project Setup

Open and run the main entry-point notebook:

```bash
notebooks/run-workflow.ipynb
```

This notebook will:
* Configure the CUDA environment
* Install and validate all dependencies
* Test GPU availability and memory
* Authenticate with Hugging Face
* Display a comprehensive setup summary with next steps

### 🤖 Step 2: Run a Starter Notebook

Choose a starter notebook based on your project interest:

| Notebook | Description |
|----------|------------|
| `chatbot-starter.ipynb` | Conversational AI with LLM streaming |
| `image-gen-starter.ipynb` | Text-to-image with diffusion models |
| `document-analyzer-starter.ipynb` | PDF/Markdown Q&A analysis |
| `voice-assistant-starter.ipynb` | Speech-to-text + text-to-speech |

### 📦 Step 3: Register Model with MLflow

Log and serve your model as an MLflow `pyfunc` model:

```bash
notebooks/register-model.ipynb
```

This registers the model so it can be queried over HTTP via AI Studio Publishing Services.

### 🌐 Step 4: Launch the Streamlit Web App

1. After completing the local deployment, open the Streamlit web app using the deployment URL provided by AI Studio.
2. For additional details on how the Streamlit app works, refer to the `README.md` file in the `demo/streamlit` folder.

---

## 📞 Contact & Support

  - **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

  - **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

  - **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

  - **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio)
````

---

## 7. Source Modules (`src/`)

The v2.0.0 architecture separates concerns into two layers:
1. **Shared utilities** (`src/utils.py`, `src/gpu_monitor.py`, `src/model_manager.py`) — framework-agnostic helpers
2. **MLflow 3-layer architecture** (`src/mlflow/`) — standardized universal structure for model logging, loading, and inference

### 7.1 — `src/__init__.py`

Empty file (matches existing pattern).

### 7.2 — `src/utils.py`

Must include the following utilities, matching the pattern in existing blueprints (e.g., `agentic-feedback-analyzer-with-langgraph/src/utils.py`):

```python
# Required functions and classes:

class EmojiStyledJupyterHandler(logging.Handler):
    """Rich HTML-styled log output for Jupyter notebooks with emoji icons per log level."""

logger  # Module-level logger using EmojiStyledJupyterHandler

def log_timing(func):
    """Decorator that logs execution time of a function."""

def load_config(config_path: str) -> dict:
    """Load and return YAML config file. Uses yaml.safe_load. Returns empty dict if not found."""

def get_model_path(model_name: str) -> str:
    """
    Get the full path to the model file using MODEL_ARTIFACTS_PATH env var.
    Extracts filename from model_name, joins with artifacts path.
    Used by both notebooks and the MLflow loader.
    """

def get_response_from_llm(llm, system_prompt: str, user_prompt: str) -> str:
    """Format and send a prompt to a LlamaCpp model using Meta-Llama chat template."""

def display_image(image_bytes: bytes, width: int = 400) -> str:
    """Render image bytes as inline HTML in Jupyter."""

def json_schema_from_type(input_type: type) -> dict:
    """Convert a Python type to a JSON schema dict for MLflow signatures."""

def log_asset_status(asset_path: str, asset_name: str) -> None:
    """Log pass/fail status of a given asset path."""
```

### 7.3 — `src/gpu_monitor.py`

GPU monitoring utilities that produce **Plotly figures rendered inline in notebooks** (no separate server, no port forwarding).

```python
# Required functions:

def get_gpu_stats() -> dict:
    """
    Collect current GPU statistics: utilization %, memory used/total,
    temperature, and power draw. Uses gputil and torch.cuda.
    Returns dict with keys: gpu_name, utilization, memory_used_mb,
    memory_total_mb, memory_percent, temperature, power_draw.
    Gracefully returns empty/default values if GPU is not available.
    """

def create_gpu_dashboard(history: list[dict]) -> plotly.graph_objects.Figure:
    """
    Create a multi-panel Plotly figure with:
    - GPU Utilization % over time (line chart)
    - Memory Usage over time (area chart)
    - Temperature over time (line chart)
    - Current stats summary (indicator gauges)
    Returns a plotly Figure object that can be displayed with fig.show() in notebooks.
    """

def log_gpu_metrics_to_mlflow(stats: dict):
    """
    Log GPU stats as MLflow metrics: gpu_utilization, gpu_memory_used_mb,
    gpu_temperature. Uses mlflow.log_metric().
    """

class GPUMonitor:
    """
    Stateful monitor that collects GPU stats over time.
    Methods:
    - snapshot(): Collect and store current stats
    - dashboard(): Return Plotly figure of all collected snapshots
    - log_to_mlflow(): Log latest snapshot to MLflow
    - summary(): Return formatted string summary of current GPU state
    """
```

### 7.4 — `src/model_manager.py`

Centralized model loading helpers that use the **datafabric** path pattern.

```python
# Required functions:

DATAFABRIC_BASE = "/home/jovyan/datafabric"

def verify_model_exists(model_path: str) -> bool:
    """Check if model path exists and log status."""

def load_llm(model_path: str, n_ctx: int = 8192, **kwargs) -> LlamaCpp:
    """
    Load a LlamaCpp model with Blackwell-optimized defaults:
    - n_gpu_layers=-1 (offload all layers to GPU)
    - n_batch=512
    - f16_kv=True
    - use_mmap=False
    - temperature=0.0
    - seed=42
    - num_threads=multiprocessing.cpu_count()
    Returns a LlamaCpp instance.
    """

def load_diffusion_pipeline(model_path: str, **kwargs):
    """
    Load a diffusion model (SDXL-Turbo or FLUX) from datafabric.
    Uses diffusers.AutoPipelineForText2Image with:
    - torch_dtype=torch.float16
    - variant="fp16" (if available)
    - Moves to CUDA device
    Returns a diffusion pipeline.
    """

def load_whisper_model(model_path: str, **kwargs):
    """
    Load a Whisper model for speech-to-text.
    Uses the openai-whisper library or transformers pipeline.
    Returns a whisper model instance.
    """

def get_quantization_config(model_size_b: float) -> dict:
    """
    Return BitsAndBytes quantization config based on model size:
    - Under 13B: 4-bit NF4
    - 13B-70B: FP8 or 8-bit
    - 70B+: 4-bit GPTQ recommendation
    """

def safe_cuda_cleanup():
    """Clear CUDA cache and call garbage collection."""
```

### 7.5 — MLflow 3-Layer Architecture (`src/mlflow/`)

This is the **universal pattern** used by all v2.0.0 blueprints. The `src/mlflow/` directory contains exactly 4 files that cleanly separate business logic, model loading, and MLflow registration.

#### 7.5.1 — `src/mlflow/__init__.py`

Lazy-loading module exports. **Must match this exact pattern:**

```python
"""
MLflow models-from-code implementation for Educational Quickstart.

This module provides the standardized universal structure for MLflow model logging
and loading using the models-from-code approach.
"""

__all__ = ["Model", "Logger"]


def __getattr__(name):
    """Dynamic import for backwards compatibility and lazy loading."""
    if name == "Model":
        from .model import Model
        return Model
    if name == "Logger":
        from .logger import Logger
        return Logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

#### 7.5.2 — `src/mlflow/model.py` (Business Logic Layer)

**Standalone Model class with NO MLflow inheritance.** This is a key v2.0.0 change — the Model class is a pure domain object. It does NOT inherit from `mlflow.pyfunc.PythonModel`.

```python
"""
Standalone Model class.

Business Logic Layer
- Handles multi-capability AI inference (chatbot, image gen, document analysis, voice)
- Manages model initialization, embeddings, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

class Model:
    """
    Educational quickstart model with multi-capability support.
    Pure domain functionality with zero MLflow dependencies.
    """

    def __init__(self, config, docs_path=None, model_path=None, secrets=None):
        """
        Initialize the Model with configuration and paths.
        Constructor signature follows v2.0.0 universal pattern.

        Args:
            config: Configuration dictionary (loaded from config.yaml)
            docs_path: Path to documents directory (for document analysis)
            model_path: Path to LLM model file (from datafabric)
            secrets: Dictionary containing secrets (optional)
        """
        self.config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM and other components based on config."""
        # Load LlamaCpp model from model_path
        # Set up context window from config.get("context_window", 8192)
        # Initialize any RAG components if docs_path provided

    def predict(self, model_input, params=None):
        """
        Core business logic for multi-capability inference.
        Must accept pandas DataFrame and return pandas DataFrame.

        Args:
            model_input: DataFrame with capability-specific columns
            params: Optional parameters for different operations

        Returns:
            pandas.DataFrame with response columns
        """
```

**Key v2.0.0 differences from old pattern:**
- **No `mlflow.pyfunc.PythonModel` inheritance** — Model is a pure Python class
- **No `load_context(self, context)`** — initialization happens in `__init__` via constructor args
- **Constructor signature**: `(config, docs_path=None, model_path=None, secrets=None)` — standardized across all generative-ai blueprints
- **`predict(self, model_input, params=None)`** — takes DataFrame directly (not `context, model_input`)
- The config dict comes from `config.yaml`, loaded by the loader

#### 7.5.3 — `src/mlflow/loader.py` (MLflow Loader Layer)

Entry point for MLflow's models-from-code approach. The `_load_pyfunc` function is called by MLflow to reconstruct the Model from logged artifacts.

```python
"""
MLflow models-from-code loader module for Educational Quickstart.
This module provides the _load_pyfunc function required by MLflow's models-from-code approach.
"""

import os
import logging

logger = logging.getLogger(__name__)


def _load_pyfunc(data_path: str):
    """
    MLflow models-from-code loader function.
    Called by MLflow to load the model from artifacts.

    Args:
        data_path: Path to model artifacts directory containing:
            - config.yaml: Model configuration
            - data/: Document directory
            - secrets.yaml: Secrets (optional)
            - models/: LLM model files (optional)
            - demo/: Demo folder with UI components (optional)

    Returns:
        Model: Initialized model instance ready for prediction
    """
    from src.mlflow.model import Model
    from src.utils import load_config

    # Load config
    config_path = os.path.join(data_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config = load_config(config_path)

    # Load secrets if available
    secrets_path = os.path.join(data_path, "secrets.yaml")
    if os.path.exists(secrets_path):
        from src.utils import load_secrets_to_env, load_secrets
        load_secrets_to_env(secrets_path)
        secrets = load_secrets()
    else:
        secrets = None

    # Set up documents path
    docs_path = os.path.join(data_path, "data")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents directory not found at: {docs_path}")

    # Resolve model path from config
    model_path = config.get("model_path")
    if model_path:
        from src.utils import get_model_path
        models_artifacts_path = os.path.join(data_path, "models")
        os.environ["MODEL_ARTIFACTS_PATH"] = models_artifacts_path
        resolved_model_path = get_model_path(model_path)
        model_path = resolved_model_path

    # Initialize and return Model
    model = Model(config=config, docs_path=docs_path, model_path=model_path, secrets=secrets)
    return model
```

**Key pattern details:**
- Imports `Model` from `src.mlflow.model` (not from a flat src/ module)
- Uses `src.utils.load_config` to parse config.yaml
- Reads `config.get("model_path")` to resolve model location
- Sets `MODEL_ARTIFACTS_PATH` env var for path resolution inside artifacts
- Returns a fully initialized Model instance

#### 7.5.4 — `src/mlflow/logger.py` (MLflow Registration Layer)

Handles artifact organization and MLflow model logging via the models-from-code approach.

```python
"""
Logger Service implementation for MLflow model logging.

MLflow Registration Layer
- Provides log_model functionality for models
- Handles artifact organization and temporary directory management
- Uses MLflow's models-from-code approach for deployment
- Manages configuration, documents, secrets, and demo assets
"""

import os
import logging
import shutil
import tempfile
import yaml

logger = logging.getLogger(__name__)


class Logger:
    """
    Logger Service for MLflow model logging.
    Packages model artifacts for deployment via models-from-code approach.
    """

    @classmethod
    def log_model(
        cls,
        signature,
        artifact_path="AIStudio-Model",
        config_path="configs/config.yaml",
        docs_path="data/",
        secrets_dict=None,
        model_path=None,
        demo_folder=None,
    ):
        """
        Log model using MLflow's models-from-code approach.

        Organizes artifacts into a temp directory, then calls mlflow.pyfunc.log_model
        with loader_module="src.mlflow.loader" so MLflow knows how to reconstruct
        the Model at load time.

        Final MLflow artifact structure:
        /artifacts/
          └── data/                    # MLflow automatically created
              ├── config.yaml          # Configuration
              ├── data/                # Documents directory
              ├── demo/                # UI components
              ├── models/              # Model files (optional)
              └── secrets.yaml         # Secrets (optional)

        Args:
            signature: MLflow ModelSignature defining input/output schema
            artifact_path: Name for the logged model artifact
            config_path: Path to configuration file
            docs_path: Path to documents directory
            secrets_dict: Dict with secrets to persist as YAML (optional)
            model_path: Path to model file (optional)
            demo_folder: Path to demo folder (optional)
        """
        import mlflow

        # Create temp directory for organizing artifacts
        temp_base = tempfile.gettempdir()
        temp_dir = os.path.join(temp_base, "model_artifacts")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            # Copy config.yaml
            shutil.copy2(config_path, os.path.join(temp_dir, "config.yaml"))

            # Copy documents to data/ subdirectory
            data_temp_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_temp_dir, exist_ok=True)
            if docs_path and os.path.exists(docs_path):
                for item in os.listdir(docs_path):
                    item_path = os.path.join(docs_path, item)
                    if os.path.isfile(item_path):
                        shutil.copy2(item_path, data_temp_dir)
                    elif os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(data_temp_dir, item))

            # Copy demo folder if provided
            if demo_folder and os.path.exists(demo_folder):
                shutil.copytree(demo_folder, os.path.join(temp_dir, "demo"))

            # Write secrets.yaml if provided
            if secrets_dict:
                with open(os.path.join(temp_dir, "secrets.yaml"), "w") as f:
                    yaml.safe_dump(secrets_dict, f)

            # Copy model files if provided
            if model_path and os.path.exists(model_path):
                models_temp_dir = os.path.join(temp_dir, "models")
                os.makedirs(models_temp_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    shutil.copy2(model_path, os.path.join(models_temp_dir, os.path.basename(model_path)))
                else:
                    shutil.copytree(model_path, models_temp_dir, dirs_exist_ok=True)

            # Log model using models-from-code approach
            mlflow.pyfunc.log_model(
                name=artifact_path,
                loader_module="src.mlflow.loader",
                data_path=temp_dir,
                code_paths=["../src"],
                signature=signature,
                pip_requirements="../requirements.txt",
            )
        except Exception as e:
            logger.error(f"Error during model logging: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
```

**Key v2.0.0 Logger pattern details:**
- `@classmethod log_model(cls, ...)` — called as `Logger.log_model(...)` (no instance needed)
- Uses `mlflow.pyfunc.log_model(name=..., loader_module="src.mlflow.loader", ...)` — the `loader_module` parameter tells MLflow to call `_load_pyfunc` from `src.mlflow.loader` when loading
- `code_paths=["../src"]` — packages the entire `src/` directory with the model
- `pip_requirements="../requirements.txt"` — ensures dependencies are tracked
- Temp directory cleanup in `finally` block
- `data_path=temp_dir` — MLflow stores this under `/artifacts/data/`

---

## 8. Notebooks

All notebooks must follow the **exact visual and structural pattern** of existing blueprint notebooks:
- Cell 1: Centered HTML `<h1>` title
- Cell 2: `📘 Project Overview:` markdown
- Cell 3: `# Notebook Overview` with bulleted TOC
- Start Execution cell with `start_time = time.time()`
- `sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))` for importing from `src/`
- `%pip install -r ../requirements.txt --quiet`
- `%%time` magic on long-running cells
- Final cell: `Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio).`
- Final timing cell: elapsed minutes/seconds log
- All code cells must have **comments on every non-obvious line**, written for readers who have never seen Python
- All markdown cells must have **plain-language explanations** of what each section does and why

### 8.1 — `notebooks/run-workflow.ipynb` (Main Entry Point)

This is the **Project Setup** notebook — the primary entry point for all students. Mirrors the role described in the raw specification's `Project_Setup.ipynb` Section 4.5.1.

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1 style="text-align: center; font-size: 50px;"> 🎓 Educational Quickstart — Project Setup </h1>` |
| 2 | Markdown | Overview | `📘 Project Overview:` Describe the blueprint purpose: multi-capability AI dev environment for SCAD students. |
| 3 | Markdown | TOC | `# Notebook Overview` with sections: Start Execution, CUDA Configuration, Install & Import Libraries, GPU Validation, AI Library Verification, Hugging Face Authentication, GPU Monitoring Dashboard, Setup Summary, Quick Reference Guide |
| 4 | Markdown | Section | `# Start Execution` |
| 5 | Python | Start | `import os, sys, time; sys.path.append(...); start_time = time.time()` and import `logger` from `src.utils` |
| 6 | Markdown | Section | `# CUDA Configuration` — Explain: these environment variables prevent GPU memory fragmentation and must run before any other GPU code. |
| 7 | Python | CUDA | Set `os.environ["PYTORCH_CUDA_ALLOC_CONF"]`, `os.environ["CUDA_LAUNCH_BLOCKING"]`. Log success. |
| 8 | Markdown | Section | `# Install and Import Libraries` — Explain: this installs additional packages not included in the base image (estimated 2–5 minutes). |
| 9 | Python | Install | `%%time` + `%pip install -r ../requirements.txt --quiet` |
| 10 | Python | Imports | Import `torch`, `transformers`, `diffusers`, `accelerate`, `mlflow`, `datasets`, `huggingface_hub`, `plotly`, `PIL`, `yaml`. Import from `src.utils` and `src.gpu_monitor`. |
| 11 | Markdown | Section | `# GPU Validation` — Explain: 4-step test to confirm GPU is working (detection, memory allocation, matrix multiply, cleanup). Expected: all 4 pass. |
| 12 | Python | GPU Test | 4-step GPU validation: (1) `torch.cuda.is_available()`, (2) `torch.cuda.get_device_name(0)`, (3) allocate tensor + matrix multiply, (4) `torch.cuda.empty_cache()`. Print pass/fail for each. If GPU not detected, print clear warning about CPU fallback. |
| 13 | Markdown | Section | `# AI Library Verification` — Explain: verifies all critical libraries imported successfully. |
| 14 | Python | Lib Test | Import-test each library in try/except blocks: `torch`, `transformers`, `diffusers`, `accelerate`, `bitsandbytes`, `mlflow`, `huggingface_hub`, `plotly`, `langchain`, `chromadb`. Print version + pass/fail. Print summary: `X/10 libraries verified`. |
| 15 | Markdown | Section | `# Hugging Face Authentication` — Explain: required for downloading gated models. Provides step-by-step instructions for getting a token from huggingface.co/settings/tokens. |
| 16 | Python | HF Auth | `from huggingface_hub import login`. Try to login using `HF_TOKEN` env var or secrets. If not available, provide `login()` interactive prompt. Wrap in try/except with friendly error message and re-auth instructions. |
| 17 | Markdown | Section | `# GPU Monitoring Dashboard` — Explain: interactive Plotly charts showing current GPU state. |
| 18 | Python | Dashboard | Use `GPUMonitor` from `src.gpu_monitor`. Take a snapshot. Create and display Plotly dashboard with `fig.show()`. Log metrics to MLflow. |
| 19 | Markdown | Section | `# Setup Summary` — Explain: comprehensive status report. |
| 20 | Python | Summary | Print: GPU name+memory, CUDA version, Python version, library count, HF auth status, model path status (check datafabric), config file status. Print "Next Steps" guiding students to starter notebooks. |
| 21 | Markdown | Section | `# Quick Reference Guide` — Code snippets for each project type. |
| 22 | Python | Reference | Print formatted code snippets showing: (a) load a text model, (b) generate text, (c) load diffusion pipeline, (d) generate image, (e) load documents, (f) Plotly visualization example. All snippets use datafabric paths. |
| 23 | Python | Timing | Elapsed time calculation + logger completion message. |
| 24 | Markdown | Footer | `Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio).` |

### 8.2 — `notebooks/chatbot-starter.ipynb`

**Derived from:** Three-Model LLM Deployment, Chatbot MLflow Deployment, Multi-Modal AI System (chat module).

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1>` Chatbot Starter |
| 2 | Markdown | Overview | Conversational AI using LLM with streaming, system prompts, and memory |
| 3 | Markdown | TOC | Start Execution, Install & Import, Configure Settings, Verify Assets, Load Model, Conversation Loop, GPU Monitoring, Timing |
| 4 | Markdown | Section | `# Start Execution` |
| 5 | Python | Start | Standard start cell with sys.path, logger, start_time |
| 6 | Markdown | Section | `# Install and Import Libraries` |
| 7 | Python | Install | `%pip install -r ../requirements.txt --quiet` |
| 8 | Python | Imports | Import torch, LlamaCpp, IPython.display, yaml, src modules |
| 9 | Markdown | Section | `# Configure Settings` |
| 10 | Python | Config | `MODEL_PATH = "/home/jovyan/datafabric/meta-llama3.1-8b-Q8/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"`, context window, max tokens constants. System prompt variable. Conversation history list. |
| 11 | Markdown | Section | `# Verify Assets` |
| 12 | Python | Verify | Use `log_asset_status` to check model path |
| 13 | Markdown | Section | `# Load Model` — Explain: loads the LLM with GPU acceleration. Expected time ~30–60 seconds. |
| 14 | Python | Load | `%%time` + Use `load_llm()` from model_manager or direct LlamaCpp instantiation with Blackwell-optimized defaults (n_gpu_layers=-1, f16_kv=True, etc.) |
| 15 | Markdown | Section | `# Chat: Single Response` — Explain: send one message and see the response. Students can modify the question. |
| 16 | Python | Chat | Define a user question. Call `get_response_from_llm()`. Display response with `Markdown(response)`. |
| 17 | Markdown | Section | `# Chat: Conversation Loop` — Explain: multi-turn conversation with memory. |
| 18 | Python | Loop | Implement simple conversation loop with history tracking. Show 3 example turns. Display formatted conversation history. |
| 19 | Markdown | Section | `# GPU Monitoring` |
| 20 | Python | Monitor | Use `GPUMonitor.snapshot()` + `GPUMonitor.dashboard()` → `fig.show()` |
| 21 | Python | Timing | Standard timing cell |
| 22 | Markdown | Footer | Built with ❤️ |

**Key details:**
- System prompt support: allow students to customize system prompts (e.g., "You are a helpful art teacher")
- Conversation memory: maintain a list of `{"role": ..., "content": ...}` dicts
- Streaming: if LlamaCpp supports streaming, show streaming output; otherwise show complete response
- **Next Steps** markdown cell suggesting: change the system prompt, try different questions, modify temperature

### 8.3 — `notebooks/image-gen-starter.ipynb`

**Derived from:** FLUX.1-dev Pipeline, Multi-Modal Image Gen, SDXL-Turbo deployment.

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1>` Image Generation Starter |
| 2 | Markdown | Overview | Text-to-image generation using diffusion models with parameter controls |
| 3 | Markdown | TOC | Sections list |
| 4–5 | | Start Execution | Standard start cell |
| 6–7 | | Install & Import | Standard install + import diffusers, torch, PIL, plotly, base64, io |
| 8–9 | | Configure Settings | `MODEL_PATH = "/home/jovyan/datafabric/sdxl-turbo/"` (or configurable). Parameters: `NUM_INFERENCE_STEPS = 4`, `GUIDANCE_SCALE = 0.0` (turbo), `WIDTH = 512`, `HEIGHT = 512`, `SEED = 42`. |
| 10–11 | | Verify Assets | Check model path exists |
| 12–13 | | Load Pipeline | `%%time` + Load with `AutoPipelineForText2Image.from_pretrained()`, float16, move to CUDA |
| 14–15 | | Generate Single Image | Define prompt. Generate image. Display with `IPython.display.Image` or `PIL.Image.show()`. |
| 16–17 | | Generate Image Gallery | Generate 4 images with different prompts. Display as grid using Plotly subplots (`plotly.subplots.make_subplots` with `plotly.graph_objects.Image`). Use `fig.show()`. |
| 18–19 | | Parameter Exploration | Show how changing seeds / guidance_scale / steps affects output. Display comparison grid via Plotly. |
| 20–21 | | GPU Monitoring | `GPUMonitor` snapshot + dashboard |
| 22 | | Timing | Standard timing cell |
| 23 | Markdown | Next Steps | Suggest: change prompts, try negative prompts, adjust parameters |
| 24 | Markdown | Footer | Built with ❤️ |

**Key details:**
- If SDXL-Turbo model is not in datafabric, notebook must print friendly message: "Image generation model not found in datafabric. Please follow the Setup instructions in README.md to download the model."
- Display generated images inline using Plotly or PIL + IPython.display
- Save generated images to `../data/output/` directory (create if not exists)

### 8.4 — `notebooks/document-analyzer-starter.ipynb`

**Derived from:** Multi-Modal AI System (document analysis module), Agentic Feedback Analyzer.

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1>` Document Analyzer Starter |
| 2 | Markdown | Overview | PDF/Markdown/text upload and analysis with Q&A. Chunks documents, generates per-chunk answers, synthesizes final answer. |
| 3 | Markdown | TOC | Sections list |
| 4–5 | | Start Execution | Standard |
| 6–7 | | Install & Import | Standard + langchain loaders, RecursiveCharacterTextSplitter, LlamaCpp |
| 8–9 | | Configure Settings | `MODEL_PATH`, `INPUT_PATH = Path("../data/input")`, `CONTEXT_WINDOW = 8192`, `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| 10–11 | | Verify Assets | Check model path + input path |
| 12–13 | | Load Model | `%%time` + load LlamaCpp |
| 14–15 | | Load Documents | Use langchain loaders (TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, etc.) to load all files from INPUT_PATH. Support: .txt, .csv, .xlsx, .docx, .pdf, .md. Log count per file. |
| 16–17 | | Chunk Documents | RecursiveCharacterTextSplitter. Log chunk count. **Plotly chart**: display bar chart of chunk sizes using `plotly.express.bar()` → `fig.show()`. |
| 18–19 | | Define Question | `QUESTION = "What are the main themes in these documents?"` — students modify this. |
| 20–21 | | Generate Per-Chunk Answers | Loop through chunks, generate answer per chunk using LLM with progress bar (tqdm). |
| 22–23 | | Synthesize Final Answer | Combine chunk answers into grouped synthesis. Display with `Markdown()`. |
| 24–25 | | Visualization | **Plotly chart**: display chunk response lengths as a bar chart, display token usage overview. |
| 26–27 | | GPU Monitoring | `GPUMonitor` snapshot + dashboard |
| 28 | | Timing | Standard |
| 29 | Markdown | Next Steps | Suggest: upload own PDFs to data/input/, change the question, try different chunk sizes |
| 30 | Markdown | Footer | Built with ❤️ |

### 8.5 — `notebooks/voice-assistant-starter.ipynb`

**Derived from:** Voice-Controlled Assistant project (without Trello). Focuses on general voice command processing.

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1>` Voice Assistant Starter |
| 2 | Markdown | Overview | Speech-to-text transcription using Whisper, command processing with LLM, text-to-speech feedback |
| 3 | Markdown | TOC | Sections list |
| 4–5 | | Start Execution | Standard |
| 6–7 | | Install & Import | Standard + whisper, sounddevice, soundfile, librosa |
| 8–9 | | Configure Settings | `STT_MODEL_PATH = "/home/jovyan/datafabric/whisper-large-v3/"`, `LLM_MODEL_PATH`, `SAMPLE_RATE = 16000`, `DURATION = 5` (seconds) |
| 10–11 | | Verify Assets | Check STT model path + LLM model path |
| 12–13 | | Load Whisper Model | `%%time` + Load Whisper model for speech-to-text |
| 14–15 | | Load LLM | `%%time` + Load LlamaCpp for command processing |
| 16–17 | | Audio Recording | Provide function to record audio from microphone using sounddevice. Include fallback: if no microphone available, load sample audio from `data/input/`. Display waveform using **Plotly line chart** → `fig.show()`. |
| 18–19 | | Speech-to-Text | Transcribe audio using Whisper. Display transcription. |
| 20–21 | | Command Processing | Send transcription to LLM for processing/response. Display LLM response. |
| 22–23 | | Full Pipeline Demo | End-to-end: record → transcribe → process → respond. If no mic, use pre-recorded sample. |
| 24–25 | | Audio Visualization | **Plotly charts**: audio waveform, spectrogram visualization using librosa + plotly. |
| 26–27 | | GPU Monitoring | `GPUMonitor` snapshot + dashboard |
| 28 | | Timing | Standard |
| 29 | Markdown | Next Steps | Suggest: try different audio inputs, change LLM system prompt for different response styles, experiment with Whisper model sizes |
| 30 | Markdown | Footer | Built with ❤️ |

**Key details:**
- **No Trello integration.** The voice assistant processes general commands/questions.
- Graceful fallback when microphone is not available (use a sample .wav file in `data/input/`)
- If Whisper model is not in datafabric, degrade gracefully with friendly message

### 8.6 — `notebooks/register-model.ipynb`

Follows the **exact** v2.0.0 pattern of existing `register-model.ipynb` notebooks (e.g., `agentic-feedback-analyzer-with-langgraph`). Uses the universal `Logger.log_model()` approach.

**Cell sequence:**

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 1 | Markdown | Title | `<h1 style="text-align: center; font-size: 50px;"> 📦 Register Model </h1>` |
| 2 | Markdown | Overview | `📘 Project Overview:` Register an AI model with MLflow for deployment via HP AI Studio Publishing Services |
| 3 | Markdown | TOC | `# Notebook Overview` with sections: Start Execution, Define User Constants, Install and Import Libraries, Configure Settings, Verify Assets, MLflow Registration, Test Registered Model |
| 4 | Markdown | Section | `# Start Execution` |
| 5 | Python | Start | `import os, sys, time; sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))` + import from `src.utils` |
| 6 | Python | Timer | `start_time = time.time(); logger.info("Notebook execution started.")` |
| 7 | Markdown | Section | `# Define User Constants` |
| 8 | Python | Constants | Define test input constants (e.g., `QUESTION = "What is AI?"`) |
| 9 | Markdown | Section | `# Install and Import Libraries` |
| 10 | Python | Install | `%%time` + `%pip install -r ../requirements.txt --quiet` |
| 11 | Python | Imports | Import `mlflow`, `MlflowClient`, `ModelSignature`, `Schema`, `ColSpec`, `yaml`, `warnings`. **Key import:** `from src.mlflow import Logger` |
| 12 | Markdown | Section | `# Configure Settings` |
| 13 | Python | Config | Load config, resolve model_path, define `EXPERIMENT_NAME`, `RUN_NAME`, `MODEL_NAME`. Set `CONTEXT_WINDOW`, `MAX_TOKENS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`. |
| 14 | Markdown | Section | `# Verify Assets` |
| 15 | Python | Verify | `log_asset_status()` for input data and model path |
| 16 | Markdown | Section | `# MLflow Registration` |
| 17 | Python | Set MLflow | `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "/phoenix/mlflow"))` + `mlflow.set_experiment(EXPERIMENT_NAME)` |
| 18 | Python | Signature | Define `input_schema` and `output_schema` using `Schema([ColSpec(...)])`. Create `signature = ModelSignature(inputs=input_schema, outputs=output_schema)`. |
| 19 | Python | Register | `%%time` + **v2.0.0 registration pattern:** |

**Cell 19 — v2.0.0 MLflow Registration Pattern (verbatim):**

```python
%%time

# === Start MLflow run, log, and register ===
with mlflow.start_run(run_name=RUN_NAME) as run:
    print(f"🚀 Started MLflow run: {run.info.run_id}")

    # Log and register the model using the new universal Logger
    Logger.log_model(
        signature=signature,
        artifact_path=MODEL_NAME,
        config_path="../configs/config.yaml",
        docs_path="../data/input",
        model_path=model_path,
        demo_folder="../demo"
    )

    # Construct the URI for the logged model
    model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"

    # Register the model into MLflow Model Registry
    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

logger.info(f"✅ Model '{MODEL_NAME}' successfully logged and registered.")
```

| Cell # | Type | Section | Content |
|--------|------|---------|---------|
| 20 | Python | Version | Retrieve latest version: `client = MlflowClient()`, `client.get_latest_versions(MODEL_NAME, stages=["None"])` |
| 21 | Python | Load Test | `%%time` + `loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{latest_version}")` |
| 22 | Python | Inference | Run sample inference: `input_payload = [{"question": QUESTION}]`, `results = loaded_model.predict(input_payload)` |
| 23 | Python | Timing | Standard elapsed time calculation |
| 24 | Markdown | Footer | `Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio).` |

**Key v2.0.0 differences from old pattern:**
- Uses `from src.mlflow import Logger` (not a service-specific class)
- Calls `Logger.log_model(signature=..., artifact_path=..., config_path=..., docs_path=..., model_path=..., demo_folder=...)` — standardized signature
- `mlflow.register_model(model_uri=..., name=...)` is called **separately** after `Logger.log_model()`
- The Logger handles all artifact organization (temp dir, config copy, docs copy, secrets, demo folder)
- No `MODEL_ARTIFACTS` dict — arguments are passed directly to `Logger.log_model()`

---

## 9. Streamlit Demo App (`demo/streamlit/`)

### 9.1 — `demo/streamlit/main.py`

The Streamlit app must call the MLflow `/invocations` endpoint at `http://localhost:5002/invocations` (matching existing pattern). It must support **multiple project types** via a tabbed or sidebar-selection interface.

**Required structure:**

```python
import streamlit as st
import requests
import json
import base64

st.set_page_config(
    page_title="Educational Quickstart",
    page_icon="🎓",
    layout="wide"
)

# CSS styling (load from assets/styles.css)
# Logo bar (HP, Z, AIS)
# Gradient header

# Sidebar: select project type (Chatbot, Image Gen, Document Analyzer, Voice Assistant)
# Based on selection, render appropriate input form
# On submit, POST to http://localhost:5002/invocations with correct payload format
# Display response

endpoint_url = "http://localhost:5002/invocations"

# Tab/sidebar selection for:
# 1. Chatbot: text input for question + system prompt → display answer
# 2. Image Gen: text input for prompt + sliders for parameters → display image
# 3. Document Analyzer: file upload + question → display answer
# 4. Voice Assistant: file upload (audio) or text fallback → display transcription + response
```

**Payload format** (must match MLflow pyfunc `predict` input):

```python
# Chatbot
payload = {"inputs": [{"question": question, "system_prompt": system_prompt}], "params": {}}

# Image Gen
payload = {"inputs": [{"prompt": prompt, "seed": seed, "num_inference_steps": steps}], "params": {}}

# Document Analyzer
payload = {"inputs": [{"question": question, "input_text": document_text}], "params": {}}

# Voice Assistant
payload = {"inputs": [{"audio_base64": audio_b64, "command_text": text}], "params": {}}
```

### 9.2 — `demo/streamlit/pyproject.toml`

```toml
[tool.poetry]
name = "educational-quickstart-streamlit-webapp"
version = "0.1.0"
description = "A Streamlit front-end for AIS Educational Quickstart Blueprint Project"
authors = ["HP AI Studio <ai-studio@hp.com>"]
package-mode = false

[tool.poetry.dependencies]
python = ">=3.11,<4.0"
streamlit = ">=1.43.1,<2.0.0"
pillow = ">=11.3.0"
urllib3 = ">=2.5.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

### 9.3 — `demo/streamlit/README.md`

Follow the exact pattern of existing Streamlit README:

```markdown
# How to Successfully Use the Streamlit Web App

## 1. Install Required Versions
Ensure that the following are installed on your machine:
- **Python** version **≥ 3.11** (https://www.python.org/downloads/)
- **Poetry** version **≥ 2.0.0 and < 3.0.0** (https://python-poetry.org/docs/)

## 2. Set Up the Virtual Environment and Install Dependencies
Navigate to the project's root directory and run the following command to set up a virtual environment using Poetry and install all required packages:
```bash
python -m poetry install
```

## 3. Launch the Streamlit Web App
Still in the project's root directory, start the Streamlit app by running:
```bash
python -m poetry run streamlit run "main.py"
```

## 4. Select the Correct API Endpoint When Using the App
When interacting with the app:
- **Choose the exact and correct API URL** to connect to your deployed model.
- **Important:** The MLflow endpoint **must** use **HTTPS** (not HTTP).
- **Note:** In **Z by HP AI Studio**, the **port number** for your MLflow API **changes with each deployment**, so always verify the correct URL and port before starting a session.
```

### 9.4 — `demo/streamlit/assets/styles.css`

Custom CSS matching brand guidelines. Minimal stylesheet with:
- Gradient header styling
- Result box styling
- Card layout for multi-project view

### 9.5 — `demo/streamlit/static/`

Include placeholder logo files: `HP-logo.png`, `Z-logo.png`, `AIS-logo.png`. These should be the same logo files used in other blueprints. Copy from an existing blueprint's `demo/streamlit/static/` folder.

---

## 10. Documentation (`docs/`)

### 10.1 — `docs/streamlit-ss.png`

Screenshot of the Streamlit UI. **Implementation agent:** create a placeholder PNG or leave a TODO. This will be replaced with an actual screenshot after the app is functional.

### 10.2 — `docs/architecture-diagram.png`

Architecture diagram showing the blueprint flow:
```
┌─────────────────────────────────────────────────────┐
│                  Educational Quickstart               │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Chatbot  │  │Image Gen│  │Doc Anal.│  │ Voice   │ │
│  │ Starter  │  │ Starter │  │ Starter │  │ Assist. │ │
│  └────┬─────┘  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │              │            │             │      │
│       └──────────────┴────────────┴─────────────┘      │
│                        │                               │
│              ┌─────────┴─────────┐                     │
│              │   src/ modules    │                      │
│              │  (model_manager,  │                      │
│              │   gpu_monitor,    │                      │
│              │   services)       │                      │
│              └─────────┬─────────┘                     │
│                        │                               │
│              ┌─────────┴─────────┐                     │
│              │  /datafabric/     │                      │
│              │  (LLM, Diffusion, │                      │
│              │   Whisper, etc.)  │                      │
│              └─────────┬─────────┘                     │
│                        │                               │
│              ┌─────────┴─────────┐                     │
│              │  MLflow Registry  │──► Publishing        │
│              └───────────────────┘    Services → API   │
│                        │                               │
│              ┌─────────┴─────────┐                     │
│              │  Streamlit UI     │                      │
│              └───────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

**Implementation agent:** Create this as a PNG using Plotly, matplotlib, or a simple diagramming approach. Alternatively, create a Mermaid diagram in the notebook and export it.

---

## 11. Data Directory (`data/`)

### 11.1 — `data/input/sample_feedback.txt`

A sample plaintext feedback document (100–200 words) that the Document Analyzer starter can process out of the box. Example content about a fictional product review.

### 11.2 — `data/input/sample_prompts.txt`

A text file with 5–10 example prompts for the chatbot and image generation starters:

```
# Chatbot prompts
What is artificial intelligence?
Explain machine learning to a 10-year-old.
Write a short poem about creativity and technology.

# Image generation prompts
A serene mountain landscape at sunset, digital art
A futuristic city with flying cars, cyberpunk style
A watercolor painting of a cat reading a book
An abstract geometric pattern in vibrant colors
```

---

## 12. Plotly Interactive Dashboard

**Critical constraint:** Plotly dashboards run **inside notebooks only** via `fig.show()`. No separate server. No port forwarding.

### 12.1 — Where Plotly Is Used

| Notebook | Plotly Usage |
|----------|-------------|
| `run-workflow.ipynb` | GPU monitoring dashboard (utilization, memory, temperature gauges) |
| `chatbot-starter.ipynb` | GPU monitoring dashboard after inference |
| `image-gen-starter.ipynb` | Image gallery display via Plotly subplots; GPU monitoring |
| `document-analyzer-starter.ipynb` | Chunk size distribution bar chart; response length analysis; GPU monitoring |
| `voice-assistant-starter.ipynb` | Audio waveform visualization; spectrogram; GPU monitoring |

### 12.2 — GPU Dashboard Specification

The `src/gpu_monitor.py` creates a Plotly figure with **4 subplots**:

1. **GPU Utilization (%)** — Line chart over time, green/yellow/red color zones
2. **Memory Usage (MB)** — Area chart over time with used vs. total
3. **Temperature (°C)** — Line chart with warning threshold line at 80°C
4. **Current Stats** — Plotly Indicator gauges for current utilization, memory, temp

The figure must:
- Use `plotly.subplots.make_subplots(rows=2, cols=2)` layout
- Have a clean, professional theme (`plotly_white` template)
- Include proper axis labels and titles
- Work with `fig.show()` in Jupyter (no external renderer needed)
- Gracefully handle absence of GPU (show all zeros with warning text)

### 12.3 — MLflow Metric Logging

Every time `GPUMonitor.snapshot()` is called, the following metrics are logged to MLflow:

```python
mlflow.log_metric("gpu_utilization", stats["utilization"])
mlflow.log_metric("gpu_memory_used_mb", stats["memory_used_mb"])
mlflow.log_metric("gpu_temperature", stats["temperature"])
```

This provides monitoring data visible in the MLflow UI's metric charts.

---

## 13. GPU Optimization & Blackwell Support

All GPU code must follow these Blackwell-optimized defaults. These are embedded in `src/model_manager.py` and used by all notebooks.

### 13.1 — Memory Management Defaults

| Configuration | Value | Notes |
|---|---|---|
| Default Quantization | 4-bit NF4 via BitsAndBytes | For models under 13B parameters |
| Large Model Strategy | FP8 / 8-bit | For 13B+ param models |
| Massive Model Strategy | 4-bit GPTQ + vLLM PagedAttention | For 70B+ param models |
| Memory Allocation | `gpu_memory_utilization=0.9` | Reserves 10% for KV-cache overhead |
| OOM Recovery | `torch.cuda.empty_cache()` + retry | Automatic on OutOfMemoryError |
| Context Management | Dynamic truncation at 80% of max context | Preserves system message |
| Batch Processing | Queue-based concurrency limit (max_size=20) | Prevents memory overflow |

### 13.2 — BitsAndBytes Config (used in model_manager.py)

```python
from transformers import BitsAndBytesConfig

quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### 13.3 — Flash Attention

When loading transformer models (not LlamaCpp), enable Flash Attention 2:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    ...
)
```

### 13.4 — torch.compile

For production inference (optional, advanced):
```python
model = torch.compile(model, mode="reduce-overhead")
```

---

## 14. MLflow Integration & Deployment Pipeline

### 14.1 — v2.0.0 Three-Layer Architecture

The v2.0.0 pattern introduces a **3-layer MLflow architecture** inside `src/mlflow/` that cleanly separates concerns:

| Layer | File | Responsibility |
|-------|------|---------------|
| **Business Logic** | `src/mlflow/model.py` | Standalone `Model` class. No MLflow dependencies. Pure domain logic with `__init__` + `predict`. |
| **Loader** | `src/mlflow/loader.py` | `_load_pyfunc(data_path)` entry point. Reconstructs Model from MLflow artifacts. Called by MLflow at load time. |
| **Registration** | `src/mlflow/logger.py` | `Logger.log_model()` classmethod. Organizes artifacts in temp dir, calls `mlflow.pyfunc.log_model(loader_module="src.mlflow.loader")`. |

This replaces the previous pattern where service classes inherited `mlflow.pyfunc.PythonModel` with `load_context()` and `predict(context, model_input)`.

### 14.2 — Models-From-Code Approach

MLflow 3.1.0 uses the **models-from-code** approach instead of cloudpickle serialization:

1. **Logging:** `Logger.log_model()` calls `mlflow.pyfunc.log_model(name=..., loader_module="src.mlflow.loader", data_path=temp_dir, code_paths=["../src"], ...)`.
   - `loader_module` tells MLflow which module contains the `_load_pyfunc` function
   - `code_paths=["../src"]` packages the entire `src/` directory as model code
   - `data_path` points to the temp directory with organized artifacts
   - `pip_requirements="../requirements.txt"` tracks dependencies

2. **Loading:** When `mlflow.pyfunc.load_model()` is called, MLflow:
   - Extracts the code from `code_paths`
   - Calls `src.mlflow.loader._load_pyfunc(data_path)` where `data_path` points to the extracted artifacts
   - The loader reads config.yaml, resolves model paths, initializes the Model, and returns it

3. **Prediction:** MLflow calls `model.predict(model_input)` on the returned Model instance.

### 14.3 — Three-Stage Pipeline

| Stage | Notebook | Description |
|-------|----------|-------------|
| **Develop & Test** | Starter notebooks | Students develop and test in Jupyter. Inline Plotly feedback. |
| **Register** | `register-model.ipynb` | Uses `Logger.log_model()` + `mlflow.register_model()`. Packages artifacts via models-from-code. |
| **Deploy** | AI Studio Publishing Services | Auto-creates REST API endpoints. Streamlit app queries these endpoints. |

### 14.4 — MLflow Tracking URI

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "/phoenix/mlflow"))
```

This matches the pattern used by all existing blueprints.

### 14.5 — Model Registration Pattern (v2.0.0)

```python
from src.mlflow import Logger

# Define signature
input_schema = Schema([ColSpec("string", "question")])
output_schema = Schema([ColSpec("string", "answer"), ColSpec("string", "messages")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log and register
with mlflow.start_run(run_name=RUN_NAME) as run:
    Logger.log_model(
        signature=signature,
        artifact_path=MODEL_NAME,
        config_path="../configs/config.yaml",
        docs_path="../data/input",
        model_path=model_path,
        demo_folder="../demo"
    )
    model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
```

### 14.6 — Service Endpoint

After deployment, the model is accessible at:
- **Local:** `http://localhost:5002/invocations`
- **Published:** HTTPS URL provided by AI Studio (port changes per deployment)

### 14.7 — Containerization Support

Include instructions in the README (or a separate markdown note in `docs/`) for advanced students who want to containerize. Reference the base image:

```
Base: Local GenAI image (pytorch/pytorch with CUDA 12.1)
Ports: 8501 (Streamlit), 5000 (MLflow external)
```

Do **not** include an actual Dockerfile — the AI Studio platform handles containerization natively.

---

## 15. Validated Model Registry

The following models have been tested and validated. The blueprint documentation and Quick Reference cells should reference these as recommended starting points.

| Category | Model | Params | VRAM | Use Case | Datafabric Path |
|----------|-------|--------|------|----------|----------------|
| **Text Generation** | `meta-llama3.1-8b-Q8` | 8B | 6–8 GB | Production chat, general AI | `/home/jovyan/datafabric/meta-llama3.1-8b-Q8/` |
| **Text Generation** | `microsoft/phi-2` | 2.7B | 4 GB | Code, light chat | `/home/jovyan/datafabric/phi-2/` |
| **Text Generation** | `HuggingFaceH4/zephyr-7b-beta` | 7B | 6 GB | Production chat | `/home/jovyan/datafabric/zephyr-7b-beta/` |
| **Text Generation** | `mistralai/Mistral-7B-Instruct` | 7B | 6 GB | Document analysis | `/home/jovyan/datafabric/mistral-7b-instruct/` |
| **Image Gen** | `stabilityai/sdxl-turbo` | 3.5B | 8 GB | Fast image gen | `/home/jovyan/datafabric/sdxl-turbo/` |
| **Image Gen** | `black-forest-labs/FLUX.1-dev` | 12B | 24 GB | High-quality text-to-image | `/home/jovyan/datafabric/flux-1-dev/` |
| **Speech-to-Text** | `openai/whisper-large-v3` | 1.5B | 4 GB | Audio transcription | `/home/jovyan/datafabric/whisper-large-v3/` |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | 335M | 1 GB | RAG search | `/home/jovyan/datafabric/bge-large-en-v1.5/` |

**Required model (must work out of box):** `meta-llama3.1-8b-Q8` — this is the same model used by multiple existing blueprints and has a known S3 download path.

**Optional models:** All others. Notebooks must degrade gracefully when optional models are absent.

---

## 16. Student Experience Requirements

### 16.1 — First-Run Timeline

| Milestone | Time Target | What Happens |
|-----------|-------------|--------------|
| **Launch** | 0 min | Student selects blueprint and creates workspace |
| **Ready** | 5 min | Workspace opens with all deps installed, GPU verified |
| **First Output** | 15 min | Student runs `run-workflow.ipynb`, sees GPU test pass, library verification pass |
| **First AI Output** | 20 min | Student runs a starter notebook, generates first AI response or image |
| **Custom Project** | 30 min | Student modifies a starter notebook with own prompts/data |
| **Deployment** | 60 min | Student registers model with MLflow and deploys via Publishing Services |

### 16.2 — Documentation Requirements (All Notebooks)

Every notebook must include:
1. **Plain-language markdown headers** explaining what each cell does and why
2. **Expected output descriptions** so students know what success looks like
3. **Time estimates** for long-running cells (e.g., "⏱️ Expected: 2–5 minutes")
4. **Inline troubleshooting** guidance (not in a separate document)
5. **"Next Steps" sections** guiding students to their next action
6. **Code comments on every non-obvious line**, written for a reader who has never seen Python
7. **Emoji-styled log messages** using `EmojiStyledJupyterHandler` for visual feedback

### 16.3 — Accessibility

- All code must be readable without prior Python knowledge
- Variable names must be descriptive (no single-letter vars except loop counters)
- Magic commands (`%%time`, `%pip`) must have markdown explanations above them
- Error messages must be in plain English with actionable suggestions

---

## 17. Error Handling & Recovery

The following error scenarios must be handled gracefully in the notebooks and `src/` modules:

| Issue | Handling |
|-------|---------|
| **GPU not detected** | Fall back to CPU mode. Print: "⚠️ No GPU detected. Running in CPU mode (slower). Check GPU drivers if this is unexpected." Set `n_gpu_layers=0` in LlamaCpp. |
| **Out of memory (OOM)** | Auto-clear CUDA cache with `torch.cuda.empty_cache()`. Reduce batch size. Retry once. Log error to monitoring. Print: "⚠️ GPU ran out of memory. Cleared cache and retrying with reduced batch size." |
| **Hugging Face auth failure** | Print step-by-step re-authentication instructions: (1) Go to huggingface.co/settings/tokens, (2) Create new token, (3) Add as AI Studio secret `HF_TOKEN`, (4) Restart workspace. |
| **Model not in datafabric** | Print: "❌ Model not found at [path]. Please follow the Setup instructions in README.md Step 2 to download the model." Do not crash — exit the cell gracefully. |
| **Model download interruption** | Use `resume_download=True` in all `snapshot_download` and `from_pretrained` calls. |
| **Library import failure** | Catch `ImportError`, identify the specific library, print: "❌ [library] not found. Run: `%pip install [library]`" |
| **MLflow not accessible** | Print: "⚠️ MLflow server not accessible. Check that MLflow is running. Continuing without experiment tracking." |
| **Notebook kernel crash** | Include markdown cell at top of each notebook: "If your kernel crashes, restart it (Kernel → Restart) and re-run cells from the top." |
| **No microphone (voice)** | Detect with try/except on sounddevice. Fall back to sample audio file. Print: "ℹ️ No microphone detected. Using sample audio file for demonstration." |

---

## 18. Acceptance Testing Criteria

### 18.1 — Environment Tests

1. `torch.cuda.is_available()` returns `True` (or graceful CPU fallback)
2. `torch.cuda.get_device_name(0)` reports a CUDA GPU
3. GPU memory allocation and matrix multiplication complete without error
4. All libraries in `requirements.txt` import successfully
5. MLflow server starts and is accessible at `/phoenix/mlflow`
6. Plotly figures render inline in Jupyter with `fig.show()`

### 18.2 — Workflow Tests

7. `run-workflow.ipynb` runs all cells sequentially without errors
8. `chatbot-starter.ipynb` generates a text response from LLM
9. `image-gen-starter.ipynb` produces an image (if model available) or prints graceful error
10. `document-analyzer-starter.ipynb` processes sample_feedback.txt and returns an answer
11. `voice-assistant-starter.ipynb` processes audio (sample file) and returns transcription
12. `register-model.ipynb` successfully registers a model with MLflow
13. Streamlit app launches and displays the multi-project interface
14. GPU monitoring Plotly dashboard displays in all notebooks

### 18.3 — Student Experience Tests

15. A user with no Python experience can run `run-workflow.ipynb` by following written instructions only
16. A user can modify a starter notebook prompt and see different output within 5 minutes
17. Error messages are understandable without technical background
18. All time estimates in markdown cells are accurate within 50%
19. Every notebook renders proper markdown formatting and inline Plotly charts

---

## 19. Implementation Checklist

Use this checklist to track implementation progress. Each item maps to a section above.

- [ ] Create `generative-ai/educational-quickstart/` directory structure (Section 3)
- [ ] Create `configs/config.yaml` with v2.0.0 format (Section 4)
- [ ] Create `requirements.txt` with `mlflow==3.1.0` (Section 5)
- [ ] Create `README.md` (Section 6)
- [ ] Create `src/__init__.py` (empty)
- [ ] Create `src/utils.py` with shared utilities (Section 7.2)
- [ ] Create `src/gpu_monitor.py` with Plotly GPU dashboard (Section 7.3)
- [ ] Create `src/model_manager.py` with model loading helpers (Section 7.4)
- [ ] Create `src/mlflow/__init__.py` with lazy-loading exports (Section 7.5.1)
- [ ] Create `src/mlflow/model.py` — standalone Model class, NO PythonModel inheritance (Section 7.5.2)
- [ ] Create `src/mlflow/loader.py` — `_load_pyfunc` entry point (Section 7.5.3)
- [ ] Create `src/mlflow/logger.py` — `Logger.log_model()` classmethod (Section 7.5.4)
- [ ] Create `notebooks/run-workflow.ipynb` (Section 8.1)
- [ ] Create `notebooks/chatbot-starter.ipynb` (Section 8.2)
- [ ] Create `notebooks/image-gen-starter.ipynb` (Section 8.3)
- [ ] Create `notebooks/document-analyzer-starter.ipynb` (Section 8.4)
- [ ] Create `notebooks/voice-assistant-starter.ipynb` (Section 8.5)
- [ ] Create `notebooks/register-model.ipynb` with v2.0.0 Logger pattern (Section 8.6)
- [ ] Create `demo/streamlit/main.py` (Section 9.1)
- [ ] Create `demo/streamlit/pyproject.toml` (Section 9.2)
- [ ] Create `demo/streamlit/README.md` (Section 9.3)
- [ ] Create `demo/streamlit/assets/styles.css` (Section 9.4)
- [ ] Copy `demo/streamlit/static/` logos from existing blueprint (Section 9.5)
- [ ] Create `docs/` placeholder files (Section 10)
- [ ] Create `data/input/sample_feedback.txt` (Section 11.1)
- [ ] Create `data/input/sample_prompts.txt` (Section 11.2)
- [ ] Update `generative-ai/README.md` to list new blueprint (Section 2)
- [ ] Verify all notebooks execute without errors (Section 18)
- [ ] Verify Plotly dashboards render in all notebooks (Section 12)
- [ ] Verify graceful degradation when optional models are absent (Section 17)
- [ ] Verify `Logger.log_model()` → `mlflow.register_model()` pipeline works end-to-end (Section 14)
- [ ] Verify `mlflow.pyfunc.load_model()` correctly calls `_load_pyfunc` and returns working Model (Section 14)

---

**End of Implementation Specification**

This specification was derived from `raw_specification.md` (SCAD Applied AI Development, Dan Bartlett, February 2026) and adapted to match the v2.0.0 AI-Blueprints repository patterns. Key v2.0.0 adaptations: MLflow 3.1.0, 3-layer MLflow architecture (`src/mlflow/{model.py, loader.py, logger.py}`), models-from-code approach, standalone Model class (no PythonModel inheritance), and `Logger.log_model()` registration pattern. All Gradio references have been replaced with Streamlit. All `data/models/` paths have been replaced with `/home/jovyan/datafabric/`. All Plotly visualizations run inline in notebooks (no separate server). No Trello integration is included.
