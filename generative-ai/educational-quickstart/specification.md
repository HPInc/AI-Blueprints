# HP AI STUDIO — SCAD Education Blueprint Specification

**Document Type:** Specification Document
**Prepared for:** HP AI Studio Engineering Team
**Prepared by:** Dan Bartlett, SCAD Applied AI Design & Development
**Date:** February 2026
**Version:** 1.0

---

## Table of Contents
1. Executive Summary
2. Projects Analyzed
3. Common Architectural Patterns
   - 3.1 Universal Environment Layer
   - 3.2 Model Acquisition Layer
   - 3.3 GPU Optimization Layer
   - 3.4 Interface Layer
   - 3.5 Deployment & Tracking Layer
4. Blueprint Specification
   - 4.1 Blueprint Metadata
   - 4.2 Pre-Installed Dependencies
   - 4.3 Pre-Configured Environment Variables
   - 4.4 Pre-Configured Directory Structure
   - 4.5 Included Starter Notebooks
5. Blackwell GPU Optimization Requirements
   - 5.1 Memory Management Defaults
   - 5.2 Performance Monitoring
6. MLflow Integration & Deployment Pipeline
   - 6.1 Stage 1: Develop & Test
   - 6.2 Stage 2: Register with MLflow
   - 6.3 Stage 3: Deploy via Publishing Services
   - 6.4 Containerization Support
7. Student Experience Requirements
   - 7.1 First-Run Experience
   - 7.2 Documentation Requirements
   - 7.3 Error Handling & Recovery
8. Validated Model Registry
9. Acceptance Testing Criteria
   - 9.1 Environment Tests
   - 9.2 Workflow Tests
   - 9.3 Student Experience Tests
10. Appendix: Project Source References

---

## 1. Executive Summary
This document specifies the ideal HP AI Studio Blueprint designed for SCAD (Savannah College of Art and Design) students enrolled in Applied AI Design & Development. The specification is derived from the analysis of nine completed HP AI Studio projects spanning LLM deployment, multi-modal AI systems, voice-controlled assistants, image generation pipelines, and agentic AI prototypes—all built and tested on NVIDIA Blackwell GPU hardware within HP AI Studio environments.

The goal is a single, pre-configured blueprint that eliminates the 2–4 hour manual environment setup currently required, enabling creative students with no prior coding experience to begin building AI projects within 20 minutes of launching their workspace.

**Core Objective**
Deliver a one-click blueprint that installs all dependencies, configures GPU access, authenticates Hugging Face, sets up MLflow tracking, and launches a Gradio-based starter interface—ready for students to customize for text generation, image creation, document analysis, voice processing, or agentic workflows.

---

## 2. Projects Analyzed
The following table summarizes all HP AI Studio projects developed by SCAD faculty and project collaborators that informed this blueprint specification. Each project was built from scratch, tested on Blackwell GPUs, and deployed through HP AI Studio’s infrastructure.

| Project | Description | Key Models | Core Stack |
| :--- | :--- | :--- | :--- |
| **Three-Model LLM Deployment** | CPU-only light model, Phi-2, and Zephyr-7B deployed independently with Blackwell optimization | Phi-2, Zephyr-7B-Beta, DialoGPT | vLLM, BitsAndBytes, MLflow, FastAPI |
| **R1-1776 671B Deployment** | Massive 671B-parameter MoE model deployed via 4-bit GPTQ quantization across 8 GPUs | perplexity-ai/r1-1776 (DeepSeek-R1) | vLLM, PagedAttention, FP8/FP4, Gradio |
| **Multi-Modal AI System** | Document analysis, conversational AI, image generation, and custom system prompts in unified interface | Mistral-7B-Instruct, SDXL-Turbo, Nous-Hermes-2 | Gradio (multi-tab), MLflow, diffusers |
| **Multi-Modal Image Gen** | Text-to-image and image-to-image generation with reference uploads and style control | FLUX.1-dev, ControlNet, IP-Adapter | diffusers, Gradio, MLflow, xformers |
| **FLUX.1-dev Pipeline** | Production text-to-image diffusion with FP8 quantization and batch processing | black-forest-labs/FLUX.1-dev | diffusers, FP8, Flash Attention, Gradio |
| **Voice-Controlled Trello** | Speech-to-text command recognition, NLP parsing, Trello API execution, text-to-speech feedback | Whisper (INT8), XTTS-v2 | pyaudio, Trello API, WebSocket, Gradio |
| **Agentic Web Intelligence** | Autonomous web monitoring, content summarization, email drafting with ReAct agent pattern | Mistral-7B / Zephyr-7B, LangChain | ChromaDB/FAISS, LangChain, RAG |
| **Chatbot MLflow Deployment** | Gradio chatbot with auto-generated Register_Model notebook and Publishing Services integration | Zephyr-7B-Beta, Phi-2 | Gradio ChatInterface, MLflow, REST API |
| **Student Project Template** | Standardized 10-cell notebook with GPU tests, dependency installs, HF auth, and Register_Model generation | Any HuggingFace model | PyTorch, Gradio, MLflow, transformers |

*Note: All projects were built on HP Z-series workstations running Ubuntu 22.04 with NVIDIA Blackwell GPUs (64GB RAM, 20TB storage) and HP AI Studio’s containerized workspace environment.*

---

## 3. Common Architectural Patterns
Across all nine projects, a consistent architecture emerged. Every successful deployment shares the same foundational layers, regardless of whether the project involves text generation, image synthesis, voice processing, or autonomous agents.

### 3.1 — Universal Environment Layer
Every project begins with an identical environment configuration sequence, currently performed manually:

| Step | Action | Details |
| :--- | :--- | :--- |
| 1 | **CUDA Environment Variables** | Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`, `CUDA_LAUNCH_BLOCKING=0` |
| 2 | **PyTorch CUDA Installation** | `pip install torch torchvision torchaudio` with cu121 index URL, verified via `torch.cuda.is_available()` |
| 3 | **Core Library Imports** | `torch`, `transformers`, `diffusers`, `accelerate`, `gradio`, `mlflow`, `datasets`, `huggingface_hub` |
| 4 | **GPU Validation** | 4-step test: device detection, memory allocation, matrix operation, memory cleanup |
| 5 | **AI Library Installation** | `transformers`, `diffusers`, `accelerate`, `mlflow`, `gradio`, `datasets`, `safetensors`, `sentencepiece`, `bitsandbytes` |
| 6 | **Infrastructure Testing** | Import verification for all libraries with version reporting and pass/fail summary |
| 7 | **Hugging Face Authentication** | Token-based login with credential storage using `huggingface_hub.login()` |
| 8 | **Register_Model.ipynb Generation** | Auto-create MLflow deployment notebook with model wrapper, signature, and stage transition |

**Blueprint Implication:** Steps 1–8 above should be fully automated within the blueprint. Zero manual setup should be required for the base environment. This alone eliminates 2–4 hours of student time per project.

### 3.2 — Model Acquisition Layer
All projects pull models from Hugging Face using a consistent pattern: `snapshot_download` with `local_dir_use_symlinks=False`, `max_workers=8`, and `resume_download=True`. Models are stored at `data/models` with a standardized directory structure organized by capability: text-generation, text-to-image, speech-to-text, embeddings.

### 3.3 — GPU Optimization Layer
Blackwell-specific optimizations appear in every GPU-bound project:
- **FP8/FP4 Quantization:** Native Blackwell precision formats via the second-generation Transformer Engine, reducing memory 2–3.5x with less than 1% accuracy loss.
- **4-bit NF4 via BitsAndBytes:** Used universally for 7B-parameter models (Mistral, Zephyr, Phi-2), enabling single-GPU deployment.
- **vLLM with PagedAttention:** Applied to all production inference workloads for continuous batching, memory defragmentation, and 2-4x throughput improvement.
- **Flash Attention 2:** Enabled via `attn_implementation="flash_attention_2"` on all transformer models for O(N) memory scaling.
- **torch.compile:** Applied with `mode="reduce-overhead"` for Blackwell-specific kernel optimization and reduced launch latency.

### 3.4 — Interface Layer
Gradio is the universal interface framework across all projects. The consistent pattern includes:
- Server binding to `0.0.0.0` on port `7860` for HP AI Studio network accessibility
- Multi-tab interfaces (`gr.TabbedInterface`) for projects with multiple capabilities
- Queue-based concurrency control (`demo.queue(max_size=20)`) for GPU memory protection
- Streaming response support for real-time text generation and voice processing
- Custom CSS for professional appearance within the Gradio container

### 3.5 — Deployment & Tracking Layer
MLflow integration follows an identical pattern across all projects:
- MLflow tracking URI set to `http://localhost:5000` (HP AI Studio default)
- Experiment creation with descriptive naming (e.g., "multimodal-ai-system")
- Model registration with `pytorch.log_model` including `pip_requirements`
- Production stage transition for visibility in HP AI Studio "Deployments" tab
- Metric logging for GPU utilization, inference latency, memory usage, and tokens/second
- Publishing Services deployment for persistent REST API endpoints via Swagger

---

## 4. Blueprint Specification
Based on the architectural patterns identified across all projects, the following specification defines the ideal SCAD Education Blueprint for HP AI Studio.

### 4.1 — Blueprint Metadata
| Key | Value |
| :--- | :--- |
| **Blueprint Name** | SCAD Applied AI Development Environment |
| **Version** | 1.0 |
| **Target Audience** | Art & design students with no prior coding or AI experience |
| **Target Platform** | HP AI Studio with NVIDIA Blackwell GPUs |
| **Operating System** | Ubuntu 22.04 LTS |
| **GPU Requirement** | NVIDIA Blackwell (B200/B300), minimum 48GB VRAM |
| **System RAM** | 64GB minimum |
| **Storage** | 20TB (accommodates multiple large models) |
| **CUDA Version** | 12.1 with Blackwell-optimized kernels |
| **Python Version** | 3.10 |
| **Primary Interface** | Jupyter Notebook + Gradio web UI |

### 4.2 — Pre-Installed Dependencies
The blueprint container must include the following packages pre-installed and verified. These are the union of all dependencies required across the nine analyzed projects.

#### 4.2.1 — Core ML/AI Frameworks
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch torchvision torchaudio` | 2.1.0+cu121 | Core deep learning framework with CUDA 12.1 GPU support |
| `transformers` | 4.35.0+ | Hugging Face model loading, tokenization, and inference |
| `diffusers` | 0.24.0+ | Stable Diffusion, FLUX, and image generation pipelines |
| `accelerate` | 0.25.0+ | Distributed training, mixed precision, and device management |
| `bitsandbytes` | 0.41.3+ | 4-bit and 8-bit quantization (NF4, INT8) for memory optimization |
| `vllm` | latest | High-throughput LLM serving with PagedAttention and continuous batching |
| `xformers` | 0.0.23+ | Memory-efficient attention for transformer and diffusion models |
| `safetensors` | 0.4.1+ | Fast, safe model weight serialization |
| `sentencepiece` | 0.1.99 | Tokenizer support for Mistral, LLaMA, and similar models |

#### 4.2.2 — Interface & Deployment
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `gradio` | 4.8.0+ | Web UI framework for all model interactions (launches on port 7860) |
| `mlflow` | 2.9.0+ | Experiment tracking, model registry, and HP AI Studio deployment |
| `huggingface_hub` | 0.19.4+ | Model/dataset downloads, authentication, and Hub API access |
| `datasets` | latest | Hugging Face dataset loading and preprocessing |
| `fastapi uvicorn` | latest | REST API serving for multi-model unified endpoints |
| `pydantic` | 2.5.0+ | Data validation for API request/response schemas |

#### 4.2.3 — Document & Media Processing
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `pypdf2`, `pdfplumber`, `pymupdf` | latest | PDF text/table extraction and document analysis |
| `python-docx` | 1.1.0 | Word document reading and generation |
| `markdown` | 3.5.1 | Markdown file parsing for document analysis module |
| `pillow`, `opencv-python` | latest | Image processing, manipulation, and format conversion |
| `invisible-watermark` | 0.2.0 | Watermark embedding for generated images |

#### 4.2.4 — Audio & Voice Processing
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `pyaudio`, `sounddevice` | latest | Real-time audio capture and playback for voice interfaces |
| `librosa`, `soundfile`, `torchaudio` | latest | Audio processing, feature extraction, and file I/O |
| `ffmpeg` | system latest | Audio/video transcoding required by Whisper and TTS models |
| `portaudio19-dev` | system latest | System-level audio driver for pyaudio |

#### 4.2.5 — Agentic AI & RAG
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `langchain`, `langchain-community` | latest | Agent framework, tool-calling, and chain orchestration |
| `chromadb` | latest | Vector database for RAG retrieval and memory persistence |
| `faiss-gpu-cu12` | latest | GPU-accelerated similarity search for embeddings |
| `sentence-transformers` | latest | Embedding models for semantic search and RAG pipelines |

#### 4.2.6 — Monitoring & Utilities
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `psutil`, `gputil` | latest | System and GPU resource monitoring |
| `plotly` | 5.18.0 | Interactive monitoring dashboards and data visualization |
| `pyyaml`, `python-dotenv` | latest | Configuration file parsing and environment variable management |
| `git-lfs` | system latest | Large file support for model downloads from Hugging Face |

### 4.3 — Pre-Configured Environment Variables
The following environment variables must be set automatically when the blueprint workspace launches. These were required in every analyzed project and students should never need to configure them manually.

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Prevents CUDA memory fragmentation |
| `CUDA_LAUNCH_BLOCKING` | `0` | Enables async CUDA for performance |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | HP AI Studio MLflow default |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Network-accessible Gradio binding |
| `GRADIO_SERVER_PORT` | `7860` | Standard Gradio port |
| `HF_HOME` | `/data/huggingface` | Centralized model cache location |
| `MODELS_DIR` | `/data/models` | Standardized model storage path |
| `TRANSFORMERS_CACHE` | `/data/huggingface/hub` | Transformer model cache |

### 4.4 — Pre-Configured Directory Structure
The blueprint workspace must initialize with the following directory structure. This layout was validated across all projects and provides clear separation of concerns.

| Path | Purpose |
| :--- | :--- |
| `workspace/` | Project root (contains all user notebooks and scripts) |
| `workspace/core/` | Core Python modules (analyzers, generators, managers) |
| `workspace/deployment/` | MLflow registration, Dockerfiles, deployment manifests |
| `workspace/tests/` | Test suites and validation scripts |
| `workspace/utilities/` | Helper scripts (CUDA test, model download, health check) |
| `workspace/config/` | YAML configs, environment files, custom prompts |
| `workspace/outputs/` | Generated images, documents, and artifacts |
| `workspace/logs/` | Application and monitoring logs |
| `data/models/` | Downloaded Hugging Face models (persistent across restarts) |
| `data/models/text-generation/` | LLMs (Mistral, Zephyr, Phi-2, etc.) |
| `data/models/text-to-image/` | Diffusion models (FLUX, SDXL, etc.) |
| `data/models/speech-to-text/` | ASR models (Whisper variants) |
| `data/models/text-to-speech/` | TTS models (XTTS-v2, Bark, etc.) |
| `data/models/embeddings/` | Embedding models for RAG and search |
| `data/datasets/` | Downloaded or custom datasets |

### 4.5 — Included Starter Notebooks
The blueprint must include the following pre-built notebooks in the `/workspace/` directory. These are derived directly from the production templates and project code tested across all analyzed implementations.

#### 4.5.1 — Project_Setup.ipynb (Required)
The primary entry point for all students. This notebook is derived from THE_AI_TEMPLATE and has been tested and refined across multiple student cohorts.

| Cell | Purpose | Function |
| :--- | :--- | :--- |
| 1 | **CUDA Configuration** | Sets environment variables; must run first before any other cell |
| 2 | **PyTorch Installation** | Installs torch+cu121 from official index; verifies GPU binding (2–5 min) |
| 3 | **Library Imports** | Imports torch, os, sys, platform; confirms core availability |
| 4 | **GPU Validation** | 4-step test: detection, memory alloc, matrix multiply, cleanup; pass/fail report |
| 5 | **AI Library Install** | Installs transformers, diffusers, accelerate, mlflow, gradio, datasets, etc. (3–7 min) |
| 6 | **Infrastructure Test** | Verifies all library imports; reports version numbers; 5/5 pass/fail summary |
| 7 | **Hugging Face Auth** | Guided token entry with step-by-step HF instructions; saves credentials |
| 8 | **Register_Model Gen** | Auto-creates `Register_Model.ipynb` with MLflow wrapper, signature, deployment code |
| 9 | **Setup Summary** | Comprehensive status report: GPU info, auth status, created files, next steps |
| 10 | **Quick Reference** | Code snippets for model loading, Gradio UI, dataset access, error handling |

#### 4.5.2 — Register_Model.ipynb (Auto-Generated)
Created automatically by Cell 8 of `Project_Setup.ipynb`. Contains MLflow model wrapper class, model signature definition, registration code, production stage transition, and deployment instructions for HP AI Studio Publishing Services.

#### 4.5.3 — Starter Project Notebooks (Recommended)
The following starter notebooks should be included as project templates students can modify. Each corresponds to a validated project type:
*   **Chatbot_Starter.ipynb:** Basic conversational AI using Gradio ChatInterface with streaming, system prompt support, and conversation memory. Derived from the Zephyr-7B and multi-modal chat implementations.
*   **Image_Gen_Starter.ipynb:** Text-to-image generation using FLUX.1-dev or SDXL-Turbo with Gradio gallery interface, seed control, and parameter sliders. Derived from the FLUX pipeline and multi-modal image gen projects.
*   **Document_Analyzer_Starter.ipynb:** PDF/Markdown upload and analysis with question-answering capability. Derived from the multi-modal document analysis module.
*   **Voice_Assistant_Starter.ipynb:** Speech-to-text input, command processing, and text-to-speech response using Whisper and XTTS-v2. Derived from the voice-controlled Trello project.

---

## 5. Blackwell GPU Optimization Requirements
All nine projects required Blackwell-specific GPU configuration. The blueprint must pre-configure these optimizations so students benefit from hardware acceleration without manual tuning.

### 5.1 — Memory Management Defaults
| Configuration | Specification |
| :--- | :--- |
| **Default Quantization** | 4-bit NF4 via BitsAndBytes for models under 13B parameters |
| **Large Model Strategy** | FP8 via second-generation Transformer Engine for 13B+ models |
| **Massive Model Strategy** | 4-bit GPTQ with vLLM PagedAttention for 70B+ parameter models |
| **Memory Allocation** | `gpu_memory_utilization=0.9` (reserves 10% for KV-cache overhead) |
| **OOM Recovery** | Automatic `torch.cuda.empty_cache()` on OutOfMemoryError with retry logic |
| **Context Management** | Dynamic truncation at 80% of max context window with system message preservation |
| **Batch Processing** | Queue-based concurrency limiting (`max_size=20`) to prevent memory overflow |

### 5.2 — Performance Monitoring
The blueprint should include a pre-configured monitoring utility (`utilities/gpu_monitor.py`) that tracks GPU utilization, memory usage, temperature, and tokens/second, logging to both MLflow and a local Plotly dashboard accessible on port 7861.

---

## 6. MLflow Integration & Deployment Pipeline
Every analyzed project follows the same three-stage deployment workflow. The blueprint must support this pipeline natively.

### 6.1 — Stage 1: Develop & Test
Students develop and test their models directly in Jupyter notebooks within the blueprint workspace. Gradio interfaces launch inline for immediate feedback. The blueprint provides the CUDA environment, libraries, and model access needed to iterate rapidly.

### 6.2 — Stage 2: Register with MLflow
Using the auto-generated `Register_Model.ipynb`, students register their model with MLflow’s model registry. The registration process includes defining a `ModelWrapper` class with a `predict` method, specifying input/output signatures, logging parameters and metrics, and transitioning the model to Production stage.

### 6.3 — Stage 3: Deploy via Publishing Services
HP AI Studio’s Publishing Services automatically create REST API endpoints from MLflow-registered models. The blueprint should include documentation and example code for testing these endpoints via `curl` commands and Swagger UI.

**Deployment Manifest Template**
The blueprint should include a `deployment_manifest.yaml` template pre-configured with compute requirements (Blackwell GPU, 48GB VRAM, 64GB RAM), endpoint definitions (Gradio on 7860, MLflow API on 5000), storage paths, and monitoring configuration. Students modify only the model-specific fields.

### 6.4 — Containerization Support
For advanced students, the blueprint should include a `Dockerfile` template derived from the production containers used across our projects. The base image should be `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` with all blueprint dependencies pre-installed, health checks configured, and ports 7860 and 5000 exposed.

---

## 7. Student Experience Requirements
The students using this blueprint are enrolled in an art and design program. Most have no prior experience with Python, command-line interfaces, or AI/ML concepts. The blueprint must accommodate this audience through the following design principles.

### 7.1 — First-Run Experience
| Milestone | Time Target | What Happens |
| :--- | :--- | :--- |
| **Launch** | 0 minutes | Student selects SCAD blueprint and clicks "Create Workspace" |
| **Ready** | 5 minutes | Workspace opens with all dependencies installed, GPU verified, MLflow running |
| **First Output** | 15 minutes | Student runs `Project_Setup.ipynb`, sees GPU test pass, and generates first AI output |
| **Custom Project** | 30 minutes | Student modifies a starter notebook to use their own prompts or data |
| **Deployment** | 60 minutes | Student registers model with MLflow and deploys via Publishing Services |

### 7.2 — Documentation Requirements
All notebooks must include:
*   Plain-language markdown headers explaining what each cell does and why
*   Expected output descriptions so students can verify success
*   Time estimates for long-running cells (installations, model downloads)
*   Troubleshooting guidance inline (not in a separate document)
*   "Next Steps" sections guiding students to their next action
*   Code comments on every non-obvious line, written for a reader who has never seen Python

### 7.3 — Error Handling & Recovery
Based on the most common student issues encountered during the Fall 2025 course, the blueprint must handle:

| Issue | Blueprint Behavior |
| :--- | :--- |
| **GPU not detected** | Gracefully fall back to CPU mode with clear warning; suggest driver check |
| **Out of memory (OOM)** | Auto-clear CUDA cache, reduce batch size, and retry; log to monitoring |
| **Hugging Face auth failure** | Provide step-by-step re-authentication instructions inline |
| **Model download interruption** | Resume download automatically via `resume_download=True` |
| **Library import failure** | Identify specific failed library and provide targeted reinstall command |
| **MLflow model not visible** | Guide student to refresh, check Production stage, verify MLflow UI |
| **Notebook kernel crash** | Provide kernel restart instructions and cell re-execution order |

---

## 8. Validated Model Registry
The following models have been tested and validated in production across our projects. The blueprint documentation should reference these as recommended starting points for each project type.

| Category | Model | Params | VRAM | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Text Generation** | `microsoft/phi-2` | 2.7B | 4GB | Code, light chat |
| **Text Generation** | `HuggingFaceH4/zephyr-7b-beta` | 7B | 6GB | Production chat |
| **Text Generation** | `mistralai/Mistral-7B-Instruct` | 7B | 6GB | Document analysis |
| **Text Generation** | `NousResearch/Nous-Hermes-2` | 7B | 6GB | Conversational AI |
| **Text Generation** | `perplexity-ai/r1-1776` | 671B | 335GB | Advanced reasoning |
| **Image Gen** | `black-forest-labs/FLUX.1-dev` | 12B | 24GB | Text-to-image |
| **Image Gen** | `stabilityai/sdxl-turbo` | 3.5B | 8GB | Fast image gen |
| **Speech-to-Text** | `openai/whisper-large-v3` | 1.5B | 4GB | Audio transcription |
| **Text-to-Speech** | `coqui/XTTS-v2` | 400M | 2GB | Voice synthesis |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | 335M | 1GB | RAG search |

---

## 9. Acceptance Testing Criteria
The following tests must pass before the blueprint is approved for classroom deployment. These are derived from the validation procedures used across all analyzed projects.

### 9.1 — Environment Tests
1. `torch.cuda.is_available()` returns `True`
2. `torch.cuda.get_device_name(0)` reports a Blackwell GPU
3. GPU memory allocation and matrix multiplication complete without error
4. All libraries in Section 4.2 import successfully with correct versions
5. MLflow server starts and is accessible at `http://localhost:5000`
6. Gradio demo launches and is accessible at `http://localhost:7860`

### 9.2 — Workflow Tests
7. `Project_Setup.ipynb` runs all 10 cells sequentially without errors
8. `Register_Model.ipynb` is auto-generated and runs successfully
9. A Hugging Face model downloads to `data/models` and loads on GPU
10. A Gradio `ChatInterface` with streaming generates text responses
11. An image generation pipeline produces a 1024x1024 image from a text prompt
12. MLflow model registration completes and model appears in Production stage
13. HP AI Studio Publishing Services creates a working REST endpoint

### 9.3 — Student Experience Tests
14. A user with no Python experience can run `Project_Setup.ipynb` by following written instructions only
15. A user can modify a starter notebook prompt and see different output within 5 minutes
16. Error messages in notebooks are understandable without technical background
17. All time estimates in notebook markdown cells are accurate to within 50%

---

## Appendix: Project Source References
The following conversations and documents were analyzed to produce this specification. Each link leads to the original project development conversation or documentation.

**A.1 — Primary Development Conversations**
*   Three-Model LLM Deployment (Phi-2, Zephyr-7B, CPU-light)
*   R1-1776 671B Parameter Model Deployment with Gradio
*   Multi-Modal AI System (7-phase development pipeline)
*   Multi-Modal Image Generation with FLUX.1-dev and ControlNet
*   FLUX.1-dev Text-to-Image Diffusion Application
*   Voice-Controlled Trello Assistant (Whisper + XTTS-v2)
*   Agentic Web Intelligence Assistant (LangChain + RAG)
*   Gradio Chatbot MLflow Deployment with RegisterModel Generation
*   Student Template Notebook Optimization and Debugging

**A.2 — Documented Templates & Guides**
*   `THE_AI_TEMPLATE` (hpaistudiotemplate.py)
*   `Student Quick Start Guide` (studentquickstart.md) (20-minute onboarding)
*   `Template Teaching Guide` (templateguide.md) (Instructor reference)
*   `Hugging Face Model Download Guide` (huggingfacemodeldownloadguide.md)
*   `Multi-Modal Final Documentation` (multimodalfinaldocs.md)
*   `Deployment README` (deploymentreadme.md) (Phase-by-phase deployment steps)
*   `Complete Project Archive` (CompleteProjectArchive.md) (Full course history)

**End of Specification**
For questions or clarification, contact Dan Bartlett (dbartlett@scad.edu)
SCAD Applied AI Design & Development 2025–2026
