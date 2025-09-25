# 🤖 Agentic Audio RAG with LangGraph

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Model_Deployment-orange.svg?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend_App-ff4b4b.svg?logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-blue.svg?logo=langchain)
![LangChain](https://img.shields.io/badge/LangChain-LLM_Orchestration-lightgreen.svg?logo=langchain)

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

The **Agentic Audio RAG** blueprint turns speech in audio/video files into **searchable knowledge** and lets you ask questions directly about the **actual audio** (not just text). A LangGraph-driven agent retrieves the most relevant **timestamped audio segments**, and an audio-native LLM (Qwen Omni) “listens” to those clips to produce precise answers.

It delivers:

* 🎧 **Audio-native LLM QA** — the model consumes selected audio windows directly for reasoning (supports MP3, WAV, OGG, FLAC, and audio tracks from MP4, MOV, MKV, AVI, …).
* 🔊 **Audio embedding with CLAP** — builds a segment-level vector index over audio; retrieve by embedding the user’s text query into the **same audio↔text space**.
* 🧪 **Agentic RAG orchestration via LangGraph** — retrieval → (optional rerank) → generation → memory, all modular and node-based.
* 🦙 **Llama.cpp** for fast, local text LLM utilities (e.g., lightweight reranking/scoring or text-only reasoning when needed).
* 📚 **Audio-aware vector database (FAISS)** — stores CLAP embeddings for efficient semantic search over timestamped segments.
* 🧬 **Reranking stage** to sharpen selection (MMR diversification and/or lightweight LLM scoring).
* 🕒 **Evidence with timestamps** — answers highlight the exact audio spans (start/end seconds) used to support the response.
* 💾 **Disk-backed memory cache** — stores recent Q&A pairs to accelerate repeat queries.
* 📦 **MLflow integration** — experiment tracking and model packaging aligned with the agentic-feedback-analyzer blueprint.
* 🌐 **Streamlit UI** — upload media, run queries, and inspect highlighted evidence.

---

## 📁 Project Structure

```bash
agentic-audio-rag-with-langgraph/
├── configs/                             # Configuration files
│   └── config.yaml                      # Blueprint configuration (UI mode, ports, service settings)
├── data/                                # Sample media files (input directory)
│   └── inputs/                          #  └─ *.mp3 / *.wav / *.mp4 …
├── demo/                                # UI frontend code (Streamlit)
│   └── streamlit/
├── docs/                                # UI documentation & screenshots
│   ├── Streamlit UI Page - Agentic Audio RAG.pdf
│   └── streamlit-ui-ss-agentic-audio-rag.png
├── notebooks/                           # Workflow and MLflow notebooks
│   ├── register-model.ipynb
│   └── run-workflow.ipynb
├── src/                                 # Core LangGraph modules
|   ├── __init__.py
|   ├── audio_rag_model.py               # MLflow PyFunc model class
|   ├── audio_rag_nodes.py               # LangGraph nodes
|   ├── agentic_state.py                 # Shared state schema
|   ├── agentic_workflow.py              # LangGraph DAG construction
|   ├── simple_kv_memory.py              # Disk-based memory module
|   └── utils.py                         # Helper functions
├── requirements.txt                     # All required packages
└── README.md                            # Project documentation

```

---

## Configuration

The blueprint uses a centralized configuration system through `configs/config.yaml`:

```yaml
ui:
  mode: streamlit # UI mode: streamlit or static
  ports:
    external: 8501 # External port for UI access
    internal: 8501 # Internal container port
  service:
    timeout: 30 # Service timeout in seconds
    health_check_interval: 5 # Health check interval in seconds
    max_retries: 3 # Maximum retry attempts
```

---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements

* ✅ **GPU**: NVIDIA GPU with 12 GB+ VRAM (recommended for LLM acceleration)
* ✅ **RAM**: 32–64 GB system memory
* ✅ **Disk**: ≥ 10 GB free space

### Step 1: Create an AI Studio Project

1. Go to [HP AI Studio](https://hp.com/ai-studio) and create a new project.
2. Use the base image: `Local GenAI`

### Step 2: Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

## Step 3: Configure Secrets

- **Configure Secrets in YAML file (Freemium users):**
  - Create a `secrets.yaml` file in the `configs` folder and list your API keys there:
    - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.

- **Configure Secrets in Secrets Manager (Premium users):**
  - Add your API keys to the project's Secrets Manager vault, located in the `Project Setup` tab -> `Setup` -> `Project Secrets`:
    - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.
  - In `Secrets Name` field add: `HUGGINGFACE_API_KEY`
  - In the `Secret Value` field, paste your corresponding key generated by HuggingFace.

  <br>

  **Note: If both options (YAML option and Secrets Manager) are used, the Secrets Manager option will override the YAML option.**

### Step 4: Setup Configuration

- Edit `config.yaml` with relevant configuration details:
  - `model_source`: Choose between `local`, `hugging-face-cloud`, or `hugging-face-local`
  - `ui.mode`: Set UI mode to `streamlit` or `static`
  - `ports`: Configure external and internal port mappings
  - `service`: Adjust MLflow timeout and health check settings
  - `proxy`: Set proxy settings if needed for restricted networks

---

## 🚀 Usage

### 🧪 Step 1: Run LangGraph Workflow

Use the provided notebook to run the end-to-end pipeline:

```bash
notebooks/run-workflow.ipynb
```

This notebook will:

* Scan data/inputs for audio/video, normalize audio, and segment into timestamped windows
* Build a true audio embedding index over segments using CLAP (audio↔text joint space)
* Run the agentic retrieval-and-rerank workflow, sending the top audio windows to the model to listen and answer directly
* Show the generated answers together with the highlighted transcript segments and timestamps

### 🧠 Step 2: Register Model with MLflow

Log and serve the full pipeline as an MLflow `pyfunc` model:

```bash
notebooks/register-model.ipynb
```
This notebook will:

* Packages the complete **Agentic Audio RAG** workflow (vector store, reranker, LangGraph DAG, memory module) as a single MLflow artifact
* Registers the model to MLflow so it can be queried over HTTP

### 📦 Step 3: Deploy the Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Streamlit UI** via the Service URL.
- The service will automatically use the configuration logged as an artifact during model registration.

### 🌐 Step 4: Launch Streamlit UI

This web UI will allow the user to:

* Upload one or more audio / video files (or pick the samples in `data/inputs/`)
* Ask questions about their content
* See the **highlighted transcript segments** (with timestamps) that the model used to craft each answer
* Benefit from the built-in memory: repeated queries return quickly after the first run
* Connect to a local MLflow model endpoint

---

## 📞 Contact & Support

  - **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

  - **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

  - **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

  - **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio)
