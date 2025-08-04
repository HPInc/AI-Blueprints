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

The **Agentic Audio RAG** blueprint converts speech in audio/video files into searchable knowledge and information, allowing the user to query that information based on the generated transcripts through a LangGraph-driven Retrieval Augmented Generation pipeline.

It delivers:

* 🎙️ Automatic speech-to-text using OpenAI Whisper large-v3 (supports MP3, WAV, OGG, MP4, MOV, AVI …)
* 🧪 Agentic RAG workflow orchestrated with LangGraph
* 🦙 Llama.cpp for fast on-device LLM inference
* 🧬 Reranking stage to improve passage selection accuracy
* 🔍 Answer generation that also returns the exact transcript snippets and their timestamps
* 💾 Lightweight memory to cache previous Q&A pairs
* 📦 MLflow model packaging & deployment
* 🌐 Streamlit UI for uploading media, running queries and inspecting highlighted transcript segments

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

### Step 3: Add Required Assets

- Download the Meta Llama 3.1 model with 8B parameters via Models tab:

  - **Model Name**: `meta-llama3.1-8b-Q8`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
  - **Resource Type**: `public`
  - **Bucket Region**: `us-west-2`

- Make sure that the model is in the `datafabric` folder inside your workspace. If the model does not appear after downloading, please restart your workspace.

## Step 4: Configure Secrets

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

### Step 5: Setup Configuration

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

* Extract transcripts from the sample audio / video files in `data/inputs`
* Chunk the transcript and build a vector store index
* Run the agentic retrieval-and-rerank workflow on a few demo queries
* Show the generated answers together with the highlighted transcript segments and timestamps

### 🧠 Step 2: Register Model with MLflow

Log and serve the full pipeline as an MLflow `pyfunc` model:

```bash
notebooks/register-model.ipynb
```
This notebook will:

* Packages the complete **Agentic Audio RAG** workflow (Whisper encoder, vector store, reranker, LangGraph DAG, Llama.cpp generator, memory module) as a single MLflow artifact
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
