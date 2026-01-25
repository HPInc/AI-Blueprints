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

- [🧠 Overview](#🧠-overview)
- [📁 Project Structure](#📁-project-structure)
- [⚙️ Setup](#⚙️-setup)
- [🚀 Usage](#🚀-usage)
- [📞 Contact & Support](#📞-contact--support)

---

## 🧠 Overview

The **Agentic Audio RAG** blueprint turns speech in audio/video files into **searchable knowledge** and lets you ask questions directly about the **actual audio** (not just text). A LangGraph-driven agent retrieves the most relevant **timestamped audio segments**, and an audio-native LLM (Qwen Omni) “listens” to those clips to produce precise answers.

It delivers:

- 🎧 **Audio-native LLM QA** — the model consumes selected audio windows directly for reasoning (supports MP3, WAV, OGG, FLAC, and audio tracks from MP4, MOV, MKV, AVI, …).
- 🔊 **Audio embedding with CLAP** — builds a segment-level vector index over audio; retrieve by embedding the user’s text query into the **same audio↔text space**.
- 🧪 **Agentic RAG orchestration via LangGraph** — retrieval → (optional rerank) → generation → memory, all modular and node-based.
- 🦙 **Llama.cpp** for fast, local text LLM utilities (e.g., lightweight reranking/scoring or text-only reasoning when needed).
- 📚 **Audio-aware vector database (FAISS)** — stores CLAP embeddings for efficient semantic search over timestamped segments.
- 🧬 **Reranking stage** to sharpen selection (MMR diversification and/or lightweight LLM scoring).
- 🕒 **Evidence with timestamps** — answers highlight the exact audio spans (start/end seconds) used to support the response.
- 💾 **Disk-backed memory cache** — stores recent Q&A pairs to accelerate repeat queries.
- 📦 **MLflow integration** — experiment tracking and model packaging aligned with the agentic-feedback-analyzer blueprint.
- 🌐 **Streamlit UI** — upload media, run queries, and inspect highlighted evidence.

---

## 📁 Project Structure

```bash
agentic-audio-rag-with-langgraph/
├── configs/                             # Configuration files
│   └── config.yaml                      # Blueprint configuration (UI mode, ports, service settings)
├── data/                                # Runtime data (optional)
│   ├── memory/                          # Q&A cache for faster repeat queries
│   └── temp/                            # Temporary uploads (auto-cleaned)
├── demo/                                # UI frontend code (Streamlit)
│   └── streamlit/
│       ├── main.py                      # Modern UI with file upload
│       ├── assets/                      # CSS styling
│       └── static/                      # Logo images
├── docs/                                # UI documentation & screenshots
│   ├── Streamlit UI Page - Agentic Audio RAG.pdf
│   └── streamlit-ui-ss-agentic-audio-rag.png
├── notebooks/                           # Workflow and MLflow notebooks
│   ├── register-model.ipynb             # Register model with dynamic audio processing
│   └── run-workflow.ipynb               # Test workflow locally
├── src/                                 # Core LangGraph modules & MLflow integration
│   ├── __init__.py
│   ├── agentic_workflow.py              # LangGraph DAG construction
│   ├── segment_audio_embeddings.py      # Audio processing and CLAP embeddings
│   ├── simple_kv_memory.py              # Disk-based memory module
│   ├── utils.py                         # Helper functions
│   └── mlflow/                          # Universal MLflow structure
│       ├── __init__.py                  # Dynamic imports for Model and Logger
│       ├── model.py                     # Business logic layer (audio processing)
│       ├── loader.py                    # MLflow loader
│       └── logger.py                    # MLflow logger service
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

- ✅ **GPU**: NVIDIA GPU with 32 GB+ VRAM (recommended for LLM acceleration)
- ✅ **RAM**: 32–64 GB system memory
- ✅ **Disk**: ≥ 10 GB free space

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

### Step 4: Setup Configuration and Models

#### Model Storage Options

This blueprint supports **two model storage approaches**:

**Option A: Local Datafabric Storage (Recommended)**

For optimal performance and to avoid download timeouts, store models locally:

**Required Models:**

1. **Qwen2.5-Omni-7B** (~14GB) - Audio/video multimodal reasoning model
2. **clap-htsat-unfused** (~300MB) - Audio embedding model for retrieval

**Storage Locations:**

```
/home/jovyan/datafabric/
├── Qwen2.5-Omni-7B/                    # Main audio reasoning model
├── clap-htsat-unfused/                 # Audio embedding model
```

**Manual Download Instructions:**

```bash
# Using HuggingFace CLI (requires authentication)
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /home/jovyan/datafabric/Qwen2.5-Omni-7B
huggingface-cli download laion/clap-htsat-unfused --local-dir /home/jovyan/datafabric/clap-htsat-unfused
```

#### Configuration Settings

- Edit `config.yaml` with relevant configuration details:
  - `model_source`: Set to `local` for datafabric storage or `hugging-face-cloud` for remote download
  - `qwen_model_path`: Path to local Qwen model (if using local storage)
  - `clap_model_path`: Path to local CLAP model (if using local storage)
  - `ui.mode`: Set UI mode to `streamlit` or `static`
  - `ports`: Configure external and internal port mappings
  - `service`: Adjust MLflow timeout and health check settings
  - `proxy`: Set proxy settings if needed for restricted networks

---

## 🚀 Usage

### 🧪 Step 1: Run LangGraph Workflow (Optional)

Use the provided notebook to test the workflow locally:

```bash
notebooks/run-workflow.ipynb
```

This notebook will:

- Process audio/video files dynamically
- Build audio embedding index over 30-second segments using CLAP (audio↔text joint space)
- Run the agentic retrieval-and-rerank workflow, sending the top audio windows to Qwen Omni
- Show the generated answers together with timestamped evidence segments

**Note:** This step is optional for development/testing. In production, the Streamlit UI handles all processing automatically.

### 🧠 Step 2: Register Model with MLflow

Log and serve the full pipeline as an MLflow `pyfunc` model:

```bash
notebooks/register-model.ipynb
```

This notebook will:

- Package the **Agentic Audio RAG** workflow (CLAP embeddings, Qwen Omni, LangGraph DAG, memory module) as a single MLflow artifact
- Configure the model for **dynamic audio processing**
- Register the model to MLflow so it can be queried over HTTP with uploaded audio files

### 📦 Step 3: Deploy the Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Streamlit UI** via the Service URL.
- The service will automatically use the configuration logged as an artifact during model registration.

### 🌐 Step 4: Use the Streamlit UI

Once deployed, access the Streamlit UI through the Service URL. The interface allows you to:

**Upload & Process:**

1. **Upload any audio or video file** (MP3, WAV, MP4, MOV, etc.)
2. Watch the **real-time progress bar** as the system:
   - Converts audio to WAV format (ffmpeg)
   - Segments into 30-second windows
   - Generates CLAP embeddings
   - Builds the FAISS vector index
3. File processing takes ~30 seconds for a 5-minute audio file

**Ask Questions:**

1. Once processed, type your question in natural language
2. The system will:
   - Search the audio index using CLAP embeddings
   - Retrieve the most relevant 6 segments
   - Let Qwen Omni "listen" to those audio clips
   - Generate an answer based on what it heard
3. View the **timestamped evidence** showing exactly which audio segments were used

**Benefits:**

- 🔄 **Memory Cache**: Repeated questions return instantly
- 📊 **Conversation History**: Track all Q&A pairs in the session
- 🎯 **Precise Timestamps**: Jump directly to relevant audio moments

**API Usage Example:**

```python
import requests

# First request: Upload and process audio
payload = {
    "inputs": [{
        "audio_path": "/path/to/meeting.mp3",
        "question": "What were the main action items?",
        "file_id": "meeting.mp3"
    }]
}

response = requests.post(
    "http://localhost:5002/invocations",
    json=payload
)

result = response.json()["predictions"][0]
print(f"Answer: {result['answer']}")
print(f"Evidence: {result['evidence']}")

# Subsequent requests: Just use file_id (already processed)
payload = {
    "inputs": [{
        "question": "Who was assigned to each task?",
        "file_id": "meeting.mp3"  # No audio_path needed
    }]
}
```

---

## 📞 Contact & Support

- **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

- **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio)
