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

* [🧠 Overview](#-overview)
* [🎯 Learning Objectives](#-learning-objectives)
* [📖 Prerequisites & Background Reading](#-prerequisites--background-reading)
* [📁 Project Structure](#-project-structure)
* [⚙️ Setup](#️-setup)
* [🚀 Usage](#-usage)
* [📞 Contact & Support](#-contact--support)

---

## 🧠 Overview

The **Educational Quickstart** is a pre-configured, multi-capability AI development environment designed for learning AI and machine learning concepts hands-on. No prior AI experience is required.

It provides:

* 🤖 **Chatbot Starter** — Conversational AI using LLMs with streaming, system prompt support, and conversation memory
* 🎨 **Image Generation Starter** — Text-to-image generation using diffusion models with parameter controls
* 📄 **Document Analyzer Starter** — PDF/Markdown upload and analysis with question-answering capability
* 🎙️ **Voice Assistant Starter** — Speech-to-text input, command processing, and text-to-speech response
* 📊 **Interactive GPU Monitoring** — Real-time Plotly dashboards for GPU utilization, memory, and performance tracking
* 📦 **MLflow Deployment** — Full model packaging, registration, and REST API deployment pipeline
* 🌐 **Streamlit UI** — Interactive web interface for deployed model inference

This blueprint eliminates the 2–4 hour manual environment setup, enabling you to produce your first AI output within 20 minutes.

---

## 🎯 What You Will Build

By working through this blueprint, you will:

1. **Understand what Large Language Models (LLMs) are** and how text generation works at a high level
2. **Run a real AI model on GPU hardware** and observe the effect of GPU acceleration on inference speed
3. **Generate images from text prompts** using diffusion model pipelines
4. **Build a document Q&A system** using Retrieval-Augmented Generation (RAG) concepts
5. **Transcribe speech to text** with OpenAI Whisper and combine it with an LLM for a voice assistant
6. **Package and deploy an AI model** using MLflow so it can be called as a REST API
7. **Build an interactive web UI** using Streamlit to interact with your deployed model

---

## 📖 Prerequisites & Background Reading

You don't need any of these resources before starting — the notebooks are self-contained. But if you want to go deeper on any topic, these free resources are excellent:

| Topic | Resource |
|-------|----------|
| Python basics | [Python Official Tutorial](https://docs.python.org/3/tutorial/) |
| Intro to AI/ML | [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) |
| Understanding LLMs | [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) |
| Diffusion models | [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-models-course/unit1/introduction) |
| Whisper (speech AI) | [OpenAI Whisper on GitHub](https://github.com/openai/whisper) |
| PyTorch (GPU framework) | [PyTorch Tutorials](https://pytorch.org/tutorials/) |
| MLflow (model tracking) | [MLflow Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html) |
| Streamlit (web apps) | [Streamlit Documentation](https://docs.streamlit.io/) |
| CUDA (GPU programming) | [NVIDIA CUDA Overview](https://developer.nvidia.com/cuda-toolkit) |

---

## 📁 Project Structure

```bash
educational-quickstart/
├── configs/                                 # Per-capability configuration files
│   ├── chatbot.yaml                         # capability: chatbot — LLM conversational AI
│   ├── document.yaml                        # capability: document — RAG document Q&A
│   ├── image_gen.yaml                       # capability: image_gen — SDXL-Turbo
│   └── voice.yaml                           # capability: voice — Whisper + LLM
├── data/                                    # Sample input data
│   └── input/
│       ├── sample_feedback.txt              # Sample text for document analyzer
│       └── sample_prompts.txt               # Sample prompts for chatbot/image gen
├── demo/                                    # Per-capability Streamlit UIs
│   ├── chatbot/                             # Conversational Q&A frontend
│   │   ├── assets/styles.css
│   │   ├── static/                          # HP, Z, AIS logos
│   │   ├── main.py
│   │   ├── pyproject.toml
│   │   └── README.md
│   ├── document/                            # Document analyzer frontend
│   │   └── ...                             # same structure as chatbot/
│   ├── image_gen/                           # Image generation frontend
│   │   └── ...
│   └── voice/                              # Voice assistant frontend
│       └── ...
├── docs/                                    # Screenshots and architecture diagrams
│   ├── streamlit-ss.png
│   └── architecture-diagram.png
├── notebooks/                               # One-click starter notebooks
│   ├── chatbot-starter.ipynb                # Conversational AI: setup → demo → register
│   ├── image-gen-starter.ipynb              # Image generation: setup → demo → register
│   ├── document-analyzer-starter.ipynb      # Document analysis: setup → demo → register
│   └── voice-assistant-starter.ipynb        # Voice assistant: setup → demo → register
├── requirements.txt                         # All required packages
├── README.md                                # Project documentation
└── src/                                     # Core Python modules
    ├── __init__.py
    ├── utils.py                             # Shared utilities (load_config, log_asset_status)
    ├── gpu_monitor.py                       # Plotly GPU monitoring
    └── mlflow/                              # MLflow 3-layer architecture
        ├── __init__.py                      # Lazy-loading: ChatbotModel, ImageGenModel, …
        ├── loader.py                        # Config-driven loader (_load_pyfunc)
        ├── logger.py                        # MLflow registration (Logger.log_model)
        └── models/                          # Per-capability model classes
            ├── __init__.py                  # MODEL_REGISTRY dict
            ├── chatbot.py                   # ChatbotModel — LlamaCpp + system prompt
            ├── document.py                  # DocumentModel — chunk-based RAG
            ├── image_gen.py                 # ImageGenModel — SDXL-Turbo pipeline
            └── voice.py                     # VoiceModel — Whisper STT + LlamaCpp
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
- You can create or find your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> ⚠️ **FLUX.1-dev is a gated model — license acceptance required before downloading.**
> Visit **[huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**,
> click **"Access repository"**, and accept the license agreement.
> Then complete the Hugging Face auth step in `00-project-setup.ipynb` (Cell 8) before running the
> Model Download cell (Cell 9).
> Skipping this step causes only the FLUX download to fail — all other models (Zephyr, Llama 3.1, Whisper, XTTS v2) download without any license gate.

### Step 4: Configuration

Each capability has its own config file. Review the relevant file if you need to change model paths:

| Config file | Capability | Key field |
|---|---|---|
| `configs/chatbot.yaml` | Conversational AI | `model_path` |
| `configs/image_gen.yaml` | Image generation | `image_model_path` |
| `configs/document.yaml` | Document Q&A | `model_path` |
| `configs/voice.yaml` | Voice assistant | `model_path`, `stt_model_path` |

---

## 🚀 Usage

### One-Click Workflow

Each starter notebook is completely self-contained — open any one and **Run All Cells**:

| Notebook | Capability | Registers As |
|----------|-----------|--------------|
| `chatbot-starter.ipynb` | Conversational AI with LLM | `AIStudio-EQ-Chatbot` |
| `image-gen-starter.ipynb` | Text-to-image (SDXL-Turbo) | `AIStudio-EQ-ImageGen` |
| `document-analyzer-starter.ipynb` | Document RAG Q&A | `AIStudio-EQ-Document` |
| `voice-assistant-starter.ipynb` | Whisper STT + LLM | `AIStudio-EQ-Voice` |

Each notebook performs these steps automatically:
1. Install dependencies (`requirements.txt`)
2. Verify GPU, load config, check for model files
3. Initialize the capability-specific model class
4. Run an interactive demo with Plotly charts
5. Register the model to MLflow and verify the deployment

### 🌐 Launch the Streamlit Web App

After a notebook completes registration, deploy the matching UI from its `demo/` folder:

| Registered Model | Demo UI |
|---|---|
| `AIStudio-EQ-Chatbot` | `demo/chatbot/` |
| `AIStudio-EQ-ImageGen` | `demo/image_gen/` |
| `AIStudio-EQ-Document` | `demo/document/` |
| `AIStudio-EQ-Voice` | `demo/voice/` |

Open the deployment URL provided by AI Studio to launch the Streamlit app.
For Streamlit app details, refer to the `README.md` inside each demo folder.

---

## 📞 Contact & Support

  - **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

  - **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

  - **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

  - **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio)
