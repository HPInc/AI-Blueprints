# 🤖 Agentic GitHub Repo Analyzer with LangGraph

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

The **Agentic GitHub Repo Analyzer** is a production-grade LangGraph-powered blueprint designed to help users ingest, reason over, and synthesize insights from a Github repository.

It integrates:

* 📄 Multi-format document ingestion (TXT, PDF, DOCX, XLSX, CSV, Markdown, PY, JSON, YML, YAML, IPYNB)
* 🧠 Per-chunk reasoning using LLMs orchestrated via LangGraph
* 🧬 Chunk grouping to respect token limits during synthesis
* 📊 Final markdown-formatted, non-redundant answer synthesis
* 🧪 MLflow model packaging and deployment
* 🌐 Streamlit-based UI for interactive inference

---

## 📁 Project Structure

```bash
agentic-github-repo-analyzer-with-langgraph/
├── data/                                # Github repo files (input directory)
│   └── input/
├── demo/                                # UI frontend code (Streamlit)
│   └── streamlit/
├── docs/                                # Documents for UI Testing
│   ├── diagram-for-agentic-github-repo-analyzer-with-langgraph.png
│   ├── Streamlit UI Page - Agentic Github Repo Analyzer.pdf
│   └── streamlit-ui-ss-agentic-github-repo-analyzer.png
├── notebooks/                           # Workflow and MLflow notebooks
│   ├── register-model.ipynb
│   └── run-workflow.ipynb
├── requirements.txt                     # All required packages
├── README.md                            # Project documentation
└── src/                                 # Core LangGraph modules
    ├── __init__.py
    ├── agentic_feedback_model.py        # MLflow PyFunc model class
    ├── agentic_nodes.py                 # LangGraph nodes
    ├── agentic_state.py                 # Shared state schema
    ├── agentic_workflow.py              # LangGraph DAG construction
    ├── simple_kv_memory.py              # Disk-based memory module
    └── utils.py                         # Helper functions
```

---

## ⚙️ Setup

### Step 0: Minimum Hardware Requirements

* ✅ **GPU**: NVIDIA GPU with 8 GB VRAM
* ✅ **RAM**: 32–64 GB system memory
* ✅ **Disk**: ≥ 10 GB free space

### Step 1: Create an AI Studio Project

1. Go to [HP AI Studio](https://hp.com/ai-studio) and create a new project.
2. Use the base image: `Local GenAI`

### Step 2: Add Required Assets

- Download the Meta Llama 3.1 model with 8B parameters via Models tab:

  - **Model Name**: `meta-llama3.1-8b-Q8`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
  - **Resource Type**: `public`
  - **Bucket Region**: `us-west-2`

- Make sure that the model is in the `datafabric` folder inside your workspace. If the model does not appear after downloading, please restart your workspace.

---

## 🚀 Usage

### 🧪 Step 1: Run LangGraph Workflow

Use the provided notebook to run the end-to-end pipeline:

```bash
notebooks/run-workflow.ipynb
```

This notebook will:

* Ingest GitHub repo files
* Rewrite user questions
* Generate per-chunk responses
* Synthesize final answer via chunk-group aggregation

### 🧠 Step 2: Register Model with MLflow

Log and serve the full pipeline as an MLflow `pyfunc` model:

```bash
notebooks/register-model.ipynb
```

This registers the model so it can be queried over HTTP.

### Step 3: Launch the Streamlit Web App

1. After completing the local deployment, open the Streamlit web app using the deployment URL provided by AI Studio.
2. For additional details on how the Streamlit app works, refer to the `README.md` file in the `demo/streamlit` folder.

<!-- ### Streamlit Preview

![Streamlit Screenshot](docs/streamlit-ss.png) -->

---

## 📞 Contact & Support

  - **Troubleshooting:** Refer to the [**Troubleshooting**](https://github.com/HPInc/AI-Blueprints/tree/main?tab=readme-ov-file#troubleshooting) section of the main README in our public AI-Blueprints GitHub repo for solutions to common issues.

  - **Issues & Bugs:** Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

  - **Docs:** [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

  - **Community:** Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio)
