# Code Generation RAG with Langchain

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![LangChain](https://img.shields.io/badge/LangChain-used-lightgreen.svg?logo=langchain)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Configuration](#configuration)
- [🔧 Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and integrates with evaluation and tracking frameworks. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.

---

## Project Structure

```text
├── configs
│   └── config.yaml                                                    # Blueprint configuration (UI mode, ports, service settings)
├── docs
│   └── successful-swagger-ui-code-generation-result.pdf               # Streamlit UI page
├── notebooks
│   ├── core/                                                          # Core modules within notebooks
│   │   ├── chroma_embedding_adapter.py
│   │   ├── code_generation_service.py
│   │   ├── dataflow/
│   │   ├── extract_text/
│   │   ├── generate_metadata/
│   │   ├── prompt_templates.py
│   │   └── vector_database/
│   ├── register-model.ipynb                                           # Model registration notebook
│   └── run-workflow.ipynb                                             # Main workflow notebook
├── src
│   ├── service/
│   │   ├── __init__.py
│   │   └── base_service.py
│   ├── __init__.py
│   ├── trt_llm_langchain.py
│   └── utils.py                                                       # Utility functions for config loading
├── README.md
└── requirements.txt
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

## Setup

### Step 0: Minimum Hardware Requirements

To ensure smooth execution, make sure your system meets the following minimum hardware specifications:

- RAM: 32 GB
- VRAM: 6 GB
- GPU: NVIDIA GPU

### Quickstart

### Step 1: Create an AIstudio Project

1. Create a **New Project** in AI Studio
2. Select the template Text Generation with Langchain
3. Add a title description and relevant tags.

### Step 2: Verify Project Files

1. Launch a workspace.
2. Navigate to `code-generation-with-langchain/` to ensure all files were cloned correctly.

## Alternative Manual Setup

### Step 1: Create an AIStudio Project

1. Create a **New Project** in AI Studio.
2. (Optional) Add a description and relevant tags.

### Step 2: Create a Workspace

1. Choose **Local GenAI** as the base image when creating the workspace.

### Step 3: Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation..

### Step 4: Download the Llama‑3 Model

1. Download the Meta Llama 3.1 model with 8B parameters via Models tab:

- **Model Name**: `meta-llama3.1-8b-Q8`
- **Model Source**: `AWS S3`
- **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`
- **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace.

2. The model will be available under the /datafabric directory in your workspace.

### Step 5: Configure Secrets

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

### Step 7: Setup Configuration

1. Edit `config.yaml` with relevant configuration details:
  - `model_source`: Choose between `local`, `hugging-face-cloud`, or `hugging-face-local`
  - `ui.mode`: Set UI mode to `streamlit` or `static`
  - `ports`: Configure external and internal port mappings
  - `service`: Adjust MLflow timeout and health check settings
  - `proxy`: Set proxy settings if needed for restricted networks


---

## Usage

### Step 1: Run the Workflow Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/run-workflow.ipynb
```

This will:
- Run the full RAG pipeline

### Step 2: Run the Register Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/register-model.ipynb
```

This will:
- Register the model in MLflow

### Step 3: Deploy the Chatbot Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).