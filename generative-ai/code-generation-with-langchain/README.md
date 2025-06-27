# Code Generation RAG with Langchain and Galileo

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and integrates with PromptQuality and Galileo for evaluation and tracking. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.

---

## Project Structure

```
├── README.md                                       # Project documentation
├── notebooks
│   └── code-generation-with-langchain.ipynb        # Main notebook for the project
├── core                                            # Core Python modules
│   ├── dataflow
│   │   └── dataflow.py
│   ├── extract_text
│   │   └── github_repository_extractor.py
│   ├── generate_metadata
│   │   └── llm_context_updater.py
│   ├── vector_database
│   │   └── vector_store_writer.py
│   └── code_generation_service.py
├── configs                                         # Configuration files
│   ├── config.yaml
│   └── secrets.yaml


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
2. Navigate to `02-code-generation-with-langchain/notebooks/code-generation-with-langchain.ipynb` to ensure all files were cloned correctly.

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

### Step 4: Log the Llama‑3 Model

1. Download the Meta Llama 3.1 model with 8B parameters via Models tab:

- **Model Name**: `meta-llama3.1-8b-Q8`
- **Model Source**: `AWS S3`
- **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
- **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace.

2. The model will be available under the /datafabric directory in your workspace.

### Step 5: Configure Secrets and Paths

1. Add your API keys to the `secrets.yaml` file under the `configs` folder:

- `HUGGINGFACE_API_KEY`
- `GALILEO_API_KEY`
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/code-generation-with-langchain.ipynb
```

This will:

- Run the full RAG pipeline
- Integrate Galileo evaluation, protection, and observability
- Register the model in MLflow

### Step 2: Deploy the Chatbot Service

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
