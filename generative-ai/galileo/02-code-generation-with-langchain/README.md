# Code Generation RAG with Langchain and Galileo

## 📚 Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contact & Support](#contact--support)

---

## 🧠 Overview

This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and integrates with PromptQuality and Galileo for evaluation and tracking. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.

---

```
├── README.md
├── notebooks
│   └── code-generation-with-langchain.ipynb
├── core
│   ├── dataflow
│   │   └── dataflow.py
│   ├── extract_text
│   │   └── github_repository_extractor.py
│   ├── generate_metadata
│   │   └── llm_context_updater.py
│   ├── vector_database
│   │   └── vector_store_writer.py
│   └── code_generation_service.py
├── configs
│   ├── config.yaml
│   └── secrets.yaml
└── requirements.txt


```

---

## Setup

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

### Step 3: Log Model

1. Download the **LLaMA2-7B** model from AWS S3 using the Models tab in your AI Studio project:
  - **Dataset Name**: `llama2-7b`
  - **Dataset Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/llama2-7b`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace.

2. The model will be available under the /datafabric directory in your workspace.

### Step 4:  Configure Secrets and Paths
1. Add your API keys to the `secrets.yaml` file under the `configs` folder:
  - `HUGGINGFACE_API_KEY`
  - `GALILEO_API_KEY`
- Edit `config.yaml` with relevant configuration details.

---

## 🚀 Usage

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

## Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

> Built with ❤️ using Z by HP AI Studio.