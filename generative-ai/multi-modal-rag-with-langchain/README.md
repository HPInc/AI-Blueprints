# 🤖 MultiModal RAG with LangChain, Transformers and Torch

<div align="center">

  ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)

</div>

# 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

🤖 MultiModal RAG with LangChain, Transformers and Torch
📚 Contents
🧠 Overview

🗂 Project Structure

⚙️ Setup

🚀 Usage

📞 Contact and Support

Overview
This project implements an AI-powered Multimodal **RAG (Retrieval-Augmented Generation)** chatbot. It's built using **LangChain** for orchestration, **Hugging Face Transformers** and **PyTorch** for the underlying multimodal model, and **MLflow** for model evaluation and observability. The chatbot leverages the **Z by HP AI Studio Local GenAI image** and the **`InternVL3-8B-Instruct`** model to generate contextual answers, grounded in both documents and images, to user queries about internal documentation. In this example, the primary data source is an Azure DevOps Wiki.

---

## Project Structure

```
multimodal-rag-with-langchain-mlflow/
├── data/                                              # Data assets used in the project
│   ├── context/
│   │   ├── images/                                    # Directory for images referenced in wiki pages
│   │   └── wiki_flat_structure.json                   # JSON metadata for ADO Wiki data
│   ├── chroma_store/                                  # Persistent directory for ChromaDB vector stores
│   │   └── manifest.json                              # Manifest file for tracking indexed context files
│   └── memory/
│       └── memory.json                                # Lightweight on-disk key-value store for caching
├── notebooks/
│   └── multimodal-rag-with-langchain-mlflow.ipynb     # Main notebook for the project
├── src/                                               # Core Python modules
│   ├── __init__.py
│   └── utils.py                                       # Utility functions for configuration and caching
├── configs/
│   └── config.yaml                                    # Configuration parameters (non-sensitive)
├── README.md                                          # Project documentation
└── requirements.txt                                   # Python dependencies
```

---

## Setup

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- RAM: 32 GB
- VRAM: 24 GB
- GPU: NVIDIA GPU

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.
- Upload requirements.txt file to the pip packages section of your AI Studio workspace.


### Step 3: Clone the Repository

1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. `Navigate to generative-ai/multimodal-rag-with-langchain-mlflow to ensure all files are cloned correctly after workspace creation.`

### Step 4: Add the Model to Workspace

- Download the **InternVL3-8B-Instruct-Q8** model from AWS S3 using the Models tab in your AI Studio project:
  - **Model Name**: `InternVL3-8B-Instruct-Q8`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your jupyter notebook workspace. If the model does not appear after downloading, please restart your workspace.
  
### Step 5: Configure Paths and Config
- Edit `config.yaml` with relevant configuration details. (Currently no config required)

---

## Usage

### Step 1: Run the Multimodal RAG Workflow

Execute the following notebook inside the `notebooks/` folder to see the Multimodal RAG workflow in action:

- **`run-workflow.ipynb`**

### Step 2: Register the Multimodal RAG Model in MLflow

Run the following notebook in the `notebooks/` folder to register the Multimodal RAG model in MLflow:

- **`register-model.ipynb`**

### Step 3: Deploy the Multimodal RAG Service Locally
- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.
- From the Swagger page, click the demo link to interact with the locally deployed vanilla RAG chatbot via the Streamlit UI.

### Successful Demonstration of the User Interface  

![Multimodal RAG HTML UI]()  

![Multimodal RAG Streamlit UI]()  

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
