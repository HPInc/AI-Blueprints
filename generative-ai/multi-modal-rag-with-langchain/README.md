# 🤖 MultiModal RAG with LangChain, Transformers, and Torch

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

🤖 MultiModal RAG with LangChain, Transformers, and Torch

Overview
This project implements an AI-powered Multimodal **RAG (Retrieval-Augmented Generation)** chatbot. It's built using **LangChain** for orchestration, **Hugging Face Transformers** and **PyTorch** for the underlying multimodal model, and **MLflow** for model evaluation and observability. The chatbot leverages the **Z by HP AI Studio Local GenAI image** and the **`InternVL3-8B-Instruct`** model to generate contextual answers, grounded in both documents and images, to user queries about internal documentation. In this example, the primary data source is an Azure DevOps Wiki.

---

## Project Structure

```
multi-modal-rag-with-langchain/
├── configs/
├── core/
├── data/
│   └── chroma_store/
├── context/
│   ├── images/
│   ├── AIStudioDoc.pdf
│   ├── wiki_flat_structure_mini.json
│   └── wiki_flat_structure.json
├── demo/
├── notebooks/
│   ├── register-model.ipynb
│   └── run-workflow.ipynb
├── src/
│   └── service/
│       ├── local_genai_judge.py
│       └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### Step 0: Hardware Requirements

#### Minimum Hardware Requirements

To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- RAM: 32 GB
- VRAM: 12 GB
- GPU: NVIDIA GPU

#### Recommended Hardware Requirements

For optimal performance, especially when working with larger models or datasets, consider the following recommended hardware specifications:

- RAM: 64 GB
- VRAM: 24 GB
- GPU: NVIDIA RTX A6000 or equivalent

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.
- Upload `requirements.txt` file from the project directory to the pip packages section of your AI Studio workspace.

### Step 3: Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Navigate to `generative-ai/multimodal-rag-with-langchain-mlflow` to ensure all files are cloned correctly after workspace creation.

### Step 4: Add the Model to the Workspace

- Download the **InternVL3-8B-Instruct** model from AWS S3 using the Models tab in your AI Studio project:
  - **Model Name**: `InternVL3-8B-Instruct`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `TBD`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your jupyter notebook workspace. If the model does not appear after downloading, please restart your workspace.

### Step 4: Manual Model Download to the Workspace

- If you prefer to download the model manually, you can do so from the Hugging Face Model Hub:

- Go to the [OpenGVLab/InternVL3-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/tree/main).
- Download the model files manually and place them in a folder named `InternVL3-8B-Instruct`.
- Upload the `InternVL3-8B-Instruct` folder with the model using the Models tab in your AI Studio project:
  - **Model Name**: `InternVL3-8B-Instruct`
  - **Model Source**: `Local`
  - **Model Path**: `C:\path_to_your_model\InternVL3-8B-Instruct`
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

### Swagger API Request Body Schema

```
{
  "inputs": {
    "query": [
      "string"
    ],
    "force_regenerate": [
      true
    ]
  },
  "params": {}
}
```

### Successful Demonstration of the User Interface

![Multimodal RAG HTML UI](TBD)

![Multimodal RAG Streamlit UI](TBD)

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
