# 🤖 Vanilla RAG with LangChain and Galileo

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

This project is an AI-powered vanilla **RAG (Retrieval-Augmented Generation)** chatbot built using **LangChain** and **Galileo** for model evaluation, protection, and observability. It leverages the **Z by HP AI Studio Local GenAI image** and the **LLaMA2-7B** model to generate contextual and document-grounded answers to user queries about **Z by HP AI Studio**.

---

## Project Structure

```
├── core
│   └── chatbot_service                                                 # Core Python modules      
│       ├── __init__.py
│       └── chatbot_service.py                              
├── data                                                                # Data assets 
│   └── AIStudioDoc.pdf                                                 # AIStudio documentation
├── demo                                                                # UI-related files
│   ├── assets
│   ├── index.html
│   ├── source
│   └── streamlit-webapp
├── docs
│   ├── html_ui_for_vanilla_rag.png                                     # HTML UI Screenshot
│   ├── streamlit_ui_for_vanilla_rag.png.png                            # Streamlit UI Screenshot
│   └── successful streamlit ui result for vanilla rag.pdf              # Successful Streamlit UI Screenshot
├── notebooks
│   └── vanilla-rag-with-langchain-and-galileo.ipynb                    # Main notebook for the project
├── README.md                                                           # Project documentation
└── requirements.txt                                                    # Python dependencies
```

---

## Setup

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- RAM: 32 GB 
- VRAM: 6 GB 
- GPU: NVIDIA GPU 

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation.

### Step 4: Add the Model to Workspace

- Download the **LLaMA2-7B** model from AWS S3 using the Models tab in your AI Studio project:
  - **Model Name**: `llama2-7b`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/llama2-7b`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace. If the model does not appear after downloading, please restart your workspace.
  
### Step 5: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file located in the `configs` folder:
  - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.
  - `GALILEO_API_KEY`: Required to connect to Galileo for evaluation, protection, and observability features.
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/vanilla-rag-with-langchain-and-galileo.ipynb
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
- From the Swagger page, click the demo link to interact with the locally deployed vanilla RAG chatbot via UI.

### Successful Demonstration of the User Interface  

![Vanilla RAG HTML UI](docs/html_ui_for_vanilla_rag.png)  

![Vanilla RAG Streamlit UI](docs/streamlit_ui_for_vanilla_rag.png.png)  

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
