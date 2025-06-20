# Text Summarization with LangChain and Galileo

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

This project demonstrates how to build a semantic chunking and summarization pipeline for texts using **LangChain**, **Sentence Transformers**, and **Galileo** for model evaluation, protection, and observability. It leverages the **Z by HP AI Studio Local GenAI image** and the **LLaMA2-7B** model to generate concise and contextually accurate summaries from text data.

---

## Project Structure

```
├── README.md                                                               # Project documentation
├── core                                                                    # Core Python modules
│   └── service
│       ├── __init__.py
│       └── text_summarization_service.py                                   # Code for chatbot service
├── data                                                                    # Data assets used in the project
│   ├── I_have_a_dream.txt
│   └── I_have_a_dream.vtt
├── notebooks
│   └── text-summarization-with-langchain-and-galileo.ipynb           # Main notebook for the project
└── requirements.txt                                                        # Python dependencies
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
- (Optional) Add a description and relevant tags.

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation..

### Step 4: Add the Model to Workspace

- Download the **LLaMA2-7B** model from AWS S3 using the Models tab in your AI Studio project:
  - **Dataset Name**: `llama2-7b`
  - **Dataset Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/llama2-7b`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace.

### Step 5: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file under the `configs` folder:
  - `HUGGINGFACE_API_KEY`
  - `GALILEO_API_KEY`
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/text-summarization-with-langchain-and-galileo.ipynb
```

This will:

- Set up the semantic chunking pipeline
- Create the summarization chain with LangChain
- Integrate Galileo evaluation, protection, and observability
- Register the model in MLflow

### Step 2: Deploy the Summarization Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.
- Use the API endpoints to generate summaries from your text data.

### Successful Demonstration of the User Interface

![text Summarization Demo UI](docs/ui_summarization.png)

:warning: Current implementation of deployed model **do not** perform the chunking steps: summarization is run directly by the LLM model. In the case of suggested local model (i.e. Llama2-7b), texts with more than 1000 words may cause instabilities when summarization is triggered on the UI. We recommend using different models or smaller texts to avoid these problems.

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
