# 🤖 Agentic RAG  

# Content  
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contact and Support](#contact-and-support)

# Overview  
This Agentic RAG project presents two different notebooks, each demonstrating a distinct architecture:

### RAG with Agentic Workflow

This notebook implements a **Retrieval-Augmented Generation (RAG)** pipeline with an **Agentic Workflow**, using a local **Llama 2** model and **ChromaDB** for intelligent question-answering.  

The system dynamically determines whether additional context is needed before generating responses, ensuring higher accuracy and relevance.

### Agentic RAG

This notebook showcases a **Hugging Face** model integrated with a **retriever tool**, enabling it to fetch and use relevant context dynamically when answering questions about **Z by HP AI Studio**.  

The solution is primarily built using the **LangChain** and **SmolAgents** libraries, creating an agent capable of context-aware retrieval and response generation.

# Project Structure  
```
├── data/                 
│   ├── AIStudioDoc.pdf  
├── notebooks
│   ├── agentic_rag.ipynb
│   └── rag_with_agentic_workflow.ipynb
├── README.md
└── requirements.txt
```  

# Setup  

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   

### Step 2: Create a Workspace  
1. Select **Local GenAI** as the base image.    

### Step 3: Verify Project Files  
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/aistudio-samples.git
   ```  
2. Navigate to `gen-ai/agentic_rag_with_langchain_and_smolagents` to ensure all files are cloned correctly after workspace creation.  

### Step 4: Add the Model to Workspace

- Download the **LLaMA2-7B** model from AWS S3 using the Models tab in your AI Studio project:
  - **Model Name**: `llama2-7b`
  - **Model Source**: `AWS S3`
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/llama2-7b`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace. If the model does not appear after downloading, please restart your workspace.

# Usage  

### Step 1: Use the Agentic Workflow

Run the following notebook to see the Agentic Workflow in action:  
- **`rag_with_agentic_workflow.ipynb`**

### Step 2: Use the Agent with Retriever Tool

Run the following notebook to see the agent enhanced with a retriever tool in action:  
- **`agentic_rag.ipynb`**

# Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).