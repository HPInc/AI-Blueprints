# 🤖 Agentic RAG  

# 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

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
│   ├── context
│   │   └── aistudio
├── notebooks
│   └── Agentic RAG for AI Studio with TRT-LLM and LangGraph.ipynb
├── src
│   ├── __init__.py
│   └── trt_llm_langchain.py
├── README.md
└── requirements.txt
```  

# Setup  

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- RAM: 64 GB 
- VRAM: 32 GB 
- GPU: NVIDIA GPU 
- Disk: 32 GB Free Space

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   

### Step 2: Create a Workspace  
1. Select **NeMo Framework (version 25.04)** as the base image.    

### Step 3: Verify Project Files  
1. Clone the GitHub repository:  
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```  
2. Navigate to `generative-ai/agentic_rag_with_trt-llm_and_langgraph` to ensure all files are cloned correctly after workspace creation.  


# Usage  

### Step 1: Use the Agentic Workflow

Run the following notebook to see the Agentic Workflow in action:  
- **`Agentic RAG for AI Studio with TRT-LLM and LangGraph.ipynb`**



# Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  

---

> Built with ❤️ using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).
