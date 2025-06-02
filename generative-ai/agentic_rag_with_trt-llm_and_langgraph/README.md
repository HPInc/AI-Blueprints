# 🤖 Agentic RAG for AI Studio with TRT-LLM and LangGraph



# 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

# Overview  
This repository contains a single integrated pipeline—**Agentic RAG for AI Studio with TRT-LLM and LangGraph**—that implements a Retrieval-Augmented Generation (RAG) workflow using:

- **TensorRT-backed Llama-3.1-Nano (TRT-LLM)**: for fast, GPU-accelerated inference.
- **LangGraph**: to orchestrate an agentic, multi-step decision flow (relevance check, memory lookup, query rewriting, retrieval, answer generation, and memory update).
- **ChromaDB**: as a local vector store over Markdown context files (about AI Studio).
- **SimpleKVMemory**: a lightweight on-disk key-value store to cache query-answer pairs.

---

# Project Structure  
```
agentic_rag_with_trt-llm_and_langgraph/
├── data/
│   └── context/
│       └── aistudio
├── notebooks/
│   └── Agentic RAG for AI Studio with TRT-LLM and LangGraph.ipynb
├── src/
│   ├── __init__.py
│   └── trt_llm_langchain.py
├── README.md
└── requirements.txt
```  

---

# Setup  

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- GPU: NVIDIA GPU with at least 32 GB VRAM (for TensorRT-LLM engine)

- RAM: ≥ 64 GB system memory

- Disk: ≥ 32 GB free

- CUDA: Compatible CUDA toolkit (11.8 or 12.x) installed on your system

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

---

# Usage  

### Step 1: Use the Agentic Workflow

Run the following notebook to see the Agentic Workflow in action:  
- **`Agentic RAG for AI Studio with TRT-LLM and LangGraph.ipynb`**

---

# Contact and Support  

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**Z by HP AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
