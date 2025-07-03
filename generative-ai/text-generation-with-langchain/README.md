# Text Generation with Galileo

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![LangChain](https://img.shields.io/badge/LangChain-used-lightgreen.svg?logo=langchain)
![Streamlit UI](https://img.shields.io/badge/User%20Interface-Streamlit-ff4b4b.svg?logo=streamlit)
![Galileo](https://img.shields.io/badge/Galileo-integrated-purple.svg)

</div>

## Content
* [🧠 Overview](#overview)
* [🗂 Project Structure](#project-structure)
* [⚙️ Setup](#setup)
* [🚀 Usage](#usage)
* [📞 Contact and Support](#contact-and-support)

## Overview 
This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It also integrates Galileo Prompt Quality for evaluation and logging, and supports multi-source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.

## Proect Struture
```
├── demo/                                             # UI-related files
│   ├── main.py                         
│   ├── poetry.lock                      
│   ├── pyproject.toml                   
│   └── README.md                       
│
├── docs/
│   └── streamlit_sucess.png                          # Streamlit Sucess UI Screenshot     
│
├── notebooks/
│   └── text-generation-with-langchain.ipynb          # Main notebook for the project
│
├── core/                                             # Core Python modules
│   ├── analyzer/
│   │   └── scientific_paper_analyzer.py 
│   ├── deploy/
│   │   └── text_generation_service.py   
│   ├── extract_text/
│   │   └── arxiv_search.py            
│   └── generator/
│       └── script_generator.py          
│
├── README.md                                         # Project documentation                  
├── requirements.txt                                  # Python dependencies             


```

## Setup

### 0 ▪ Minimum Hardware Requirements

Ensure your environment meets the minimum hardware requirements for smooth model inference:

- RAM: 16 GB  
- VRAM: 8 GB  
- GPU: NVIDIA GPU

### Quickstart

### Step 1: Create an AIstudio Project
1. Create a **New Project** in AI Studio
2. Select the template Text Generation with Langchain
3. Add a title description and relevant tags.

### Step 2: Verify Project Files
1. Launch a workspace.
2. Navigate to `03-text-generation/notebooks/text-generation-with-langchain.ipynb` to ensure all files were cloned correctly.


## Alternative Manual Setup

### Step 1: Create an AIStudio Project
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags.

### Step 2: Create a Workspace
1. Choose **Local GenAI** as the base image when creating the workspace.

### Step 3: Log Model
1. In the Datasets tab, click Add Dataset.
2. Upload the model file: `ggml-model-f16-Q5_K_M.gguf.`
3. The model will be available under the /datafabric directory in your workspace.

### Step 4: Verify Project Files  
1. In the Project Setup tab, under Setup, clone the project repository:
   ```
   git clone git clone https://github.com/HPInc/AI-Blueprints.git
   ```  
2. Navigate to `03-text-generation/notebooks/text-generation-with-langchain.ipynb` to ensure all files are cloned correctly after workspace creation.  

### Step 5: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.

## Usage 
1. Open and execute the notebook `text-generation-with-langchain.ipynb`
2. In the **Run and Approve section**, you can customize prompts, add presentation sections, and view results directly in the Galileo Console.
```python
generator.add_section(
    name="title",
    prompt="Generate a clear and concise title for the presentation that reflects the content. Add a subtitle if needed. Respond using natural language only."
)
```
3.  Deploy the Text Generation Service
- In AI Studio, navigate to **Deployments > New Service**.  
- Give your service a name (e.g. “Text Generation Service”), then select the registered Scrript Generation Sevice.  
- Pick the desired model version and enable **GPU acceleration** for best performance.  
- Click **Deploy** to launch the service.

4.  Swagger / Raw API
#### Example payload for text-only translation:
```jsonc
{
  "inputs": {
    "query": [
      "graph neural networks"
    ],
    "max_results": [
      1
    ],
    "chunk_size": [
      1200
    ],
    "chunk_overlap": [
      400
    ],
    "do_extract": [
      true
    ],
    "do_analyze": [
      true
    ],
    "do_generate": [
      true
    ],
    "analysis_prompt": [
      "Summarize the content in English (≈150 words)."
    ],
    "generation_prompt": [
      "Create a concise 5-point presentation script based on the summary."
    ]
  },
  "params": {}
}

````
Paste that into the Swagger “/invocations” endpoint and click **Try it out** to see the raw JSON response.

5. Lauch the Streamlit UI
-  To launch the Streamlit UI, follow the instructions in the README file located in the `demo/` folder.
-  Enter the **fields** and have fun


### Successful UI demo
![Automated Evaluation Streamlit UI](docs/streamlit_sucess.png)  



## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.


---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html)
