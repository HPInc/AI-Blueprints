<h1 style="text-align: center; font-size: 40px;"> NGC Integration Blueprint Projects for HP AI Studio </h1>

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

We provide **5 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

## Repository Structure

The repository is organized into the following structure:

```
├── agentic-rag-with-tensorrtllm/
|    ├── data/                                     # Data assets used in the project
|    │   └── context/
|    │       └── aistudio
|    ├── docs/
|    |   ├── architecture-for-agentic-rag.png       # Architecture screenshot of the agentic RAG system
|    |   └── Build Custom Agentic RAG Systems.pptx  # Walkthrough for building general agentic RAG systems
|    |
|    ├── notebooks/
|    |   ├── register-model.ipynb                 # Notebook for registering trained models to MLflow
|    │   └── run-workflow.ipynb                   # Notebook for executing the pipeline
|    ├── src/                                     # Core Python modules
|    │   ├── __init__.py
|    │   ├── trt_llm_langchain.py
|    |   └── workspace.sh
|    ├── README.md                                # Project documentation
|    └── requirements.txt                         # Python dependencies
|
├── audio-translation-with-nemo-models/
│    ├── data/                                     # Data assets used in the project
│    |   ├── ForrestGump.mp3
│    |   └── June18.mp3
│    ├── demo                                      # UI-related files
│    |   └── ...
│    ├── docs
│    |   ├── successful-streamlit-ui-audio-translation-result.pdf  # React UI screenshot
│    |   └── successful-swagger-ui-audio-translation-result.pdf    # Streamlit UI screenshot
|    ├── notebooks
|    |   ├── register-model.ipynb                  # Notebook for registering trained models to MLflow
│    |   └── run-workflow.ipynb                    # Notebook for executing the pipeline
|    ├── README.md                                 # Project documentation
|    └── requirements.txt                          # Python dependencies (used with pip install)
|
├── data-analysis-with-cudf/
|    ├── docs
|    |   ├── Analysis-with-Pandas-5M.png          # Stock analysis using Pandas UI screenshot (5M dataset size)
|    |   ├── Analysis-with-Pandas-10M.png         # Stock analysis using Pandas UI screenshot (10M dataset size)
|    |   ├── Analysis-with-Pandas-and-cuDF-5M.png     # Stock analysis using Pandas and cuDF (5M dataset size)
|    |   └── Analysis-with-Pandas-and-cuDF-10M.png    # Stock analysis using Pandas and cuDF(10M dataset size)
|    ├── notebooks
|    │   ├── stock-analysis-with-pandas                # Directory of notebooks using Pandas only (CPU)
|    |   |   └── run-workflow.ipynb                    # Notebook used for executing the pipeline
|    │   └── stock-analysis-with-pandas-and-cudf       # Directory of notebooks using cuDF (GPU)
|    |       └── run-workflow.ipynb                    # Notebook used for executing the pipeline
|    ├── README.md                                 # Project documentation
|    └── requirements.txt                          # Python dependencies (used with pip install)
|
├── data-visualization-with-cudf/                  
|    ├── docs
|    |   ├── ui-opencellid-EU.png                  # opencellid UI screenshot (European Union map)
|    │   └── ui-opencellid-US.png                  # opencellid UI screenshot (United States map)
|    ├── notebooks
|    │   └── run-workflow.ipynb                    # Main notebook for the project
|    ├── src                                       # Core Python modules
|    │   └── opencellid_downloader.py
|    ├── README.md                                 # Project documentation
|    └── requirements.txt                          # Python dependencies (used with pip install)
│
├── vacation-recommendation-with-bert/
|   ├── data                                       # Data assets used in the project
|   │   ├── ForrestGump.mp3
|   │   └── June18.mp3
|   ├── demo                                       # UI-related files
|   │   └── ...
|   ├── docs
|   │   ├── successful-streamlit-ui-audio-translation-result.pdf    # Streamlit UI screenshot pdf file
|   │   └── successful-swagger-ui-audio-translation-result.pdf      # Swagger UI screenshot pdf file
|   ├── notebooks
|   |   ├── register-model.ipynb                            # Notebook for registering trained models to MLflow
|   │   └── run-workflow.ipynb                              # Notebook for executing the pipeline
|   ├── README.md                                           # Project documentation
|   └── requirements.txt                                    # Python dependencies (used with pip install)
|



```

# 🤖 Agentic RAG for AI Studio with TRT-LLM and LangGraph

This project contains a single integrated pipeline—**Agentic RAG for AI Studio with TRT-LLM and LangGraph**—that implements a Retrieval-Augmented Generation (RAG) workflow using:

- **TensorRT-backed Llama-3.1-Nano (TRT-LLM)**: for fast, GPU-accelerated inference.
- **LangGraph**: to orchestrate an agentic, multi-step decision flow (relevance check, memory lookup, query rewriting, retrieval, answer generation, and memory update).
- **ChromaDB**: as a local vector store over Markdown context files (about AI Studio).
- **SimpleKVMemory**: a lightweight on-disk key-value store to cache query-answer pairs.

# 🎙️ Audio Translation with NeMo Models

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet
2. **Text Translation (TT)** from English to Spanish using NMT
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.

# 📡 OpenCellID Exploratory Data Analysis with Panel and cuDF

This project is a GPU-accelerated, interactive **exploratory data analysis (EDA)** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It uses **Panel** and **cuDF** to deliver lightning-fast geospatial analysis and visualization.

You can explore cell tower distributions by radio type, operator, country, and time window — rendered live on an interactive map with full GPU acceleration.

# 📈 Stock Analysis with Pandas and cuDF

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Original Example Notebook**: This version, created by NVIDIA, runs the entire evaluation within a single notebook. It includes downloading the data and restarting the kernel to activate the cuDF extension.

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from the **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.

# 🌍 Vacation Recommendation Service

The **Vacation Recommendation Service** is an AI-powered system designed to provide personalized travel recommendations based on user queries.

It utilizes the **NVIDIA NeMo Framework** and **BERT embeddings** to generate relevant suggestions tailored to user preferences.


# Contact and Support

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
