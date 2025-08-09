# Generative AI

The blueprint projects in this folder demonstrate how to build generative AI applications with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **7 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

### 📊 Automated Evaluation with Structured Outputs

**Automated Evaluation with Structured Outputs** turns a local **Meta-Llama-3** model into an MLflow-served scorer that rates any batch of texts (e.g., project abstracts) against arbitrary rubric criteria.

- Generates scores locally via `llama.cpp` (no data leaves your machine)
- Registers the evaluator as a **pyfunc** model in MLflow
- Exposes a REST `/invocations` endpoint
- Ships two front-ends — a **Streamlit** dashboard and a pure **HTML/JS** UI — for instant human-friendly interaction and CSV download.

### Code Generation with Langchain

This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and integrates with PromptQuality and Galileo for evaluation and tracking. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.

### Fine Tuning with ORPO

This project demonstrates a full-stack LLM fine-tuning experiment using ORPO (Open-Source Reinforcement Pretraining Objective) to align a base language model with human preference data. It leverages the Z by HP AI Studio Local GenAI environment, and uses models such as LLaMA 3, Gemma 1B, and Mistral 7B as foundations.

We incorporate:

Galileo PromptQuality for evaluating model responses with human-like scorers (e.g., context adherence)
TensorBoard for human feedback visualization before fine-tuning
A flexible model selector and inference runner architecture
A comparative setup to benchmark base vs fine-tuned models on the same prompts

### Image Generation with Stable Diffusion

This notebook performs image generation inference using the Stable Diffusion architecture, with support for both standard and DreamBooth fine-tuned models. It loads configuration and secrets from YAML files, enables local or deployed inference execution, and calculates custom image quality metrics, such as entropy and complexity. The pipeline is modular, supports Hugging Face model loading, and integrates with PromptQuality for evaluation and tracking.

### Text Generation with LangChain

This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It also integrates Galileo Prompt Quality for evaluation and logging, and supports multi-source model loading, including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.

### Text Summarization with LangChain

This project demonstrates how to build a semantic chunking and summarization pipeline for texts using LangChain, Sentence Transformers, and Galileo for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the Meta Llama 3.1 model with 8B parameters to generate concise and contextually accurate summaries from text data.

### Vanilla RAG with LangChain

This project is an AI-powered vanilla RAG (Retrieval-Augmented Generation) chatbot built using LangChain and Galileo for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the Meta Llama 3.1 model with 8B parameters to generate contextual and document-grounded answers to user queries about Z by HP AI Studio.
