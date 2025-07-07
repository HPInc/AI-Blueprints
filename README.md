<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Projects for HP AI Studio 🚀 </h1>

<p align="center">
  <img src="assets/images/ai_studio_logo.png" alt="AI Studio Logo" width="200">
</p>

## Table of Contents

- [Troubleshooting](#troubleshooting) 

---

## 📋 Overview

Welcome to the **AI Blueprint Projects** repository! This collection contains a variety of projects that demonstrate how to build, train, and deploy machine learning and AI models using **HP AI Studio**. Each project is designed to be a **blueprint** that provides practical, real-world examples of AI development workflows.

---

## 🚀 Getting Started

To get started with these AI Blueprint projects:

1. **Clone this repository** to your local machine or workspace.
2. **Navigate to any project folder** that interests you.
3. **Follow the README.md instructions** within each project directory.
4. **Set up the required dependencies** as outlined in each project's requirements file.

```
```

---

## 📚 Projects Overview

Below is a comprehensive list of all available AI Blueprint projects, organized by category. Each project includes detailed documentation, code examples, and step-by-step instructions.

---

## 🧠 Deep Learning Projects

### 📝 Question Answering with BERT

**🔗 Path:** `deep-learning/question-answering-with-bert/`

**📝 Description:** This project demonstrates how to fine-tune a BERT model for question-answering tasks using the SQuAD dataset. It covers data preprocessing, model training, evaluation metrics, and deployment strategies for production use.

**🔧 Key Technologies:** BERT, Transformers, PyTorch, Hugging Face, SQuAD Dataset

**📋 Features:**
- Pre-trained BERT model fine-tuning
- SQuAD dataset integration
- Performance evaluation and metrics
- Model deployment examples

---

### 🌸 Classification with SVM

**🔗 Path:** `data-science/classification-with-svm/`

**📝 Description:** A comprehensive guide to building classification models using Support Vector Machines (SVM). This project uses the classic Iris dataset to demonstrate feature selection, model training, hyperparameter tuning, and performance evaluation.

**🔧 Key Technologies:** scikit-learn, pandas, matplotlib, seaborn

**📋 Features:**
- Data preprocessing and visualization
- SVM model implementation
- Hyperparameter optimization
- Cross-validation techniques
- Model performance analysis

---

### 🖌️ Classification with Keras

**🔗 Path:** `deep-learning/classification-with-keras/`

**📝 Description:** Build and train neural networks for image classification using Keras and TensorFlow. This project focuses on handwritten digit recognition using the MNIST dataset, covering model architecture design, training optimization, and evaluation.

**🔧 Key Technologies:** Keras, TensorFlow, MNIST, Neural Networks

**📋 Features:**
- Neural network architecture design
- MNIST dataset handling
- Model training and validation
- Performance visualization
- Prediction and inference

---

### 🛡️ Spam Detection with NLP

**🔗 Path:** `deep-learning/spam-detection-with-nlp/`

**📝 Description:** Implement natural language processing techniques to build an effective spam detection system. This project covers text preprocessing, feature extraction, model training, and evaluation using various NLP libraries and techniques.

**🔧 Key Technologies:** NLTK, scikit-learn, pandas, Text Processing

**📋 Features:**
- Text preprocessing and cleaning
- Feature extraction (TF-IDF, n-grams)
- Multiple classification algorithms
- Model comparison and evaluation
- Real-time spam detection

---

## 🤖 Generative AI Projects

### 📊 Automated Evaluation with Structured Outputs

**🔗 Path:** `generative-ai/automated-evaluation-with-structured-outputs/`

**📝 Description:** This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and includes evaluation tracking capabilities. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.

**🔧 Key Technologies:** LangChain, Transformers, Structured Outputs

**📋 Features:**
- Code snippet extraction from Jupyter notebooks
- LLM-powered natural language generation
- Structured output generation
- PromptQuality for evaluating model responses with human-like scorers (e.g., context adherence)
- Multi-source model support (local/hosted)
- GitHub integration for notebook crawling
- Vector store integration for RAG workflows
- Configurable prompt templating

---

### 📄 Text Summarization with LangChain

**🔗 Path:** `generative-ai/text-summarization-with-langchain/`

**📝 Description:** This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It includes evaluation and logging capabilities, and supports multi-source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.

**🔧 Key Technologies:** LangChain, Sentence Transformers, LLaMA

**📋 Features:**
- Semantic chunking and summarization
- This project demonstrates how to build a semantic chunking and summarization pipeline for texts using LangChain and Sentence Transformers for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the Meta Llama 3.1 model with 8B parameters to generate concise and contextually accurate summaries from text data.

---

### 🤖 Vanilla RAG with LangChain

**🔗 Path:** `generative-ai/vanilla-rag-with-langchain/`

**📝 Description:** This project is an AI-powered vanilla RAG (Retrieval-Augmented Generation) chatbot built using LangChain for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the Meta Llama 3.1 model with 8B parameters to generate contextual and document-grounded answers to user queries about Z by HP AI Studio.

**🔧 Key Technologies:** LangChain, ChromaDB, LLaMA, RAG

**📋 Features:**
- Document loading and processing
- Vector storage with ChromaDB
- Question answering with context
- Multi-source model support
- Configurable pipeline

---

## 🔗 NGC Integration Projects

### 🤖 Agentic RAG with TensorRT-LLM

**🔗 Path:** `ngc-integration/agentic-rag-with-tensorrtllm/`

**📝 Description:** Build advanced agentic RAG systems using NVIDIA's TensorRT-LLM for optimized inference performance. This project demonstrates how to create intelligent agents that can reason, plan, and execute complex tasks using retrieval-augmented generation.

**🔧 Key Technologies:** TensorRT-LLM, NVIDIA NGC, Agentic AI, RAG

**📋 Features:**
- High-performance LLM inference with TensorRT
- Agentic behavior implementation
- Advanced RAG architectures
- NVIDIA NGC integration
- Multi-step reasoning capabilities

---

## 🛠️ Installation and Setup

Each project contains its own setup instructions, but here are the general steps:

### Prerequisites

- **Python 3.8+**
- **pip** or **conda** package manager
- **Git** for version control
- **HP AI Studio** access (for cloud-based projects)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Blueprints
   ```

2. **Navigate to a specific project:**
   ```bash
   cd <project-category>/<project-name>
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Follow project-specific instructions** in the individual README files.

---

## 📚 Documentation

Each project includes:

- **📖 README.md**: Detailed setup and usage instructions
- **📓 Jupyter Notebooks**: Interactive code examples and tutorials
- **📋 requirements.txt**: Python dependency specifications
- **⚙️ Configuration files**: Model and environment settings
- **📊 Sample data**: Example datasets for testing and learning

---

## 🤝 Contributing

We welcome contributions to the AI Blueprint Projects! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/your-feature`)
3. **Make your changes** and test thoroughly
4. **Commit your changes** (`git commit -am 'Add new feature'`)
5. **Push to the branch** (`git push origin feature/your-feature`)
6. **Create a Pull Request**

### Contribution Guidelines

- Follow existing code style and structure
- Include comprehensive documentation
- Add unit tests where applicable
- Update README files as needed
- Ensure compatibility with HP AI Studio

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for full details.

---

## 🆘 Troubleshooting

### Common Issues

**Issue: Package installation failures**
- Solution: Ensure you're using the correct Python version (3.8+)
- Try using virtual environments: `python -m venv venv && source venv/bin/activate`

**Issue: CUDA/GPU related errors**
- Solution: Verify GPU drivers and CUDA installation
- Check compatibility between PyTorch/TensorFlow versions and CUDA

**Issue: Memory errors during model training**
- Solution: Reduce batch size or use gradient accumulation
- Consider using mixed precision training

**Issue: Data loading problems**
- Solution: Check file paths and permissions
- Verify data format matches expected input

### Getting Help

If you encounter issues:

1. **Check the project-specific README** for detailed instructions
2. **Review the troubleshooting section** in individual projects
3. **Search existing issues** in the repository
4. **Create a new issue** with detailed information about your problem

---

## 📞 Support

For additional support:

- **📧 Email**: [Support Team](mailto:support@example.com)
- **💬 Community Forum**: [AI Studio Community](https://community.example.com)
- **📚 Documentation**: [HP AI Studio Docs](https://docs.example.com)

---

**🚀 Happy Building with AI Studio!**