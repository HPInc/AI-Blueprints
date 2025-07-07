# 🤖 Vanilla RAG with LangChain

<div align="center">
  <img src="../../assets/images/ai_studio_logo.png" alt="AI Studio Logo" width="150">
</div>

---

# 📚 Contents

- [📋 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [Usage](#usage)
- [🔧 Configuration](#-configuration)
- [🎯 Key Features](#-key-features)
- [📊 Example Usage](#-example-usage)
- [🧪 Demo Application](#-demo-application)

---

## 📋 Overview

This project is an AI-powered vanilla **RAG (Retrieval-Augmented Generation)** chatbot built using **LangChain** for model evaluation, protection, and observability. It leverages the **Z by HP AI Studio Local GenAI image** and the Meta Llama 3.1 model with 8B parameters to generate contextual and document-grounded answers to user queries about **Z by HP AI Studio**.

---

## 📁 Project Structure

```
vanilla-rag-with-langchain/
│
├── configs/
│   ├── config.yaml                                # Configuration parameters
│   └── secrets.yaml                               # API keys and credentials
│
├── core/
│   └── chatbot_service/
│       └── chatbot_service.py                     # Main chatbot service implementation
│
├── data/
│   └── AIStudioDoc.pdf                            # Sample PDF document for RAG
│
├── demo/
│   └── streamlit_app.py                           # Streamlit demo application
│
├── docs/
│   └── architecture.md                            # Technical documentation
│
├── notebooks/
│   ├── register-model.ipynb                       # Model registration notebook
│   └── vanilla-rag-with-langchain.ipynb                               # Main notebook for the project
│
├── src/
│   ├── service/
│   │   └── base_service.py                        # Base service class
│   ├── utils.py                                   # Utility functions
│   └── prompt_templates.py                        # Prompt templates
│
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **HP AI Studio** account and access
- **Python 3.8+**
- **CUDA-compatible GPU** (optional, for faster processing)

### Step 1: Set up HP AI Studio

1. **Access HP AI Studio** and create a new workspace.
2. **Upload this project** to your workspace.
3. **Select the Z by HP AI Studio Local GenAI image** when launching your workspace.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

   ```

### Step 3: Download the Required Model

- Navigate to the **Data Fabric** section in HP AI Studio.
- **Download the Meta Llama 3.1 8B model** to your workspace:
  - **Bucket Name**: `meta-llama3.1-8b-Q8`
  - **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace. If the model does not appear after downloading, please restart your workspace.

### Step 5: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file located in the `configs` folder:
  - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

```bash
notebooks/vanilla-rag-with-langchain.ipynb
```

### Step 2: Key Features

- Load and process PDF documents
- Create vector embeddings for document retrieval
- Generate contextual answers using RAG
- Integrate evaluation, protection, and observability
- Support for multiple model sources

---

## 🔧 Configuration

### config.yaml
```yaml
model:
  source: "local"  # Options: "local", "huggingface_hosted", "huggingface_cloud"
  model_name: "meta-llama/Llama-3.1-8b"
  
rag:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5
  
retrieval:
  similarity_threshold: 0.7
```

### secrets.yaml
```yaml
HUGGINGFACE_API_KEY: "your_huggingface_api_key_here"
```

---

## 🎯 Key Features

- **📄 Document Processing**: Load and process PDF documents using LangChain loaders
- **🔍 Vector Storage**: Store document embeddings using ChromaDB for efficient retrieval
- **❓ Question Answering**: Generate contextual answers using retrieval-augmented generation
- **🤖 Multi-Model Support**: Compatible with local LLaMA models and HuggingFace-hosted models
- **⚙️ Configurable Pipeline**: Flexible configuration through YAML files
- **📊 Evaluation Framework**: Built-in metrics for response quality assessment
- **🛡️ Content Protection**: Safeguards against sensitive information exposure
- **📈 Observability**: Comprehensive logging and monitoring capabilities

---

## 📊 Example Usage

```python
from core.chatbot_service.chatbot_service import ChatbotService

# Initialize the chatbot service
chatbot = ChatbotService(config_path="configs/config.yaml")

# Ask a question
question = "What is HP AI Studio?"
response = chatbot.ask(question)

print(f"Question: {question}")
print(f"Answer: {response}")
```

---

## 🧪 Demo Application

Run the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

The demo provides an interactive chat interface where you can:
- Upload PDF documents
- Ask questions about the uploaded content
- View retrieval context and sources
- Monitor response quality metrics

---

## 📚 Additional Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Meta Llama Documentation](https://llama.meta.com/)
- [HP AI Studio Documentation](https://developers.hp.com/ai-studio)

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Happy Building with AI Studio!**