# Text Summarization with LangChain

<div align="center">
  <img src="../../assets/images/ai_studio_logo.png" alt="AI Studio Logo" width="150">
</div>

---

## 📋 Overview

This project demonstrates how to build a semantic chunking and summarization pipeline for texts using **LangChain** and **Sentence Transformers** for model evaluation, protection, and observability. It leverages the **Z by HP AI Studio Local GenAI image** and the Meta Llama 3.1 model with 8B parameters to generate concise and contextually accurate summaries from text data.

---

## 📁 Project Structure

```
text-summarization-with-langchain/
│
├── configs/
│   ├── config.yaml                                # Configuration parameters
│   └── secrets.yaml                               # API keys and credentials
│
├── core/
│   └── summarizer_service/
│       └── summarizer_service.py                  # Main service implementation
│
├── data/
│   └── sample_text.txt                            # Sample text for summarization
│
├── demo/
│   └── streamlit_app.py                           # Streamlit demo application
│
├── docs/
│   └── architecture.md                            # Technical documentation
│
├── notebooks/
│   ├── register-model.ipynb                       # Model registration notebook
│   └── text-summarization-with-langchain.ipynb                          # Main notebook for the project
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

### Step 4: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file located in the `configs` folder:
  - `HUGGINGFACE_HUB_TOKEN`
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

```bash
notebooks/text-summarization-with-langchain.ipynb
```

### Step 2: Key Features

- Build semantic chunking pipeline
- Generate contextual summaries
- Integrate evaluation, protection, and observability
- Support for multiple model sources

---

## 🔧 Configuration

### config.yaml
```yaml
model:
  source: "local"  # Options: "local", "huggingface_hosted", "huggingface_cloud"
  model_name: "meta-llama/Llama-3.1-8b"
  
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  
summarization:
  max_length: 150
  temperature: 0.7
```

### secrets.yaml
```yaml
HUGGINGFACE_API_KEY: "your_huggingface_api_key_here"
```

---

## 🎯 Key Features

- **📝 Semantic Chunking**: Intelligent text segmentation using sentence transformers
- **🤖 AI-Powered Summarization**: Generate concise summaries using Meta Llama 3.1
- **⚡ Multi-Source Support**: Local models, HuggingFace hosted, and cloud APIs
- **📊 Evaluation Framework**: Built-in metrics for summary quality assessment
- **🛡️ Content Protection**: Safeguards against sensitive information exposure
- **📈 Observability**: Comprehensive logging and monitoring capabilities

---

## 📊 Example Usage

```python
from core.summarizer_service.summarizer_service import SummarizerService

# Initialize the service
summarizer = SummarizerService(config_path="configs/config.yaml")

# Load and process text
text = "Your long text content here..."
summary = summarizer.summarize(text)

print(f"Summary: {summary}")
```

---

## 🧪 Demo Application

Run the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

---

## 📚 Additional Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Meta Llama Documentation](https://llama.meta.com/)
- [HP AI Studio Documentation](https://developers.hp.com/ai-studio)

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Happy Summarizing with AI Studio!**