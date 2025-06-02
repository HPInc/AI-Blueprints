# Notebook templates for integrating AI Studio and Galileo

In this repository, we provide a series of use cases to illustrate the integration between AI Studio and Galileo, often through the use of LangChain to orchestrate the language pipelines and allow the logging of metrics into Galileo.

## Repository Structure

The repository is organized into the following structure:

```
├── configs/                    # Shared configuration resources
│    ├── config.yaml            # Configuration settings
│    └── secrets.yaml           # API keys (gitignored, you must create this file and fill with the necessary API keys)
│
├── src/                      # Common utility functions and shared code between notebooks
│
├── 00-chatbot-with-langchain/  # Chatbot template
│    ├── data/                  # Specific data files for the chatbot example
│    │    └── AIStudioDoc.pdf   # Documentation used for the RAG pipeline
│    ├── notebooks/             # Contains the notebook files
│    │    └── chatbot-with-langchain.ipynb   
│    ├── demo/                  # UI demo output and source code
│    ├── README.md             # Detailed documentation for the chatbot example
│    └── requirements.txt       # Dependencies for the chatbot example
│
├── 01-summarization-with-langchain/  # Summarization template
│    ├── data/                  # Specific data files for the summarization example
│    │    ├── I_have_a_dream.txt
│    │    └── I_have_a_dream.vtt
│    ├── notebooks/             # Contains the notebook files
│    │    └── summarization-with-langchain.ipynb
│    ├── README.md             # Detailed documentation for the summarization example
│    └── requirements.txt       # Dependencies for the summarization example
│
├── 02-code-generation-with-langchain/  # Code generation template
│    ├── notebooks/             # Contains the notebook files
│    │    └── code-generation-with-langchain.ipynb
│    └── README.md             # Detailed documentation for the code generation example
│
└── 03-text-generation-with-langchain/  # Text generation template
     ├── notebooks/             # Contains the notebook files
     │    └── text-generation-with-langchain.ipynb
     ├── README.md             # Detailed documentation for the text generation example
     └── requirements.txt       # Dependencies for the text generation example
```

## Available Templates

### Chatbot (00-chatbot-with-langchain/notebooks/chatbot-with-langchain.ipynb)

In this simpler example, we implement a basic chatbot assistant to AI Studio, by means of a basic RAG pipeline. In this pipeline, we load information from a document (AIStudio documentation) into a vector database, then use this information to answer questions about AI Studio through proper prompting and use of LLMs. In the example, we illustrate different ways to load the model (locally and from cloud), and also illustrate how to use Galileo's callbacks to log information from the LangChain modules.

### Summarization (01-summarization-with-langchain/notebooks/summarization-with-langchain.ipynb)

For this use case, we extend the basic scenario to include more complex pre-processing of the input. In our scenario, we break an original transcript (which might be too long) into smaller topics (chunks with semantic relevance). A chain is then built to summarize the chunks, then joining them into a single summary in the end.

Also in this example, we illustrate how to work with:
* Personalized runs from Galileo (using EvaluateRuns)
* Personalized Metrics that runs locally (using CustomScorers)

### Code Generation (02-code-generation-with-langchain/notebooks/code-generation-with-langchain.ipynb)

This use case illustrates an example where the user accesses a git repository to serve as code reference. Some level of code understanding is used to index the available code from the repository. Based on this reference, the code generator uses in-cell prompts from the user in order to generate code in new notebook cells.

### Text Generation (03-text-generation-with-langchain/notebooks/text-generation-with-langchain.ipynb)

This use case shows a process to search for a scientific paper in ArXiv, then generating a presentation based on the content of this paper.

## Configuration Files

### configs/config.yaml
This file contains non-sensitive configuration parameters such as model sources, URLs, and other settings needed across the notebooks. You can modify this file to adjust various aspects of the example templates.

The notebooks support multiple model sources which can be configured in the configs/config.yaml file:

- **local**: by loading the llama2-7b model from the asset downloaded on the project
- **hugging-face-local**: by downloading a DeepSeek model from Hugging Face and running locally
- **hugging-face-cloud**: by accessing the Mistral model through Hugging Face cloud API (requires HuggingFace API key)

### configs/secrets.yaml
This file contains your API keys and other sensitive credentials required to run the examples. For security reasons, this file is included in `.gitignore` and **must be created manually** with the following structure:

```yaml
# Galileo API key (required for all examples)
GALILEO_API_KEY: "your_galileo_api_key_here"

# HuggingFace API key (required only if using hugging-face-cloud model source)
HUGGINGFACE_API_KEY: "your_huggingface_api_key_here"

# Add any other keys needed for additional services
```

To obtain your Galileo API key, visit: https://console.hp.galileocloud.io/settings/api-keys

## Utility Directory (utils/)

The `utils/` directory contains shared utility functions, helper classes, and common code used across multiple notebooks. This centralized approach helps maintain consistency between examples and reduces code duplication. When working with the templates, you can import these utilities into your notebooks to leverage common functionality.

## Important notes when running these examples

To run the examples, you'll need to:

1. Install the requirements for each template using the provided requirements.txt file.
2. Set up your Galileo API key in the secrets.yaml file located in the configs directory.
3. Configure the model source in the config.yaml file according to your preferences.