# Agentic RAG Notebook

## Overview
This notebook implements an **Agentic Retrieval-Augmented Generation (RAG) pipeline** using **Llama 2** and **ChromaDB** for intelligent question-answering with context retrieval and agentic workflow. The model dynamically determines whether external document context is needed before answering queries, ensuring accurate and contextually relevant responses.

## Features
- **Llama 2 Integration**: Uses Llama 2 (7B) for high-quality text generation.
- **PDF Document Processing**: Extracts and splits content from a PDF document.
- **Sentence Embedding Model**: Employs `all-MiniLM-L6-v2` from `sentence-transformers` for embedding computation.
- **ChromaDB Vector Store**: Stores and retrieves document embeddings for semantic search.
- **Dynamic Context Retrieval**: Determines whether external context is necessary before generating an answer.
- **Two Answer Modes**:
  - **With Agentic RAG**: Decides to retrieve relevant document context or not before answering.
  - **Without RAG**: Generates an answer based solely on the query.

## Installation
Ensure the required dependencies are installed before running the notebook:
```sh
pip install pypdf==5.3.0 
```

## Usage
### 1. Model Setup
- Downloads **Llama 2 (7B)** if not already available.
- Initializes `LlamaCpp` for using the model.

### 2. PDF Document Loading & Processing
- Loads a PDF (`AIStudioDoc.pdf`).
- Splits it into manageable text chunks (~500 characters each).

### 3. Embedding Generation & Vector Store Initialization
- Computes embeddings for each chunk using `SentenceTransformer`.
- Stores embeddings in **ChromaDB** for efficient retrieval.

### 4. Answer Generation Workflow
#### **With Agentic RAG** (Retrieve and Generate)
1. Determines if external document context is required using `needs_context()`.
2. If context is needed, retrieves relevant chunks via `vector_search_tool()`.
3. Combines retrieved context with the user query and generates an answer.

#### **Without RAG** (Direct Query Response)
1. Directly generates an answer without retrieving external context.

### 5. Running Queries
Example queries to test the system:
```python
query = "What are the key features of Z by HP AI Studio?"
print(generate_answer_with_agentic_rag(query))  # Uses RAG
print(generate_answer_without_rag(query))  # Direct response
```

## File Structure
```
├── model/                # Directory for storing Llama 2 model
├── data/                 # Directory containing PDF document
│   ├── AIStudioDoc.pdf   # Sample document for retrieval
├── chroma_db/            # ChromaDB persistent storage
├── README.md             # Readme file for the Agentic RAG Demo
├── agentic_rag.ipynb     # Main notebook file
├── vanilla_rag.ipynb     # vanilla rag notebook file
├── InstallationGuide_RAG.pdf # installation guide
├── requirements.txt      # requirement file for vanilla_rag.ipynb notebook
```

## Key Functions
| Function Name                    | Description |
|----------------------------------|-------------|
| `needs_context(query)`           | Determines if external context is required. |
| `vector_search_tool(query)`      | Retrieves the most relevant text chunks. |
| `generate_answer_with_agentic_rag(query)` | Answers the query using retrieved context. |
| `generate_answer_without_rag(query)` | Answers the query without retrieval. |

## Conclusion
This Agentic RAG implementation ensures intelligent and context-aware responses by leveraging dynamic retrieval mechanisms. By combining **Llama 2**, **sentence embeddings**, and **ChromaDB**, it provides a robust foundation for knowledge-based AI assistants.

