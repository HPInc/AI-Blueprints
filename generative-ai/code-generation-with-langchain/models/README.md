# Models Directory

This directory contains AI models used by the code generation blueprint, organized following industry standards.

## Structure

```text
models/
├── embeddings/           # Text embedding models
│   └── all-MiniLM-L6-v2/ # Sentence transformer model for code retrieval
└── llm/                  # Large Language Models (if stored locally)
```

## Usage

- **Embedding Models**: Used for vector similarity search in code retrieval
- **LLM Models**: Used for code generation (can be local or remote)

## Model Lifecycle

1. **Development**: Models downloaded/trained and stored here during development
2. **Registration**: MLflow Logger automatically packages models from this directory
3. **Deployment**: MLflow Loader retrieves models from artifacts for inference

This structure ensures clean separation of concerns and industry-standard model organization.