# RAG Chatbot Demo

This is a React-based user interface for the `vanilla-rag-chatbot` example. It allows users to have interactive conversations with an AI chatbot that leverages Retrieval-Augmented Generation (RAG) to provide context-aware responses based on document knowledge.

## Features

- Interactive query interface for asking questions to the AI
- Dual viewing modes: Black Box (simplified) and White Box (detailed)
- Document retrieval visualization with ChromaDB vector chunks
- Prompt engineering transparency
- Interaction history tracking
- Local LLM (Llama7b) integration

## Running the Application

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Build for production:

```bash
npm run build
```

4. Preview the production build:

```bash
npm run preview
```

## Interface

The UI provides two viewing modes:
- **Black Box Mode**: Shows only the query input and the AI-generated response
- **White Box Mode**: Shows the complete RAG pipeline components:
  - ChromaDB vector database chunks
  - Interaction history
  - Prompt engineering details
  - Local LLM (Llama7b) output

Each component in the White Box view has an information icon that can be toggled to learn more about its role in the RAG pipeline.

## API Integration

This UI sends requests to the API endpoint provided by MLFlow. The application uses a RESTful approach to communicate with the backend:

### API Request Format

The application sends POST requests to the `/invocations` endpoint with the following JSON structure:

```json
{
  "inputs": {
    "question": ["Your question here"]
  },
  "params": {}
}
```

### API Response Format

The API returns a JSON response with the following structure:

```json
{
  "predictions": {
    "output": "The AI-generated response text",
    "history": "Text representation of conversation history",
    "prompt": "The prompt template used for the query",
    "chunks": ["Retrieved document chunks from the vector database"]
  }
}
```

### Implementation Details

- The application displays different components of the RAG pipeline based on the selected view
- Retrieved document chunks show which parts of the knowledge base were used for answering
- The prompt engineering section shows how the query and context are formatted for the LLM
- Full interaction history is maintained and displayed in the UI
- The local Llama7b model processes queries without requiring cloud resources
````
