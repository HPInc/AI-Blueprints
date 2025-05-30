# Transcript Summarization Demo

This is a React-based user interface for the `transcript-summarization` example. It allows users to upload text documents (PDF, DOC, TXT formats), extract text from them, and display the API output as a summary.

## Features

- Upload text documents (TXT, PDF, DOC/DOCX)
- Extract text from various file formats
- Submit text for AI summarization
- View both the original text and the summary
- Toggle between detailed and simple views

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
- **Simple View**: Shows only the summary output
- **Detailed View**: Shows both the original text and the generated summary side by side

## API Integration

This UI sends requests to the API endpoint provided by MLFlow. The application uses a RESTful approach to communicate with the backend:

### API Request Format

The application sends POST requests to the `/invocations` endpoint with the following JSON structure:

```json
{
  "inputs": {
    "text": ["Your document text here"]
  },
  "params": {}
}
```

### API Response Format

The API returns a JSON response with the following structure:

```json
{
  "predictions": [
    {
      "summary": "The generated summary text"
    }
  ]
}
```


### Successful Demonstration of the User Interface  

![Transcript Summarization Demo UI](docs/ui_summarization.png) 

### Implementation Details

- The application extracts text from uploaded documents (PDF, DOC, TXT) using client-side libraries
- Text extraction is handled by pdfjs-dist for PDFs and mammoth for Word documents
- The extracted text is then sent to the API for summarization
- The application handles both array and object response formats from the API
- Error handling is implemented for API connection issues and invalid responses
