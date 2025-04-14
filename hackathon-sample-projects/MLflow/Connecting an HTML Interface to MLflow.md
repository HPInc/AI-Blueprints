# Connecting an HTML Interface to MLflow Model Deployments

## Introduction

MLflow provides a powerful platform for deploying machine learning models as REST APIs. This guide explains how to create an HTML interface that interacts with your MLflow-deployed model.

## Understanding Your Model's API

Before building your HTML interface, you need to understand how to communicate with your MLflow model.

### 1. Inspect Your Model Signature

The model signature defines the expected inputs and outputs:

```python
# Example of how your model signature might be defined
signature = infer_signature(
    model_input={"query": ["example movie query"]},
    model_output=[{"title": "Movie 1", "similarity": 0.95}]
)
```

To view this signature:
- Check the MLflow UI (navigate to your model)
- Or use code: `mlflow.models.get_model_info("models:/Netflix_Similarity/latest").signature`

### 2. Test the API with Python First

Before building your HTML page, test the API directly:

```python
import requests
import json

# Test with a simple query
test_payload = {
    "dataframe_split": {
        "columns": ["query"],
        "data": [["action movies with explosions"]]
    },
    "params": {
        "top_n": 5,
        "embedding_index": 0
    }
}

response = requests.post(
    "http://localhost:5000/invocations", 
    headers={"Content-Type": "application/json"},
    data=json.dumps(test_payload)
)

print("Status:", response.status_code)
print("Response:", response.json())
```

### 3. Common MLflow Request Formats

MLflow models typically expect one of these formats:

```json
{
  "dataframe_split": {
    "columns": ["query"],
    "data": [["your movie search query"]]
  },
  "params": {
    "top_n": 5,
    "embedding_index": 0
  }
}
```

Or sometimes:

```json
{
  "instances": [
    {"query": "your movie search query"}
  ],
  "parameters": {
    "top_n": 5,
    "embedding_index": 0
  }
}
```

### 4. Response Structure

The response depends on your model implementation, but might look like:

```json
[
  {
    "title": "The Matrix",
    "genre": "Sci-Fi/Action", 
    "similarity": 0.89
  },
  {
    "title": "Inception",
    "genre": "Sci-Fi/Thriller",
    "similarity": 0.76
  }
]
```

## Building Your HTML Interface

### 1. Create the Basic HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netflix Content Recommender</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .input-section {
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .loading {
            display: none;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Netflix Content Recommender</h1>
    
    <div class="input-section">
        <label for="query">What would you like to watch?</label>
        <input type="text" id="query" placeholder="e.g., sci-fi with robots">
        <button onclick="getSimilarContent()">Find Similar Content</button>
        <div class="loading" id="loading">Loading...</div>
    </div>
    
    <div id="results"></div>
    
    <script>
        // JavaScript will go here
    </script>
</body>
</html>
```

### 2. Add JavaScript to Connect to the MLflow API

```javascript
function getSimilarContent() {
    const query = document.getElementById('query').value;
    if (!query) {
        alert("Please enter a search query");
        return;
    }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';
    
    // Prepare the request payload based on your model's expected format
    const payload = {
        "dataframe_split": {
            "columns": ["query"],
            "data": [[query]]
        },
        "params": {
            "top_n": 5,
            "embedding_index": 0
        }
    };
    
    // Send request to MLflow model server
    fetch('http://localhost:5000/invocations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = 
            `<p style="color:red">Error: ${error.message}</p>`;
    })
    .finally(() => {
        document.getElementById('loading').style.display = 'none';
    });
}

function displayResults(data) {
    // Create results table based on your model's response format
    let html = '<h2>Similar Content</h2>';
    html += '<table>';
    html += '<tr><th>Title</th><th>Genre</th><th>Similarity</th></tr>';
    
    data.forEach(item => {
        html += `<tr>
            <td>${item.title || 'N/A'}</td>
            <td>${item.genre || 'N/A'}</td>
            <td>${item.similarity ? (item.similarity * 100).toFixed(1) + '%' : 'N/A'}</td>
        </tr>`;
    });
    
    html += '</table>';
    document.getElementById('results').innerHTML = html;
}
```

## Handling CORS Issues

MLflow's serving doesn't support CORS by default, which can block browser requests. Options to solve this:

### Option 1: Enable MLserver

```bash
mlflow models serve -m "models:/Netflix_Similarity/latest" -p 5000 --enable-mlserver
```

### Option 2: Use a Proxy Server

Create a simple proxy with Flask:

```python
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    # Forward the request to MLflow server
    mlflow_response = requests.post(
        'http://localhost:5000/invocations',
        headers={'Content-Type': 'application/json'},
        data=request.data
    )
    return jsonify(mlflow_response.json())

if __name__ == '__main__':
    app.run(port=8000)
```

Then update your fetch URL to `http://localhost:8000/predict`.

## Troubleshooting

### Common Issues:

1. **CORS Errors**:
   - Error: "Access to fetch has been blocked by CORS policy"
   - Solution: Use a proxy server as described above

2. **Incorrect Request Format**:
   - Error: "Invalid input format"
   - Solution: Double-check your model signature and adjust the request payload

3. **Model Not Running**:
   - Error: "Failed to establish a connection"
   - Solution: Verify your MLflow model server is running with `mlflow models serve`

4. **Wrong Response Format**:
   - Issue: JavaScript errors when processing response
   - Solution: Console.log the response and adjust your displayResults function

### Debugging Tips:

- Use your browser's developer tools (F12) to inspect network requests and responses
- Test the API with Postman or curl before implementing in HTML/JavaScript
- Log both the request and response in your JavaScript code

## Complete Example

For a complete working example, combine all the HTML and JavaScript code above into a single file named `index.html`, and ensure your MLflow model is running at `http://localhost:5000`.

