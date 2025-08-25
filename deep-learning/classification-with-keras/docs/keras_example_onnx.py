#!/usr/bin/env python3
"""
Simple Triton Inference Script for MNIST Keras Model

Este script:
- Usa uma variável BASE_64 para receber a imagem MNIST (28x28, escala de cinza, PNG ou JPEG) codificada em base64.
- Faz a inferência no Triton usando apenas essa imagem.
- Mostra o resultado da predição.
"""

import numpy as np
import requests
import json
from PIL import Image
import base64
import io

# Change for the BASE_64 of your choice
BASE_64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APn+prW0uL66itbSCSe4lbbHFEpZmPoAOtaWt+FtZ8Ox28mqWghS4LCNlmSQblxuU7GO1huGQcHnpWPRXoOiWF/pfhiwh0K2ln8R+JfMWNoh89vaK2w7f7pdg2Wzwq9sk1X+IY03SItJ8JabMLk6QsjX1wpOJLuQr5gHsuxQP/rVw1Fen+EfFmueF/AN3qkup3C2yMbPR7QkBXmbJkk9SsYOcfd3MK8yd3lkaSRizsSzMxyST1JptFXLnVb280+ysJ5y9rYhxbx4AEe9tzdBzk9z7elU6K//2Q=="

# Endpoint Triton
TRITON_ENDPOINT = "http://localhost:8000/v2/models/keras/infer"


def image_from_base64(b64_string: str) -> np.ndarray:
    """Decode base64 to array shape (1,28,28,1)"""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.reshape(1, 28, 28, 1)

def run_triton_inference(endpoint: str, image: np.ndarray):
    """Executa inferência no Triton para uma única imagem."""
    print("🚀 MNIST Triton Inference (BASE64)")
    print("=" * 40)
    print(f"🌐 Endpoint: {endpoint}")
   
    try:
        server_url = endpoint.split('/v2/')[0]
        health_response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Triton server not ready. Status: {health_response.status_code}")
            return
        print("✅ Triton server is ready")
    except requests.RequestException as e:
        print(f"❌ Cannot connect to Triton server: {e}")
        print("Please ensure Triton server is running on the specified endpoint")
        return

    # Prepare Triton inference request
    try:
        inputs = [{
            "name": "input",
            "shape": list(image.shape),
            "datatype": "FP32",
            "data": image.flatten().tolist()
        }]
        outputs = [{"name": "output"}]
        request_data = {"inputs": inputs, "outputs": outputs}

        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ Inference failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return

        result = response.json()
        probabilities = np.array(result["outputs"][0]["data"])
        print("🏆 Probabilities:", probabilities)

    except Exception as e:
        print(f"❌ Inference error: {e}")

if __name__ == "__main__":
    b64_clean = BASE_64.strip().replace('\n', '').replace('\r', '')
    image = image_from_base64(b64_clean)
    run_triton_inference(TRITON_ENDPOINT, image)