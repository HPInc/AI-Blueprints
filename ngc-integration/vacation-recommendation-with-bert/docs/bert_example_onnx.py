import requests
import numpy as np
from transformers import BertTokenizer

import json

TRITON_URL = "http://localhost:8000/v2/models/bert_tourism_onnx/infer"

query = "Suggest to me a Safari vacation"


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
encoded = tokenizer(
    query, padding=True, truncation=True, return_tensors="pt", max_length=128
)
input_ids = encoded["input_ids"].numpy().astype(np.int64)
attention_mask = encoded["attention_mask"].numpy().astype(np.int64)
token_type_ids = encoded["token_type_ids"].numpy().astype(np.int64)
payload = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": list(input_ids.shape),
            "datatype": "INT64",
            "data": input_ids.flatten().tolist(),
        },
        {
            "name": "attention_mask",
            "shape": list(attention_mask.shape),
            "datatype": "INT64",
            "data": attention_mask.flatten().tolist(),
        },
        {
            "name": "token_type_ids",
            "shape": list(token_type_ids.shape),
            "datatype": "INT64",
            "data": token_type_ids.flatten().tolist(),
        },
    ]
}

resp = requests.post(TRITON_URL, data=json.dumps(payload))
resp.raise_for_status()
result = resp.json()

embedding = np.array(result["outputs"][0]["data"], dtype=np.float32).reshape(1, -1)
print(embedding)
