import requests
import numpy as np
from transformers import MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

text = "my mom always said life like a box of chocolates never know what you're going to get"

tokens = tokenizer(text, return_tensors="np", padding=True)
input_ids = tokens["input_ids"].astype(np.int64)
attention_mask = tokens["attention_mask"].astype(np.int64)

decoder_input_ids = np.array([[pad_token_id]], dtype=np.int64)
decoder_attention_mask = np.array([[1]], dtype=np.int64)

max_length = 50
output_tokens = []

for step in range(max_length):
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": input_ids.shape,
                "datatype": "INT64",
                "data": input_ids.tolist()
            },
            {
                "name": "attention_mask",
                "shape": attention_mask.shape,
                "datatype": "INT64",
                "data": attention_mask.tolist()
            },
            {
                "name": "decoder_input_ids",
                "shape": decoder_input_ids.shape,
                "datatype": "INT64",
                "data": decoder_input_ids.tolist()
            },
            {
                "name": "decoder_attention_mask",
                "shape": decoder_attention_mask.shape,
                "datatype": "INT64",
                "data": decoder_attention_mask.tolist()
            }
        ],
        "outputs": [
            {
                "name": "logits"
            }
        ]
    }

    response = requests.post("http://localhost:8000/v2/models/Helsinki-NLP/infer", json=payload)

    if response.status_code != 200:
        print("Erro:", response.status_code)
        print(response.text)
        break

    result = response.json()
    logits = np.array(result["outputs"][0]["data"]).reshape(result["outputs"][0]["shape"])

    next_token_id = int(np.argmax(logits[0, -1]))
    output_tokens.append(next_token_id)

    if next_token_id == eos_token_id:
        break

    decoder_input_ids = np.append(decoder_input_ids, [[next_token_id]], axis=1)
    decoder_attention_mask = np.ones_like(decoder_input_ids)

translated = tokenizer.decode(output_tokens, skip_special_tokens=True)
print("Translation:", translated)
