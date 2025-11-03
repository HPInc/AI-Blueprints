import numpy as np
from transformers import AutoTokenizer
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = "Hi"
inputs = tokenizer(
    text, return_tensors="np", padding="max_length", truncation=True, max_length=8
)

input_ids = np.ascontiguousarray(inputs["input_ids"].astype(np.int64))
attention_mask = np.ascontiguousarray(inputs["attention_mask"].astype(np.int64))

# --- Connect to Triton via gRPC ---
triton_client = grpcclient.InferenceServerClient(
    url="localhost:8001"
)  # default gRPC 8001

# --- Create inputs ---
input0 = grpcclient.InferInput(
    "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
)
input1 = grpcclient.InferInput(
    "attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)
)

input0.set_data_from_numpy(input_ids)
input1.set_data_from_numpy(attention_mask)

# --- Create output (receive FP16) ---
output = grpcclient.InferRequestedOutput("logits")

# --- Do the inference ---
response = triton_client.infer(
    model_name="nemotron_model", inputs=[input0, input1], outputs=[output]
)

# --- Get the result ---
logits = response.as_numpy("logits")
print("logits shape:", logits.shape)
print(logits)
