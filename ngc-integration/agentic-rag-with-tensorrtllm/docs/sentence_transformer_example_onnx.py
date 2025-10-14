import numpy as np
from transformers import AutoTokenizer
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

# Initialize the gRPC client
client = grpcclient.InferenceServerClient(
    url="localhost:8001"
)  # default gRPC port 8001

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
text = "What is AI Studio?"
inputs = tokenizer(
    text, return_tensors="np", padding="max_length", truncation=True, max_length=128
)

# Create input tensors for Triton
input_ids = InferInput("input_ids", inputs["input_ids"].shape, "INT64")
attention_mask = InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")

# Fill the tensors with numpy data
input_ids.set_data_from_numpy(inputs["input_ids"].astype(np.int64))
attention_mask.set_data_from_numpy(inputs["attention_mask"].astype(np.int64))

# Define the output we want
output = InferRequestedOutput("last_hidden_state")

# Do the inference
response = client.infer(
    model_name="embedding_model", inputs=[input_ids, attention_mask], outputs=[output]
)

# Get the result
embedding = response.as_numpy("last_hidden_state")
print("Embedding shape:", embedding)
