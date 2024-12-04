from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import time

content = ""
with open("content.txt", "r") as file:
    content = file.read()

n_gpu_layers = 16
n_batch = 512
n_ctx = 4096
max_tokens = 1024
stop_tags = []

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

start_time = time.time()

llm = LlamaCpp(
  model_path="/mnt/d/SabIA/LlamaChat7b/LlamaChat7b/llama-2-7b/llama-2-7b-chat/ggml-model-f16-Q5_K_M.gguf",
  n_gpu_layers=n_gpu_layers,
  n_batch=n_batch,
  n_ctx=n_ctx,
  max_tokens=max_tokens,
  f16_kv=True,  
  callback_manager=callback_manager,
  verbose=False,
  stop=stop_tags,
  streaming=False,
  temperature=0.4,
)
    
load_time = time.time() - start_time

output = llm(content, echo=True)

total_time = time.time() - start_time

print(output)

print(f"Total time: {total_time}")
print(f" - Loading time: {load_time}")
print(f" - Loading time: {total_time - load_time}")