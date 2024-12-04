import pip
from os import environ

environ["CUDACXX"] = "/usr/local/cuda-12/bin/nvcc"
environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
environ["FORCE_CMAKE"] = "1"

pip.main(["install", "llama-cpp-python==0.2.55", "numpy==1.23"])
pip.main(["install", "langchain==0.1.17", "tiktoken", "chromadb==0.4.24", "PyMuPDF"])

import os
import time
import mlflow
from src.tokenizers.sabia_tokenizer import SabIATokenizer
from src.models.sabia_embeddings_model import SabIAEmbeddingsModel
from src.vectordbs.chroma_vectordb import ChromaVectorDB
from src.chains.demo_loader_chain import DemoLoaderChain
from src.chains.demo_query_chain import DemoQueryChain
from framework_classes.memory import Memory
from framework_classes.message import Message
from src.models.demo_model import DemoModel
from src.prompt_templates.demo_prompt_template import DemoPromptTemplate
from src.prompts.demo_prompt import DemoPrompt

class MainApplication:
    def __init__(self):
        base_model_path = "/home/jovyan/datafabric/Llama7b"
        inference_model_path = os.path.join(base_model_path, "ggml-model-f16-Q5_K_M.gguf")

        tokenizer = SabIATokenizer()
        embedding = SabIAEmbeddingsModel(model_path=inference_model_path, tokenizer=tokenizer, n_gpu_layers=33)
        vectordb = ChromaVectorDB(embedding_model=embedding)

        self.demo_loader_chain = DemoLoaderChain(vectordb=vectordb, tokenizer=tokenizer, docs_paths=["./docs/AIStudioDoc.pdf"])
        self.llm_model = DemoModel(model_path=inference_model_path, n_gpu_layers=32, stop_tags=["</answer>"])
        self.prompt_template = DemoPromptTemplate()
        self.prompt = DemoPrompt(self.prompt_template, vectordb)
        self.memory = Memory(20)

    def run(self):
        while True: 
            user_input = input("You: ") 
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            message = Message("User", user_input)
            history = self.memory.get_history()
            content, _ = self.prompt.get_prompt(message, history)
            prediction = self.llm_model.predict(content) + " "
            start = prediction.index("Answer: <answer>") + 16
            prediction = prediction[start:prediction.find("User", start)].strip()
            self.memory.add_message(message)
            self.memory.add_message(Message("Assistant", prediction))
            print("Assistant:", prediction)

if __name__ == "__main__":
    app = MainApplication()
    app.run()
