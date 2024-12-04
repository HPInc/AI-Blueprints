# CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python=0.2.55 --no-cache-dir --force-reinstall

import pip
from os import environ

environ["CUDACXX"] = "/usr/local/cuda-12/bin/nvcc"
environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
environ["FORCE_CMAKE"] = "1"

pip.main(["install", "llama-cpp-python==0.2.55", "--force-reinstall"])
pip.main(["install", "langchain", "tiktoken", "chromadb", "PyMuPDF"])

from src.tokenizers.sabia_tokenizer import SabIATokenizer
from src.models.sabia_embeddings_model import SabIAEmbeddingsModel
from src.vectordbs.chroma_vectordb import ChromaVectorDB
from src.chains.demo_loader_chain import DemoLoaderChain
from src.chains.demo_query_chain import DemoQueryChain
import time

timepoints = []
timepoints.append(time.time())
print("-------- Starting SabIA modules ---------")
tokenizer = SabIATokenizer()
embedding = SabIAEmbeddingsModel(tokenizer=tokenizer)
vectordb = ChromaVectorDB(embedding_model=embedding)

timepoints.append(time.time())
print("-------- Embeddings Loaded ---------")

docs_paths = ["./docs/AIStudioDoc.pdf"]
demo_loader_chain = DemoLoaderChain(vectordb=vectordb, tokenizer=tokenizer, docs_paths=docs_paths)
demo_loader_chain.run()

timepoints.append(time.time())
print("-------- Loader chain finished ---------")

demo_query_chain = DemoQueryChain(vectordb=vectordb)
timepoints.append(time.time())

demo_query_chain.start()


print("-------- Query chain loaded ---------")


