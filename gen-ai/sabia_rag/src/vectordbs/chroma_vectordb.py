from framework_classes.chunk import Chunk
from framework_classes.vectordb import VectorDB
from framework_classes.embeddings_model import EmbeddingsModel

from typing import Any
import chromadb

class ChromaVectorDB(VectorDB):
    """
        Specialization of a `VectorDB` that uses the ChromaDB library.\n
        Functions:
            - add(): adds a text string and its metadata to the vector database.
            - query(): similarity search in the vector database, based on a text string.
    """

    def __init__(self, embedding_model: EmbeddingsModel, collection_name: str = "SabIA_collection", distance_function: str = None) -> None:
        """Creates an instance with a connection to a colletion from the vectorDB. If the collection does not exist, creates a new one."""

        self.embbeding = embedding_model
        chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path="./data/vectordb_data", settings=chroma_settings)
        self.collection = self.client.get_or_create_collection(
            collection_name, 
            metadata={
                "hnsw:space": (distance_function if distance_function else "cosine")
            }
        )


    def add(self, chunks: list[Chunk]) -> None:
        """
            Add a text string and its metadata
            to the vector database\n
            Parameter:
                A text string of content and its metadata (optional)
            Return:
                None
        """

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.metadata["title"]+"_"+str(chunk.index))
            embeddings.append(self.embbeding.embed(chunk.content))
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )


    def query(self, text: str, filters:dict = None) -> Any:
        """
            Similarity search in the vector database, based
            on a text string.
             It's possible to use filters in the search (example: 'age=20').\n
            Parameter:
                A text string of content and filters (optional)
            Return:
                A list of most similar registers to the input
        """
        embedded_text = self.embbeding.embed(text)
        data = self.collection.query(embedded_text, where=filters, n_results=5)
        chunks = []

        for i in range(len(data["ids"][0])):
            id = data["ids"][0][i]
            chunks.append(Chunk(content=data["documents"][0][i], metadata=data["metadatas"][0][i], index=id[:id.rfind("_")]))

        return chunks