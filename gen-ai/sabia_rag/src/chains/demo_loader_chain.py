from framework_classes.tokenizer import Tokenizer
from framework_classes.vectordb import VectorDB
from framework_classes.chain import Chain
from src.doc_loaders.sabia_doc_loader import SabIADocLoader
from src.doc_splitters.sabia_doc_splitter import SabIADocSplitter
import os

class DemoLoaderChain(Chain):
    """
        Specialization of a `Chain`.\n
        Functions:
            - start(): execute the DemoLoaderChain.
            - run(): filter documents among a list of documents, those required to be uploaded.
    """

    def __init__(self, vectordb: VectorDB = None, tokenizer: Tokenizer= None, docs_paths:list[str] = []) -> None:
        """Creates an instance with a VectorDB, Tokenizer and a list of file paths ."""
        self.db = vectordb
        self.doc_loader = SabIADocLoader()
        self.doc_splitter = SabIADocSplitter(tokenizer)
        self.docs = docs_paths
    
    def start(self) -> None:
        """
            Execute the DemoLoaderChain.\n
            Parameter:
                None
            Return:
                None
        """

        return self.run()

    def run(self) -> None:
        """
            Filter documents among a list of documents, those required to be uploaded.\n
            Parameter:
                None
            Return:
                None
        """
          
        saved = (
            set(
                [x[:x.rfind("_")] for x in self.db.collection.get(include=[])["ids"]]
            )
        )
        paths = []

        for doc in self.docs:
            filename, _ = os.path.splitext(os.path.basename(doc))
            if not filename in saved:
                paths.append(doc)

        docs = self.doc_loader.load_documents(paths)
        docs_chunks = self.doc_splitter.split_documents(docs)
        for chunks in docs_chunks:
            self.db.add(chunks)
            