from framework_classes.doc_loader import DocLoader
from framework_classes.document import Document
import fitz
import os

class SabIADocLoader(DocLoader):
    """
        Specialization of a `DocLoader` that uses the PyMuPDF library.\n
        Functions:
            - load_document(): load one document.
            - load_documents(): load one or more documents through the call of the load_document function.
    """

    def __init__(self) -> None:
        """Creates an instance  ."""
        self.loader = fitz

    def load_document(self, doc_path: str = '') -> Document:
        """
            Open a document and extract its content and metadata.\n
            Parameter:
                A document file path.
            Return:
                A Document object.
        """
        doc = self.loader.open(doc_path)
        filename, _ = os.path.splitext(os.path.basename(doc_path))
        metadata = {}
        metadata["format"] = doc.metadata["format"]
        metadata["title"] = (doc.metadata["title"] if doc.metadata["title"]!="" else filename)
        content = ""

        for i in range(doc.page_count):
            content = content + doc.load_page(i).get_text("text")

        return Document(metadata = metadata, content = content)

    def load_documents(self, doc_paths: list[str] = []) -> list[Document]:
        """
            Load one or more documents using the load_document() function.\n
            Parameter:
                A list of document file paths.
            Return:
                A list of Document objects.
        """
        
        docs = []
        for doc_path in doc_paths:
            doc = self.load_document(doc_path)
            docs.append(doc)
        return docs