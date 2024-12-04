from framework_classes.doc_splitter import DocSplitter
from framework_classes.chunk import Chunk
from framework_classes.document import Document
from framework_classes.tokenizer import Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SabIADocSplitter(DocSplitter):
    """
        Specialization of a `DocSplitter`.\n
        Functions:
            - split_document(): generate chunks from a single Document.
            - split_documents(): generate chunks from a Document list.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Creates an instance with a Tokenizer and a RecursiveCharacterTextSplitter."""
        self.tokenizer = tokenizer
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 50,
            length_function = self.tokenizer.count_tokens,
            is_separator_regex = False,
        )
    
    def split_document(self, doc: Document) -> list[Chunk]:
        """
            Generate chunks from a single Document.\n
            Parameter:
                A Document
            Return:
                A list of chunk objects, formed from the contents of a Document
        """

        index = 0
        chunks = []
        contents = self.splitter.split_text(doc.content)

        for content in contents:
            metadata = doc.metadata.copy()
            metadata["size_chunk"] = self.tokenizer.count_tokens(content)
            chunks.append(Chunk(content, metadata, index))
            index +=1
        
        return chunks
    
    def split_documents(self, docs: list[Document]) -> list[list[Chunk]]:
        """
            Generate chunks from a Document list.\n
            Parameter:
                A Document list
            Return:
                A list of lists of chunk objects, formed from the contents of a Document
        """

        chunks = []
        for doc in docs:
            chunks.append(self.split_document(doc))

        return chunks