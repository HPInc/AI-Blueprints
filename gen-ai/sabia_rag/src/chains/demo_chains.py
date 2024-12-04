from framework_classes.composed_chain import ComposedChain
from src.tokenizers.sabia_tokenizer import SabIATokenizer
from src.models.sabia_embeddings_model import SabIAEmbeddingsModel
from src.vectordbs.chroma_vectordb import ChromaVectorDB
from src.chains.demo_loader_chain import DemoLoaderChain
from src.chains.demo_query_chain import DemoQueryChain

class DemoChains(ComposedChain):
    """
        Specialization of a ComposedChain.\n
        Functions:
            - run(): implements ComposedChain's summon function.
    """

    def run(self):
        """
            Implements ComposedChain's summon function.\n
            Parameter:
                None
            Return:
                None
        """
          
        tokenizer = SabIATokenizer()
        embedding = SabIAEmbeddingsModel(tokenizer=tokenizer)
        vectordb = ChromaVectorDB(embedding_model=embedding)

        docs_paths = ["./docs/AIStudioDoc.pdf"]

        demo_loader_chain = DemoLoaderChain(vectordb=vectordb, tokenizer=tokenizer, docs_paths=docs_paths)
        demo_query_chain = DemoQueryChain()

        chains = [
            ("DemoLoaderChain", demo_loader_chain.run),
            ("DemoQueryChain", demo_query_chain.start)
        ]

        self.summon(chains)