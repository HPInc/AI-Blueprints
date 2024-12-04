from framework_classes.embeddings_model import EmbeddingsModel
from framework_classes.tokenizer import Tokenizer
from langchain_community.embeddings import LlamaCppEmbeddings

class SabIAEmbeddingsModel(EmbeddingsModel):
    """
        Specialization of a `EmbeddingsModel`.\n
        Functions:
            - embed(): generates the embeddings representation of a string of tokens.
    """

    def __init__(self, model_path, tokenizer: Tokenizer) -> None:
        """Creates an instance with a Tokenizer."""
        self.model = LlamaCppEmbeddings(
            model_path=model_path,
            n_gpu_layers=16,
            n_ctx=2048,
            n_batch=128,
            f16_kv=True
            )
        self.tokenizer = tokenizer

    def embed(self, text: str) -> list[float]:
        """
            Generates the embeddings representation of a string of tokens.\n
            Parameter:
                A text string of concatenated tokens
            Return:
                A float list of tokens
        """
        print("x.x.x. Started embedding .x.x.x")
        tokens = self.tokenizer.encode(text)
        print("x.x.x. Tokenized .x.x.x")
        embeddings = self.model.embed_query(" ".join(tokens))
        return embeddings