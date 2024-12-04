from framework_classes.embeddings_model import EmbeddingsModel
from framework_classes.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

class SabIAEmbeddingsModelHf(EmbeddingsModel):
    def __init__(self, tokenizer: Tokenizer) -> None:
        """Creates an instance with a Tokenizer and initializes the Sentence Transformer model."""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = tokenizer

    def embed(self, text: str) -> list[float]:
        """
        Generates the embeddings representation of a string of tokens.
        Parameter:
            A text string of concatenated tokens
        Return:
            A float list representing the embeddings
        """
        if not text:
            # Handle empty or invalid text input gracefully
            return []

        # Assuming the tokenizer here is for preprocessing and it's necessary
        tokens = self.tokenizer.encode(text)
        
        # Ensure the output is in the correct format for the model
        processed_text = " ".join(tokens) if isinstance(tokens, list) else tokens
        
        # Generate embeddings and ensure the output is a list of floats
        embeddings = self.model.encode(processed_text, convert_to_numpy=True)
        
        # Convert numpy array to a list of floats
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
