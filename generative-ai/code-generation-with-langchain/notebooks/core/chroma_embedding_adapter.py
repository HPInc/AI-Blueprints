import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class ChromaEmbeddingAdapter:
    """
    Adapter class to make HuggingFaceEmbeddings compatible with both:
    
    1. ChromaDB's expected interface:
       __call__(self, input: List[str]) -> List[List[float]]
    
    2. LangChain's expected interface:
       embed_query(self, text: str) -> List[float]
       embed_documents(self, texts: List[str]) -> List[List[float]]
    
    This adapter bridges the gap between these interfaces.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize with a HuggingFaceEmbeddings instance or similar model
        that provides embed_documents and embed_query methods
        
        Args:
            embedding_model: A model instance with embedding methods
        """
        self.embedding_model = embedding_model
        logger.info(f"Initialized ChromaEmbeddingAdapter with model: {type(embedding_model).__name__}")
        
        # Default dimension for all-MiniLM-L6-v2
        self.default_embedding_dim = 384
        self.default_embedding = [0.0] * self.default_embedding_dim
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Make the adapter callable with the signature expected by ChromaDB
        
        Args:
            input: List of strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        if not input:
            logger.warning("Empty input provided to ChromaEmbeddingAdapter.__call__. Returning empty list.")
            return []
            
        try:
            # Call the underlying embedding model's embed_documents method
            embeddings = self.embedding_model.embed_documents(input)
            
            # Validate the returned embeddings
            if embeddings is None or len(embeddings) == 0:
                logger.warning("Embedding model returned None or empty list. Using zeros.")
                return [self.default_embedding] * len(input)
                
            # Check for None values in any of the embeddings
            valid_embeddings = []
            for i, emb in enumerate(embeddings):
                if emb is None or len(emb) == 0 or any(val is None for val in emb):
                    logger.warning(f"Invalid embedding at index {i}. Using zeros.")
                    valid_embeddings.append(self.default_embedding)
                else:
                    valid_embeddings.append(emb)
                    
            return valid_embeddings
        except Exception as e:
            logger.error(f"Error in ChromaEmbeddingAdapter.__call__: {str(e)}")
            # Return default embeddings instead of raising
            logger.warning(f"Returning default embeddings for {len(input)} inputs")
            return [self.default_embedding] * len(input)
            
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using the underlying embedding model.
        This method is required by LangChain's vectorstore interfaces.
        
        Args:
            text: String to embed
            
        Returns:
            Single embedding vector as list of floats
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided to embed_query. Using zeros.")
            return self.default_embedding
            
        try:
            # Call the underlying embedding model's embed_query method
            embedding = self.embedding_model.embed_query(text)
            
            # Validate the returned embedding
            if embedding is None or len(embedding) == 0 or any(val is None for val in embedding):
                logger.warning("Embedding model returned invalid embedding for query. Using zeros.")
                return self.default_embedding
                
            return embedding
        except Exception as e:
            logger.error(f"Error in ChromaEmbeddingAdapter.embed_query: {str(e)}")
            # Return default embedding instead of raising
            return self.default_embedding
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents using the underlying embedding model.
        This method is used by LangChain and matches the signature expected.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Simply delegate to the __call__ method which has the same signature
        return self.__call__(texts)