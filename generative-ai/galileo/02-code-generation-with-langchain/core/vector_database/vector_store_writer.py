import logging
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreWriter:
    def __init__(self, collection_name="my_collection", verbose=False, embedding_model=None):
        """
        Initializes a ChromaDB collection.

        :param collection_name: Name of the collection to upsert data into
        :param verbose: If True, log detailed info for each record
        :param embedding_model: Optional embedding model to use with ChromaDB (if supported)
        """
        self.verbose = verbose
        self.embedding_model = embedding_model
        
        persist_dir = "./chroma_db"
        import os
        os.makedirs(persist_dir, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB client initialized with persistent storage at {persist_dir}")
        except Exception as e:
            logger.warning(f"Error creating persistent client: {str(e)}, trying in-memory client")
            self.client = chromadb.Client()
            logger.info("Using in-memory client as fallback")
        
        # Collection configuration
        collection_kwargs = {"name": collection_name}
        
        # Handle embedding function differently - don't pass to ChromaDB directly
        # This avoids interface incompatibilities between LangChain and ChromaDB
        
        try:
            self.collection = self.client.get_or_create_collection(**collection_kwargs)
            logger.info(f"ChromaDB collection '{collection_name}' initialized with persistent storage.")
        except TypeError as e:
            logger.warning(f"Error creating collection: {str(e)}")
            # Fallback for older ChromaDB versions
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' initialized with fallback method.")

    def upsert_dataframe(self, df):
        """
        Extracts data from a DataFrame and upserts into the vector store.

        :param df: DataFrame with 'ids', 'code', 'metadatas', 'embeddings' columns
        """
        # Validate that we have the required columns
        required_columns = ["ids", "code", "metadatas", "embeddings"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' in DataFrame")
                raise ValueError(f"DataFrame must have '{col}' column")

        # Check for None values in embeddings and filter/replace them
        # TODO: Investigate if this is necessary, as ChromaDB should handle None values
        default_embedding = [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        valid_indices = []
        
        # Create lists for valid data
        valid_ids = []
        valid_documents = []
        valid_metadatas = []
        valid_embeddings = []
        
        # Process each row and filter out invalid entries
        for i, row in df.iterrows():
            embedding = row["embeddings"]
            if embedding is None or (isinstance(embedding, list) and (len(embedding) == 0 or any(e is None for e in embedding))):
                logger.warning(f"Invalid embedding for ID {row['ids']} - replacing with zeros")
                embedding = default_embedding
            
            # Add valid data to our lists
            valid_ids.append(str(row["ids"]))  # Ensure IDs are strings
            valid_documents.append(row["code"])
            valid_metadatas.append(row["metadatas"])
            valid_embeddings.append(embedding)
        
        # If no valid data after filtering, return
        if not valid_ids:
            logger.warning("No valid data after filtering - skipping upsert")
            return

        if self.verbose:
            for i in range(len(valid_ids)):
                logger.info(f"[UPDATING] ID: {valid_ids[i]}")
                logger.info(f"[UPDATING] Document: {valid_documents[i]}")
                logger.info(f"[UPDATING] Metadata: {valid_metadatas[i]}")
                logger.info(f"[UPDATING] Embedding: {valid_embeddings[i][:5]}...\n")  # Show only first 5 values

        # Log the shapes for debugging
        logger.info(f"Upserting {len(valid_ids)} documents")
        
        try:
            self.collection.upsert(
                documents=valid_documents,
                ids=valid_ids,
                metadatas=valid_metadatas,
                embeddings=valid_embeddings
            )
            logger.info("✅ Documents upserted successfully into ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            # Try again without embeddings if there's an error
            try:
                logger.warning("Attempting to upsert without embeddings (ChromaDB will generate them)")
                self.collection.upsert(
                    documents=valid_documents,
                    ids=valid_ids,
                    metadatas=valid_metadatas
                )
                logger.info("✅ Documents upserted successfully without embeddings.")
            except Exception as e2:
                logger.error(f"Failed again: {str(e2)}")
                raise

        logger.info("✅ Documents upserted successfully into ChromaDB.")
