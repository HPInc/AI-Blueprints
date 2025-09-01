# src/mlflow/model.py
# -*- coding: utf-8 -*-

import os
import sys
import logging
from pathlib import Path

# Data manipulation libraries
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Deep learning framework
import torch  

# NLP libraries
import nltk  # Natural Language Toolkit
from transformers import AutoTokenizer  # Tokenizer for transformer-based models

# Configure logging
logger = logging.getLogger("bert_tourism_model")

class BERTModelWithHiddenStates(torch.nn.Module):
    """PyTorch wrapper for BERT models that use **kwargs. Creates wrapper for torch conversion."""
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs

        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding


class Model:
    """
    Standalone BERT Tourism Recommendation Model containing all business logic.
    NO MLflow inheritance - pure domain functionality for vacation recommendation with BERT.
    """

    def __init__(self, config: dict, docs_path: str, model_path: str = None, secrets: dict = None):
        """
        Initialize the Model with configuration and artifacts.
        Adapted to work with generic loader that provides docs_path containing BERT artifacts.
        
        Args:
            config: Model configuration dictionary
            docs_path: Path to documents/data directory containing BERT artifacts:
                      - embeddings.csv: Precomputed embeddings
                      - corpus.csv: Tourism corpus data  
                      - tokenizer/: BERT tokenizer directory
            model_path: Path to BERT model file (optional)
            secrets: Dictionary containing secrets (optional)
        """
        self.model_config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets
        
        # Resolve BERT-specific artifact paths from generic docs_path structure
        self.embeddings_path = os.path.join(docs_path, "embeddings.csv")
        self.corpus_path = os.path.join(docs_path, "corpus.csv") 
        self.tokenizer_dir = os.path.join(docs_path, "tokenizer")
        
        # Validate required BERT artifact files
        required_files = {
            "embeddings": self.embeddings_path,
            "corpus": self.corpus_path,
            "tokenizer": self.tokenizer_dir,
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name.capitalize()} not found at: {path}")
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.chain = None
        self.prompt = None
        self.prompt_str = ""
        self.memory = []
        self.callback_manager = None
        
        # Setup environment and load components
        try:
            self._setup_environment()
            self._load_bert_artifacts()
            
            logger.info("BERT Tourism Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BERT Tourism Model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"BERT Tourism Model initialization failed: {str(e)}") from e

    def _setup_environment(self) -> None:
        """Configure environment variables based on configuration and secrets."""
        try:
            # Load secrets into environment if provided
            if self.secrets:
                for key, value in self.secrets.items():
                    os.environ[key] = str(value)
                logger.info("Secrets loaded into environment")
            
            # Configure proxy if specified in config
            if "proxy" in self.model_config and self.model_config["proxy"]:
                logger.info(f"Setting up proxy: {self.model_config['proxy']}")
                os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
                os.environ["HTTP_PROXY"] = self.model_config["proxy"]
            else:
                logger.info("No proxy configuration found")
                    
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function
    
    def _load_bert_artifacts(self) -> None:
        """Load BERT-specific artifacts: embeddings, corpus, tokenizer, and model."""
        try:
            # Local import: keeps the module-level namespace clean
            from nemo.collections.nlp.models import BERTLMModel
            
            # Load precomputed embeddings and corpus data
            self.embeddings_df = pd.read_csv(self.embeddings_path)
            self.corpus_df = pd.read_csv(self.corpus_path)
            logger.info(f"Loaded embeddings: {self.embeddings_df.shape}")
            logger.info(f"Loaded corpus: {self.corpus_df.shape}")
            
            # Load tokenizer for BERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
            logger.info(f"Loaded BERT tokenizer from: {self.tokenizer_dir}")
            
            # Set device to GPU if available, otherwise use CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load pre-trained BERT model
            if self.model_path and os.path.exists(self.model_path):
                bert_model_path = self.model_path
            else:
                # Fallback to default path from config
                bert_model_path = self.model_config.get("model_path", "/home/jovyan/datafabric/Bertlargeuncased/bertlargeuncased.nemo")
            
            self.bert_model = BERTLMModel.restore_from(bert_model_path, strict=False).to(self.device)
            logger.info(f"Loaded BERT model from: {bert_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT artifacts: {str(e)}")
            raise
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate BERT embeddings for the input query.
        
        Args:
            query: Input text query for similarity search
            
        Returns:
            NumPy array containing the [CLS] token embedding
        """
        self.bert_model.eval()  # Set model to evaluation mode
        
        # Tokenize the input query and move tensors to the selected device
        encoded_input = self.tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=128
        )
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        
        # Get the model's output embedding
        with torch.no_grad():
            output = self.bert_model.bert_model(**encoded_input)
        
        # Return the [CLS] token embedding as a NumPy array
        return output[:, 0, :].cpu().numpy()

    def predict(self, model_input: Dict[str, Any], params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Core business logic for vacation recommendation.
        Compute similarity between query and precomputed embeddings,
        then return the top 5 most similar results.
        
        Args:
            model_input: Dictionary containing query information
            params: Optional parameters (for compatibility)
            
        Returns:
            List of dictionaries containing vacation recommendations with similarity scores
        """
        # Extract the query string from model input
        query = model_input["query"][0]
        
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Compute cosine similarity between query and precomputed embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings_df.values)
        
        # Get indices of top 5 most similar results
        top_indices = np.argsort(similarities[0])[::-1][:5]
        
        # Retrieve corresponding results from the corpus
        results = self.corpus_df.iloc[top_indices].copy()
        results.loc[:, 'Similarity'] = similarities[0][top_indices]
        
        logger.info(f"Found {len(results)} recommendations")
        
        # Return results as a list of dictionaries
        return results.to_dict(orient="records")
        
    def get_wrapped_model_for_onnx(self) -> BERTModelWithHiddenStates:
        """
        Get wrapped PyTorch model for ONNX export.
        This method supports the ONNX export functionality.
        
        Returns:
            Wrapped BERT model ready for ONNX conversion
        """
        return BERTModelWithHiddenStates(self.bert_model)
        
    def get_onnx_export_config(self):
        """
        Get configuration for ONNX export.
        Returns the configuration needed for ONNX model export.
        """
        # Import here to avoid circular imports
        from ..onnx_utils import ModelExportConfig
        
        device = self.device
        wrapped_model = self.get_wrapped_model_for_onnx()
        
        # Sample inputs for ONNX export
        batch_size = 1
        seq_len = 128
        vocab_size = 30522

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        return ModelExportConfig(
            model=wrapped_model,
            model_name="bert_tourism_onnx",
            input_sample=(
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device)
            ),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "embedding": {0: "batch_size"}
            },
        )