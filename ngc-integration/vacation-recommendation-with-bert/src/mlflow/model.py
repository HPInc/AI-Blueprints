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

    def __init__(self, embeddings_path: str, corpus_path: str, tokenizer_dir: str, bert_model_path: str, **kwargs):
        """
        Direct dependency injection - no MLflow context.
        Initialize all components that were in load_context method.
        
        Args:
            embeddings_path: Path to precomputed embeddings CSV file
            corpus_path: Path to corpus data CSV file  
            tokenizer_dir: Path to BERT tokenizer directory
            bert_model_path: Path to pre-trained BERT model
        """
        # Local import: keeps the module-level namespace clean
        from nemo.collections.nlp.models import BERTLMModel
        
        # Load precomputed embeddings and corpus data
        self.embeddings_df = pd.read_csv(embeddings_path)
        self.corpus_df = pd.read_csv(corpus_path)
        
        # Load tokenizer for BERT
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained BERT model
        self.bert_model = BERTLMModel.restore_from(bert_model_path, strict=False).to(self.device)
        
        logger.info(f"Model initialized with device: {self.device}")
        logger.info(f"Embeddings shape: {self.embeddings_df.shape}")
        logger.info(f"Corpus shape: {self.corpus_df.shape}")

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