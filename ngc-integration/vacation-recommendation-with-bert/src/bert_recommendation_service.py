# src/bert_recommendation_service.py
# -*- coding: utf-8 -*-

import sys
import os  
import logging
from datetime import datetime
import warnings
from pathlib import Path

# Data manipulation libraries
import pandas as pd
import numpy as np
from tabulate import tabulate
from typing import Any, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Deep learning framework
import torch  

# NLP libraries
import nltk  # Natural Language Toolkit
from transformers import AutoTokenizer  # Tokenizer for transformer-based models
from transformers import logging as hf_logging
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec, ParamSchema, ParamSpec

# Configure loggin
logger = logging.getLogger("register_model_logger")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src directory to path to ensure onnx_utils is found
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from onnx_utils import ModelExportConfig, log_model

class BERTModelWithHiddenStates(torch.nn.Module): #Pytorch models that use **kwargs needs to create wrapper, we need more work on that
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


class BERTTourismModel(mlflow.pyfunc.PythonModel):
    # ── make the *empty* instance pickle-safe ──────────────────────────────
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass
        
    def load_context(self, context):
        """
        Load precomputed embeddings, corpus, and the pre-trained BERT model.
        """
         # local import: keeps the module-level namespace pickle-safe
        from nemo.collections.nlp.models import BERTLMModel
        
        # Load precomputed embeddings and corpus data
        self.embeddings_df = pd.read_csv(context.artifacts['embeddings_path'])
        self.corpus_df = pd.read_csv(context.artifacts['corpus_path'])
        
        # Load tokenizer for BERT
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer_dir"])
        
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained BERT model
        self.bert_model = BERTLMModel.restore_from(context.artifacts['bert_model_path'], strict=False).to(self.device)
        
    def generate_query_embedding(self, query):
        """
        Generate BERT embeddings for the input query.
        """
        self.bert_model.eval()  # Set model to evaluation mode
        
        # Tokenize the input query and move tensors to the selected device
        encoded_input = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=128)
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        
        # Get the model's output embedding
        with torch.no_grad():
            output = self.bert_model.bert_model(**encoded_input)
        
        # Return the [CLS] token embedding as a NumPy array
        return output[:, 0, :].cpu().numpy()
    
    def predict(self, context, model_input, params):
        """
        Compute similarity between query and precomputed embeddings,
        then return the top 5 most similar results.
        """
        # Extract the query string from model input
        query = model_input["query"][0]
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Compute cosine similarity between query and precomputed embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings_df.values)
        
        # Get indices of top 5 most similar results
        top_indices = np.argsort(similarities[0])[::-1][:5]
        
        # Retrieve corresponding results from the corpus
        results = self.corpus_df.iloc[top_indices].copy()
        results.loc[:, 'Similarity'] = similarities[0][top_indices]
        
        # Return results as a dictionary
        return results.to_dict(orient="records")
    
    @classmethod
    def log_model(
        cls,
        model_name,
        corpus_path,
        embeddings_path,
        tokenizer_dir,
        bert_model_online_path,
        bert_model_datafabric_path,
        demo_path,
        config_path="../configs/config.yaml"
    ):
        """
        Logs the model to MLflow with appropriate artifacts and schema.
        Now uses in-memory model loading for ONNX export efficiency.
        """

        from nemo.collections.nlp.models import BERTLMModel
        
        # Define input and output schema
        input_schema = Schema([ColSpec("string", "query")])
        output_schema = Schema([
            TensorSpec(np.dtype("object"), (-1,), "List of Pledges and Similarities")
        ])
        params_schema = ParamSchema([ParamSpec("show_score", "boolean", False)])
        
        # Define model signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

        # Define the artifacts
        artifacts={
            "corpus_path": corpus_path,
            "embeddings_path": embeddings_path, 
            "tokenizer_dir": tokenizer_dir, 
            # If you are using the downloaded bert model then uncomment the line below and comment the other bert model line that uses nemo model from datafabric
            #"bert_model_path": bert_model_online_path,            
            "bert_model_path": bert_model_datafabric_path,
            "demo": demo_path,
            "config": config_path
        }

        src_dir = str(Path(__file__).parent.resolve())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BERT model from: {bert_model_datafabric_path}")
       
        
        # Load the NeMo BERT model into memory
        bert_model =  BERTLMModel.restore_from(bert_model_datafabric_path, strict=False).to(device)
        bert_model.eval() 

        wrapped_model = BERTModelWithHiddenStates(bert_model) #it doesn't have oficial nemo export function so its necessary to recreate the model as torch to use torch conversion
     
        batch_size = 1
        seq_len = 128
        vocab_size = 30522

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
        # Create ONNX export configuration
        model_configs = ModelExportConfig(
            model=wrapped_model,
            model_name="bert_tourism_onnx",
            input_sample=(                             
                    input_ids.to(device),
                    attention_mask.to(device),
                    token_type_ids.to(device)
                ),
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['embedding'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                'embedding': {0: 'batch_size'}
            }
        )        # Log the model with automatic ONNX conversion from memory
        log_model(
                artifact_path=model_name,
                python_model=cls(),
                artifacts=artifacts,
                signature=signature,
                models_to_convert_onnx=model_configs,     
                pip_requirements= "../requirements.txt",
                code_paths=[src_dir],
        )
      
        
        # Legacy code kept for reference (commented out)
        #mlflow.pyfunc.log_model(
            #artifact_path=model_name,
            #python_model=cls(),
            #artifacts=artifacts,
            #signature=signature,
            #pip_requirements="../requirements.txt",
            #code_paths=[src_dir],
       # )

# ── MLflow code-based entry-point ─────────────────────────────────────────────
def _load_pyfunc(context):
    """
    MLflow 3.x calls this after importing the module.
    Return a *new* instance of the model (do **not** load artifacts here;
    load_context() will be invoked right after).
    """
    return BERTTourismModel()