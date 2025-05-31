"""
Code Generation Service implementation that extends the BaseGenerativeService.

This service provides code generation capabilities using LLM models with vector retrieval
and integrates with Galileo for protection, observation, and evaluation.
It can extract context from GitHub repositories to enhance code generation responses.
"""

import os
import sys
import logging
import traceback
import time
import json
import datetime
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, Document
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from galileo_protect import ProtectParser
from core.chroma_embedding_adapter import ChromaEmbeddingAdapter
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService
from src.utils import get_context_window, dynamic_retriever, format_docs_with_adaptive_context, clean_code, get_model_context_window

# Add core directory to path for local imports
core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if core_path not in sys.path:
    sys.path.append(core_path)

# Import GitHub extraction and context storage tools
from core.extract_text.github_repository_extractor import GitHubRepositoryExtractor
from core.generate_metadata.llm_context_updater import LLMContextUpdater
from core.dataflow.dataflow import EmbeddingUpdater, DataFrameConverter
from core.vector_database.vector_store_writer import VectorStoreWriter
from core.generate_metadata.async_repository_processor import AsyncRepositoryProcessor
from core.generate_metadata.repository_status_tracker import RepositoryStatusTracker, ProcessingStatus

# Set up logger
logger = logging.getLogger(__name__)

class CodeGenerationService(BaseGenerativeService):
    """
    Code Generation Service that extends the BaseGenerativeService.
    Supports both direct code generation questions and
    context retrieval from specified GitHub repositories.
    """

    def __init__(self, delay_async_init=False):
        """Initialize the code generation service.
        
        IMPORTANT: The embedding initialization order is critical - embeddings must be
        initialized before any LLM model to prevent CUDA library loading issues
        that may occur when initializing LlamaCpp models.
        
        Args:
            delay_async_init: If True, delay initialization of thread-based components
                             to allow pickling the model for MLflow serialization
        """
        super().__init__()
        self.vector_store = None
        self.retriever = None
        self.collection = None
        self.collection_name = "my_collection"
        self.embedding_path = None
        self.context_window = None
        
        # Repository cache to avoid re-processing the same repositories
        self.repository_cache = {}
        
        # The embedding_function will be initialized in load_context
        # or can be manually initialized by calling initialize_embedding_function
        self.embedding_function = None
        self.chroma_embedding_function = None
        
        # Initialize status tracker and async repository processor
        if not delay_async_init:
            self._initialize_async_components()
        else:
            # Don't initialize the thread-based components during MLflow serialization
            self.repository_status_tracker = None
            self.repository_processor = None
        
        # Configure logging to reduce noise
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Set logging format for better readability
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                formatter = logging.Formatter('[%(levelname)s] %(message)s')
                handler.setFormatter(formatter)
    
    def _initialize_async_components(self):
        """
        Initialize the asynchronous repository processing components.
        This method is separated to allow MLflow serialization without thread locks.
        """
        try:
            # Initialize the repository status tracker
            self.repository_status_tracker = RepositoryStatusTracker()
            
            # Repository processor will be initialized when needed
            self.repository_processor = None
            
            logger.info("Async repository processing components initialized")
        except Exception as e:
            logger.warning(f"Error initializing async components: {str(e)}")
            self.repository_status_tracker = None
            self.repository_processor = None

    def initialize_embedding_function(self, embedding_model_path=None):
        """Initialize the embedding function.
        
        Args:
            embedding_model_path: Path to a locally saved embedding model (optional)
            
        Returns:
            An initialized HuggingFaceEmbeddings object
        """
        logger.info("Initializing embedding function")
        
        # Determine which model path to use
        model_name = embedding_model_path if embedding_model_path else "all-MiniLM-L6-v2"
        if embedding_model_path:
            logger.info(f"Using provided embedding model path: {embedding_model_path}")
        else:
            logger.info("Using default embedding model: all-MiniLM-L6-v2")
        
        # Initialize the embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized HuggingFaceEmbeddings with model: {model_name}")
        
        # Create the adapter 
        self.chroma_embedding_function = ChromaEmbeddingAdapter(self.embedding_function)
        
        logger.info("ChromaEmbeddingAdapter successfully initialized with all required methods")
            
        return self.embedding_function
    
    def extract_repository(self, repository_url: str, metadata_only: bool = False) -> Dict[str, Any]:
        """
        Extract code and metadata from a GitHub repository.
        Uses a cache mechanism to avoid re-processing the same repository.
        
        Args:
            repository_url: URL of the GitHub repository
            metadata_only: If True, only perform fast metadata extraction without LLM processing
            
        Returns:
            A status dict with information about the asynchronous processing job
        """
        # Initialize the async repository processor if not already done
        if self.repository_processor is None and self.embedding_function is not None:
            # Use dependency injection to provide the required classes
            self.repository_processor = AsyncRepositoryProcessor(
                repository_extractor_class=GitHubRepositoryExtractor,
                llm_context_updater_class=LLMContextUpdater,
                status_tracker=self.repository_status_tracker
            )
            logger.info("AsyncRepositoryProcessor initialized")
            
        # Get current status
        status = self.repository_processor.get_repository_status(repository_url)
        
        # If already complete, use the cached result
        if status.get("status") == ProcessingStatus.COMPLETE.value:
            logger.info(f"Using cached complete repository: {repository_url}")
            
            # Get cached data and update the collection
            extracted_data = status.get("context")
            if extracted_data:
                self._store_in_vector_db(repository_url, extracted_data)
                return extracted_data
        
        # Start or continue processing asynchronously
        extraction_params = {
            "save_dir": "./repo_files",
            "verbose": False,
            "max_file_size_kb": 500,
            "max_chunk_size": 100,
            "supported_extensions": ('.py', '.ipynb', '.md', '.txt', '.json', '.js', '.ts')
        }
        
        context_params = {
            "llm_chain": self.llm if hasattr(self, 'llm') else None,
            "prompt_template": self.code_description_prompt,
            "verbose": False,
            "overwrite": not metadata_only  # Only overwrite if doing full processing
        }
        
        # Process repository in background and immediately return status
        status = self.repository_processor.process_repository_async(
            repo_url=repository_url,
            extraction_params=extraction_params,
            context_params=context_params,
            force_refresh=False,
            on_complete=self._on_repository_complete
        )
        
        # Return the current processing status
        return {"status": status, "repository_url": repository_url}
    

    
    def custom_retriever(self, query: str, top_n: int = None) -> List[Document]:
        """
        Custom retriever function 
        
        Args:
            query: The query string for retrieval
            top_n: Number of documents to retrieve (if None, determined by context window)
            
        Returns:
            List of Document objects with content and metadata
        """
        # Determine whether to use the vector_store or collection
        retrieval_source = None
        logger.info("Using vector_store for retrieval")
        # Check if the vector store has a properly set embedding function
        try:
            if not hasattr(self.vector_store._embedding_function, 'embed_query'):
                logger.warning("Vector store has invalid embedding function - reinitializing")
                # Recreate the vector store with proper embedding function
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    persist_directory="./chroma_db",
                    embedding_function=self.chroma_embedding_function
                )
                logger.info("Vector store reinitialized with proper embedding function")
        except Exception as vs_err:
            logger.error(f"Failed to check/fix vector store embedding function: {str(vs_err)}")
        
        retrieval_source = self.vector_store

        try:
            # Use class-level context window if available, or get from model
            context_window = None
            if hasattr(self, 'context_window') and self.context_window:
                context_window = self.context_window
                logger.info(f"Using stored context window: {context_window} tokens")
            elif hasattr(self, 'llm'):
                context_window = get_context_window(self.llm)
                logger.info(f"Retrieved model context window: {context_window} tokens")
            
            # Use the dynamic retriever with the proper retrieval source
            documents = dynamic_retriever(
                query=query,
                collection=retrieval_source,
                top_n=top_n,
                context_window=context_window
            )
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """
        Load or create a vector store for code retrieval.
        
        Args:
            persist_directory: Directory to store vector database
        """
        try:
            logger.info(f"Loading vector store from {persist_directory}")
            # Make sure directory exists
            import os
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize chromadb client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Try to get existing collection or create a new one
            try:
                self.collection = client.get_or_create_collection(
                    name=self.collection_name
                )
                logger.info(f"Collection '{self.collection_name}' loaded/created successfully")
            except Exception as col_err:
                logger.error(f"Error getting/creating collection: {str(col_err)}")
                logger.error(f"Exception type: {type(col_err).__name__}")
            
            # Initialize LangChain vector store
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.chroma_embedding_function
            )
            self.retriever = self.vector_store.as_retriever()
            logger.info(f"Vector store successfully loaded from {persist_directory} with embedding function")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
    
    def load_model(self, context) -> None:
        """
        Load the appropriate model based on configuration.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            model_source = self.model_config.get("model_source", "local")
            logger.info(f"Attempting to load model from source: {model_source}")
            
            if model_source == "local":
                self.load_local_model(context)
            else:
                logger.info(f"Using model source: {model_source}")
                # Import utility function for initializing LLM
                from src.utils import initialize_llm
                
                # Extract secrets from config
                secrets = {}
                if "secrets" in self.model_config:
                    secrets = self.model_config["secrets"]
                
                # Get local model path from artifacts
                local_model_path = None
                if "models" in context.artifacts:
                    local_model_path = context.artifacts["models"]
                
                # Initialize LLM using the utility function
                self.llm = initialize_llm(model_source, secrets, local_model_path)
                
            if self.llm is None:
                logger.error("Failed to initialize model from any source")
                raise ValueError("No model could be initialized")
                
            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_local_model(self, context):
        """
        Load a local LlamaCpp model.
        
        Args:
            context: MLflow model context containing artifacts
        """
        try:
            logger.info("Initializing local LlamaCpp model.")
            model_path = context.artifacts.get("models", None)
            
            logger.info(f"Model path: {model_path}")
            
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found at path: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Setting up callback manager")
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Determine optimal context window for this model using the utility function
            
            # Create a temporary model object with model_path attribute
            temp_model = type('TempModel', (), {'model_path': model_path})
            
            # Get context window using the utility function
            context_window = get_model_context_window(temp_model)
            logger.info(f"Determined context window: {context_window} tokens")
            
            logger.info("Initializing LlamaCpp with the following parameters:")
            logger.info(f"  - Model Path: {model_path}")
            
            # Check CUDA availability
            cuda_available = False
            try:
                # Try to check if CUDA is available
                import subprocess
                try:
                    subprocess.check_output(['ldconfig', '-p'], stderr=subprocess.STDOUT)
                    libcuda_check = subprocess.check_output(['ldconfig', '-p', '|', 'grep', 'libcuda.so.1'], stderr=subprocess.STDOUT, shell=True)
                    if libcuda_check:
                        cuda_available = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    cuda_available = False
            except Exception:
                cuda_available = False
                
            logger.info(f"CUDA availability check: {'Available' if cuda_available else 'Not available'}")
            
            # Configure GPU layers based on CUDA availability
            n_gpu_layers = 30 if cuda_available else 0
            logger.info(f"  - n_gpu_layers: {n_gpu_layers}, n_batch: 512, n_ctx: {context_window}")
            logger.info(f"  - max_tokens: 1024, f16_kv: True, temperature: 0.2")
            
            try:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=512,
                    n_ctx=context_window,
                    f16_kv=True,
                    callback_manager=None,
                    verbose=False,
                    max_tokens=1024,
                    temperature=0.2
                )
                
                self.llm.__dict__['_context_window'] = context_window
                self.context_window = context_window
                logger.info(f"Using local LlamaCpp model for code generation with {'GPU' if cuda_available else 'CPU'} mode.")
            except Exception as model_error:
                logger.error(f"Failed to initialize LlamaCpp model: {str(model_error)}")
                logger.error(f"Exception type: {type(model_error).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        except Exception as e:
            logger.error(f"Error in load_local_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_prompt(self) -> None:
        """Load the prompt templates for code generation."""
        # Template for code generation with repository context
        self.code_description_template = """You will receive three pieces of information: a code snippet, a file name, and an optional context. Based on this information, explain in a clear, summarized and concise way what the code snippet is doing.

Code:
{code}

File name:
{filename}

Context:
{context}

Describe what the code above does.
"""

        # Template for direct code generation without repository context
        self.code_generation_template = """You are a code generator AI that ONLY outputs working Python code.
NEVER ask questions or request clarification.
ALWAYS respond with complete, executable Python code.
DO NOT include any explanations, comments, or non-code text.
If you're uncertain about implementation details, make reasonable assumptions and provide working code.

Context:
{context}

Task: {question}
"""

        # Default prompt for backward compatibility with existing chain structure
        self.prompt_str = """You are a Python wizard tasked with generating code for a Jupyter Notebook (.ipynb) based on the given context.
Your answer should consist of just the Python code, without any additional text or explanation.

Context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)
        
        # Create additional prompt objects
        self.code_description_prompt = ChatPromptTemplate.from_template(self.code_description_template)
        self.code_generation_prompt = ChatPromptTemplate.from_template(self.code_generation_template)
    
    def load_chain(self) -> None:
        """Create the code generation chains using the loaded model, prompts, and retriever."""
        try:
            # Load the vector store first
            logger.info("Loading vector store for retrieval")
            self.load_vector_store()
            
            # Verify retriever readiness using either direct collection or fallback to LangChain retriever
            if not self.vector_store and not self.collection and not self.retriever:
                logger.error("No retrieval mechanism available")
                raise ValueError("A retrieval mechanism must be initialized before creating the chain")
                
            logger.info("Creating code generation chains")
            
            # Use class-level context window if available, or retrieve from model
            context_window = None
            if hasattr(self, 'context_window') and self.context_window:
                context_window = self.context_window
                logger.info(f"Using stored context window: {context_window} tokens")
            elif hasattr(self, 'llm'):
                context_window = get_context_window(self.llm)
                logger.info(f"Retrieved model context window: {context_window} tokens")
            
            # Create the context formatter function with adaptive formatting
            def get_formatted_context(inputs):
                # Get retrieval query (could be "query" or "question" depending on input)
                query = inputs.get("question", "")
                
                # Get documents using shared retriever
                docs = self.custom_retriever(query)
                
                if not docs:
                    logger.warning("No documents retrieved for query")
                    return ""
                    
                # Format documents with adaptive context optimization
                return format_docs_with_adaptive_context(docs, context_window=context_window)
            
            # Create the standard chain for general use
            logger.info("Creating standard chain")
            self.chain = {
                "context": get_formatted_context,
                "question": RunnablePassthrough()
            } | self.prompt | self.llm | StrOutputParser()
                
            # Create the specialized chain for repository-based code generation
            logger.info("Creating repository-based code generation chain")
            
            # This function extracts code and filename from the first document retrieved for the description prompt
            def extract_code_info_from_docs(inputs):
                # Get retrieval query (could be "query" or "question" depending on input)
                query = inputs.get("query", inputs.get("question", ""))
                
                # Get documents using shared retriever
                docs = self.custom_retriever(query)
                
                if not docs or len(docs) == 0:
                    # If no documents found, return empty values
                    return {
                        "code": "No code found",
                        "filename": "No filename found",
                        "context": "No relevant documents retrieved"
                    }
                
                # Extract code and filename from the first (most relevant) document
                doc = docs[0]
                code = doc.page_content
                filename = doc.metadata.get("filename", "unknown_file")
                
                # Format the rest of the documents as context
                remaining_docs = docs[1:] if len(docs) > 1 else []
                context = format_docs_with_adaptive_context(remaining_docs, context_window=context_window) if remaining_docs else ""
                
                return {
                    "code": code,
                    "filename": filename,
                    "context": context
                }
            
            self.repository_chain = extract_code_info_from_docs | self.code_description_prompt | self.llm | StrOutputParser()
            
            # Create a direct code generation chain without repository context
            logger.info("Creating direct code generation chain")
            self.direct_chain = {
                "context": lambda _: "",  # Empty context for direct questions
                "question": RunnablePassthrough()
            } | self.code_generation_prompt | self.llm | StrOutputParser()
            
            logger.info("Code generation chains created successfully")
        except Exception as e:
            logger.error(f"Error creating code generation chain: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, context, model_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate code based on the input parameters.
        
        Args:
            context: MLflow model context
            model_input: Input data for code generation, expecting:
                         - A dict with "inputs" containing any of:
                           - "question": User's code generation request (required)
                           - "repository_url": GitHub repository URL (optional)
                           - "metadata_only": Process only metadata without full LLM analysis (optional, default: False)
            
        Returns:
            DataFrame with the generated code in a "result" column
        """
        # Set reasonable logging levels to reduce clutter
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        logger.info(f"Received model_input: {str(model_input)[:200]}...")
        
        # Extract input data from the MLFlow API format
        if "inputs" in model_input:
            input_data = model_input["inputs"]
        else:
            input_data = model_input
        
        # Extract main fields from input data
        question = ""
        repository_url = None
        metadata_only = False
        
        # Extract question field (required)
        if "question" in input_data:
            if hasattr(input_data["question"], "iloc"):
                question = input_data["question"].iloc[0] if not input_data["question"].empty else ""
            else:
                question = input_data["question"]
        
        # Extract repository_url field (optional)
        if "repository_url" in input_data:
            if hasattr(input_data["repository_url"], "iloc"):
                repository_url = input_data["repository_url"].iloc[0] if not input_data["repository_url"].empty else None
            else:
                repository_url = input_data["repository_url"]
        else:
            logger.info("No repository_url provided, resetting repository state")
            self.reset_repository_state()
        
        # Extract metadata_only parameter (optional)
        if "metadata_only" in input_data:
            if hasattr(input_data["metadata_only"], "iloc"):
                metadata_only = input_data["metadata_only"].iloc[0] if not input_data["metadata_only"].empty else False
            else:
                metadata_only = bool(input_data["metadata_only"])
        
        # Check if question field is provided
        if not question:
            logger.warning("No question provided for code generation")
            return pd.DataFrame([{"result": "Error: No question provided for code generation."}])
        
        try:
            logger.info(f"Processing code generation request for question: {str(question)}")
            logger.info(f"Parameters: metadata_only={metadata_only}")
            
            # If repository_url is provided, process it first
            if repository_url:
                logger.info(f"Repository URL provided: {repository_url}")
                try:
                    # Extract repository data with the specified parameters
                    start_time = time.time()
                    repo_response = self.extract_repository(
                        repository_url, 
                        metadata_only=metadata_only,
                    )
                    processing_time = time.time() - start_time
                    
                    # Process the repository response
                    if isinstance(repo_response, dict) and "status" in repo_response:
                        # This is a processing status response
                        status = repo_response.get("status", {})
                        status_value = status.get("status", "unknown")
                        
                        # Handle different status values
                        if status_value == ProcessingStatus.COMPLETE.value:
                            # Repository is already processed and available
                            logger.info("Repository already processed, continuing with response generation")
                        elif status_value in [ProcessingStatus.PROCESSING.value, ProcessingStatus.NOT_STARTED.value]:
                            # Repository is being processed or just started
                            progress = status.get("progress", 0)
                            files_processed = status.get("files_processed", 0)
                            total_files = status.get("total_files", 0)
                            
                            # Return a status response
                            status_message = (
                                f"Repository processing in progress: {progress}% complete. "
                                f"Processed {files_processed}/{total_files} files. "
                                f"Please retry your request in a few moments."
                            )
                            
                            if status_value == ProcessingStatus.NOT_STARTED.value:
                                status_message = "Repository processing has started. Please retry your request in a few moments."
                                
                            return pd.DataFrame([{
                                "result": status_message,
                                "status": status_value,
                                "progress": progress,
                                "repository_url": repository_url
                            }])
                        elif status_value == ProcessingStatus.ERROR.value:
                            # Repository processing encountered an error
                            error_message = status.get("error_message", "Unknown error during repository processing")
                            logger.error(f"Repository processing error: {error_message}")
                            
                            # Fall back to direct generation
                            logger.info("Falling back to direct generation due to repository processing error")
                            result = self.direct_chain.invoke(
                                {"question": question},
                                config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                            )
                            
                            error_info = f"# Note: Repository context unavailable due to processing error\n# Error: {error_message}\n\n"
                            return pd.DataFrame([{"result": error_info + result}])
                    else:
                        # Repository processing completed
                        logger.info(f"Repository processing completed in {processing_time:.2f} seconds")
                    
                    # If we have data in the collection, use it for code generation
                    if self.collection:
                        try:
                            count = self.collection.count()
                            logger.info(f"Collection '{self.collection_name}' has {count} documents")
                            
                            # Use the repository chain with the question
                            # TODO: Is this chain input correct? Does it match with the prompt template?
                            chain_input = {"question": question, "query": question}
                            logger.info(f"Using repository chain with input: {chain_input}")
                            
                            # Process with repository context
                            if hasattr(self, 'protect_tool') and self.protect_tool is not None:
                                try:
                                    result = self.repository_chain.invoke(
                                        chain_input, 
                                        config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                    )
                                except Exception as protect_error:
                                    logger.warning(f"Error with repository chain: {str(protect_error)}")
                                    # Fall back to direct chain
                                    result = self.direct_chain.invoke(
                                        chain_input,
                                        config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                    )
                            else:
                                result = self.repository_chain.invoke(
                                    chain_input,
                                    config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                                )
                            
                            # Include repository processing info in response for observability
                            processing_info = {
                                "processing_time_seconds": processing_time,
                                "metadata_only": metadata_only,
                                "document_count": count,
                                "repository_url": repository_url
                            }
                        except Exception as count_error:
                            logger.warning(f"Could not access collection: {str(count_error)}")
                            processing_info = {
                                "processing_time_seconds": processing_time,
                                "metadata_only": metadata_only,
                                "error": "collection_access_failed",
                                "repository_url": repository_url
                            }
                    else:
                        # If no collection is available, fall back to direct generation
                        logger.warning("No collection available")
                        processing_info = {
                            "processing_time_seconds": processing_time,
                            "metadata_only": metadata_only,
                            "error": "no_collection_created",
                            "repository_url": repository_url
                        }
                except Exception as repo_error:
                    logger.error(f"Error processing repository: {str(repo_error)}")
                    processing_info = {
                        "error": f"repository_processing_failed: {str(repo_error)[:100]}",
                        "repository_url": repository_url,
                        "metadata_only": metadata_only
                    }
            else:
                # Process the request using direct generation (no repository context)
                logger.info("No repository URL provided, using direct code generation")
                # Ensure we're not using any previous repository state
                self.reset_repository_state()
                result = self.direct_chain.invoke(
                    {"question": question},
                    config={"callbacks": [self.prompt_handler] if hasattr(self, 'prompt_handler') else None}
                )
                processing_info = {"mode": "direct_generation"}
            
            logger.info("Code generation processed successfully")
            
            # Clean up the result using the imported clean_code utility function
            clean_code_result = clean_code(result)
            
            # Log processing info 
            logger.info(f"Processing info: {json.dumps(processing_info)}")
            
            # Return only the clean code without any prefixes
            return pd.DataFrame([{"result": clean_code_result}])
        except Exception as e:
            error_message = f"Error processing code generation: {str(e)}"
            logger.error(error_message)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame([{"result": f"# Error during processing\n# {error_message}\n\n# Falling back to basic response\n\n# Your question was: {question}\n\n# Please try again with metadata_only=True or a smaller repository"}])
    
    @classmethod
    def log_model(cls, secrets_path, config_path, model_path=None, embedding_model_path=None, delay_async_init=True):
        """
        Log the model to MLflow.
        
        Args:
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            model_path: Path to the LLM model file (optional)
            embedding_model_path: Path to the locally saved embedding model directory (optional)
                                 If provided, will be used instead of downloading the model
            delay_async_init: If True, delay thread-based component initialization during serialization (default: True)
                             to prevent MLflow serialization errors with thread locks
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Define model input/output schema with all parameters
        input_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "repository_url", required=False),  # Optional repository URL
            ColSpec("boolean", "metadata_only", required=False),  # Optional metadata-only flag
        ])
        output_schema = Schema([
            ColSpec("string", "result")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path,
            "config": config_path
        }
        
        if model_path:
            artifacts["models"] = model_path
            
        # Add embedding model path to artifacts if provided and exists
        # This will allow us to use a locally saved model instead of downloading it during initialization
        if embedding_model_path and os.path.exists(embedding_model_path):
            artifacts["embedding_model"] = embedding_model_path
            logger.info(f"Using local embedding model from: {embedding_model_path}")
        else:
            logger.warning("No local embedding model path provided or path doesn't exist. " 
                         "The service will download the embedding model during initialization.")
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="code_generation_service",
            python_model=cls(delay_async_init=delay_async_init),  # Create instance with delayed initialization
            artifacts=artifacts,
            signature=signature,
            code_paths=["./core", "../../src"],
            pip_requirements=[
                "mlflow==2.9.2", 
                "langchain", 
                "promptquality", 
                "galileo-protect==0.15.1", 
                "galileo-observe==1.13.2",
                "chromadb",
                "langchain_core",
                "langchain_huggingface",
                "langchain_community",
                "sentence-transformers",
                "gitpython",
                "pyyaml"
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")
    
    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, and chains.
        This is an override of the BaseGenerativeService's load_context method.
        
        Args:
            context: MLflow model context
        """
        # First, initialize the embedding function - will check for artifact model first
        embedding_model_path = None
        if "embedding_model" in context.artifacts:
            embedding_model_path = context.artifacts["embedding_model"]
            if os.path.exists(embedding_model_path):
                logger.info(f"Found saved embedding model in artifacts: {embedding_model_path}")
            else:
                logger.warning(f"Embedding model path provided in artifacts but not found: {embedding_model_path}")
                embedding_model_path = None
        
        # Initialize the embedding function with the artifact path if available, otherwise use default
        try:
            self.initialize_embedding_function(embedding_model_path)
        except Exception as e:
            logger.warning(f"Failed to initialize embedding function: {str(e)}")
            logger.warning("Will attempt to initialize default embedding model as fallback")
            try:
                self.initialize_embedding_function()
            except Exception as e2:
                logger.error(f"Failed to initialize default embedding model: {str(e2)}")
                # Continue with initialization even if embedding fails - some functions might not need it
        
        # Initialize async components if they haven't been initialized yet
        # This is needed when loading from MLflow after serialization
        if self.repository_status_tracker is None:
            self._initialize_async_components()
            logger.info("Initialized async components during model loading")
        
        # Call the parent load_context method to handle the rest of the initialization
        super().load_context(context)
    
    def reset_repository_state(self, repository_url=None):
        """
        Reset the repository state if no specific repository URL is provided.
        
        Args:
            repository_url: If provided, keep this repository's data; otherwise, reset completely
        """
        if repository_url is None:
            logger.info("Resetting active repository state - no repository URL provided")
            self.collection = None
            
            # Reset vector store if possible
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                try:
                    logger.info("Attempting to reset vector store")
                    # Create new empty vector store with the same embedding function
                    if self.embedding_function is not None:
                        self.vector_store = Chroma(embedding_function=self.embedding_function)
                        self.retriever = self.vector_store.as_retriever()
                        logger.info("Vector store reset successfully")
                except Exception as e:
                    logger.warning(f"Failed to reset vector store: {str(e)}")
        else:
            logger.info(f"Setting repository state for: {repository_url}")
            
            # If we have an async processor, check if the repository is processed there
            if self.repository_processor is not None:
                import hashlib
                repo_id = hashlib.md5(repository_url.encode()).hexdigest()
                status = self.repository_status_tracker.get_status(repo_id)
                
                if status and status.get("status") == ProcessingStatus.COMPLETE.value:
                    # Repository is fully processed in the async processor
                    logger.info(f"Repository {repository_url} found in async processor cache, activating")
                    
                    # Get the collection from the cache
                    if repository_url in self.repository_cache:
                        self.collection = self.repository_cache[repository_url]["collection"]
                        logger.info("Collection activated from repository cache")
                        return
            
            # Check traditional cache if not found in async processor
            if repository_url in self.repository_cache:
                logger.info(f"Repository {repository_url} found in cache, activating")
                self.collection = self.repository_cache[repository_url]["collection"]
    
    def _on_repository_complete(self, repo_id: str, data: List[Dict[str, Any]]) -> None:
        """
        Callback for when repository processing completes.
        This method is called by the AsyncRepositoryProcessor when a repository is fully processed.
        
        Args:
            repo_id: The unique identifier for the repository
            data: The processed data with context and embeddings
        """
        # Get the repository URL from the status tracker
        repo_url = None
        status = self.repository_status_tracker.get_status(repo_id)
        if status:
            repo_url = status.get("repository_url")
        
        if not repo_url:
            logger.warning(f"Repository URL not found for ID: {repo_id}")
            return
            
        logger.info(f"Repository processing completed for: {repo_url}")
        
        # Store the processed data in the vector database
        self._store_in_vector_db(repo_url, data)
        
    def _store_in_vector_db(self, repository_url: str, data: List[Dict[str, Any]]) -> None:
        """
        Store repository data in the vector database.
        
        Args:
            repository_url: URL of the GitHub repository
            data: Processed data with context and embeddings
        """
        try:
            # Ensure data has valid embeddings before processing
            logger.info(f"Validating embeddings for {len(data)} items")
            
            # Debug the data to understand embedding issues
            has_proper_embeddings = False
            for idx, item in enumerate(data[:3]):  # Check first few items
                embedding = item.get("embedding", None)
                if embedding and isinstance(embedding, list) and len(embedding) > 0 and embedding[0] != 0.0:
                    has_proper_embeddings = True
                    logger.info(f"Sample valid embedding found: Length={len(embedding)}, First few values: {embedding[:5]}")
                    break
            
            if not has_proper_embeddings:
                logger.warning("No proper embeddings found in data. Will regenerate embeddings if possible.")
                
                # Try to regenerate embeddings if we have an embedding model
                if self.embedding_function is not None:
                    logger.info("Regenerating embeddings using the embedding model")
                    embedding_updater = EmbeddingUpdater(embedding_model=self.embedding_function, verbose=True)
                    data = embedding_updater.update(data)
                    logger.info("Embeddings regenerated successfully")
                    
            # Now validate and fill in any remaining missing embeddings
            valid_data = []
            default_embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
            for item in data:
                # Only replace embeddings if necessary
                if "embedding" not in item or item["embedding"] is None or (
                    isinstance(item["embedding"], list) and (
                        len(item["embedding"]) == 0 or 
                        any(e is None for e in item["embedding"])
                    )):
                    logger.warning(f"Invalid or missing embedding for item {item.get('id', 'unknown')} - replacing with zeros")
                    item["embedding"] = [0.0] * default_embedding_dim
                valid_data.append(item)
            
            # Convert to DataFrame using robust DataFrameConverter
            df_converter = DataFrameConverter(verbose=True)
            data_df = df_converter.to_dataframe(valid_data)
            
            # Create a unique collection name for this repository to avoid collisions
            import hashlib
            repo_hash = hashlib.md5(repository_url.encode()).hexdigest()[:8]
            collection_name = f"repo_{repo_hash}"
            
            # Set up the persistent directory
            persist_dir = "./chroma_db"
            import os
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize the ChromaDB client and collection
            logger.info(f"Initializing ChromaDB persistent client at {persist_dir}")
            client = chromadb.PersistentClient(path=persist_dir)
            
            # Get or create the collection - do not pass embedding function here
            self.collection = client.get_or_create_collection(name=collection_name)
            
            # Use VectorStoreWriter for robust upsert with error handling
            logger.info(f"Upserting data to collection {collection_name}")
            vector_writer = VectorStoreWriter(collection_name=collection_name, verbose=True)
            vector_writer.collection = self.collection 
            vector_writer.upsert_dataframe(data_df)
            
            # Save the collection reference for later use
            self.collection = vector_writer.collection
            logger.info(f"Repository data stored in collection: {collection_name} with {len(data)} items")
            
            # Update cache with the processed data
            self.repository_cache[repository_url] = {
                "data": valid_data,
                "collection": self.collection,
                "timestamp": time.time(),
                "metadata_only": False 
            }
            
            # Update LangChain retriever from the collection
            try:
                self.vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.chroma_embedding_function
                )
                self.retriever = self.vector_store.as_retriever()
                logger.info(f"Updated LangChain retriever for collection: {collection_name} with embedding function")
            except Exception as ret_err:
                logger.error(f"Error creating retriever: {str(ret_err)}")
                
        except Exception as e:
            logger.error(f"Error storing repository data in vector DB: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
