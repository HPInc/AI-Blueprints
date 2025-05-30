"""
Chatbot Service implementation that extends the BaseGenerativeService.
This service provides a RAG (Retrieval-Augmented Generation) chatbot with 
Galileo integration for protection, observation, and evaluation.
"""
import os
import uuid
import base64
import logging
from typing import Dict, Any, List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain.schema.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from galileo_protect import ProtectParser

# Import base service class from the shared location
import sys
import os
import time

# Add the src directory to the path to import base_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.service.base_service import BaseGenerativeService
from src.utils import get_context_window, dynamic_retriever, format_docs_with_adaptive_context

# Set up logger
logger = logging.getLogger(__name__)

class ChatbotService(BaseGenerativeService):
    """
    Chatbot Service that extends BaseGenerativeService to provide
    a RAG-based conversational AI with document retrieval capabilities.
    """
    def __init__(self):
        """Initialize the chatbot service."""
        super().__init__()
        self.memory = []
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            cache_folder="/tmp/hf_cache"
        )
        self.vectordb = None
        self.retriever = None
        self.prompt_str = None
        self.docs_path = None

    def load_config(self, context):
        """
        Load configuration from context artifacts and set up the docs path.

        Args:
            context: MLflow model context containing artifacts

        Returns:
            Dictionary containing the loaded configuration
        """
        # Load base configuration
        config = super().load_config(context)

        # Set docs path from artifacts
        self.docs_path = context.artifacts.get("docs", None)

        # Add additional chatbot-specific configuration
        config.update({
            "local_model_path": context.artifacts.get("models", "")
        })

        return config

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
                logger.info("Using local LlamaCpp model source")
                self.load_local_model(context)
            elif model_source == "hugging-face-local":
                logger.info("Using local Hugging Face model source")
                self.load_local_hf_model(context)
            elif model_source == "hugging-face-cloud":
                logger.info("Using cloud Hugging Face model source")
                self.load_cloud_hf_model(context)
            else:
                logger.error(f"Unsupported model source: {model_source}")
                raise ValueError(f"Unsupported model source: {model_source}")
                
            if self.llm is None:
                logger.error("Model failed to initialize - llm is None after loading")
                raise RuntimeError("Model initialization failed - llm is None")
                
            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
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
            model_path = self.model_config.get("local_model_path", context.artifacts.get("models", ""))
            
            logger.info(f"Model path: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"The model file was not found at: {model_path}")
            
            logger.info(f"Model file exists. Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
            
            logger.info("Setting up callback manager")
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Get model filename to determine context window from lookup table
            model_filename = os.path.basename(model_path)
            
            # Default context window size, will be updated if available in MODEL_CONTEXT_WINDOWS
            context_window = 4096
            
            logger.info("Initializing LlamaCpp with the following parameters:")
            logger.info(f"  - Model Path: {model_path}")
            logger.info(f"  - n_gpu_layers: 30, n_batch: 512, n_ctx: {context_window}")
            logger.info(f"  - max_tokens: 1024, f16_kv: True, temperature: 0.2")
            
            try:
                self.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=30,
                    n_batch=512,
                    n_ctx=context_window,
                    max_tokens=1024,
                    f16_kv=True,
                    callback_manager=self.callback_manager,
                    verbose=True, 
                    stop=[],
                    streaming=False,
                    temperature=0.2,
                )
                
                # Store context window in model for later retrieval
                self.llm.__dict__['_context_window'] = context_window
                logger.info(f"LlamaCpp model initialized successfully with context window of {context_window} tokens")
            except Exception as model_error:
                logger.error(f"Failed to initialize LlamaCpp model: {str(model_error)}")
                logger.error(f"Exception type: {type(model_error).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
                
            logger.info("Using the local LlamaCpp model.")
        except Exception as e:
            logger.error(f"Error in load_local_model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def load_local_hf_model(self, context):
        """
        Load a local Hugging Face model.

        Args:
            context: MLflow model context containing artifacts
        """
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Set context window explicitly to help with document retrieval (4096 tokens for DeepSeek model)
        context_window = 4096
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None:
            context_window = tokenizer.model_max_length
            
        # Store context window in model for later retrieval  
        self.llm.__dict__['_context_window'] = context_window
        
        logger.info(f"Using the local Deep Seek model downloaded from HuggingFace with context window of {context_window} tokens.")

    def load_cloud_hf_model(self, context):
        """
        Load a cloud-based Hugging Face model.

        Args:
            context: MLflow model context containing artifacts
        """
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=self.model_config["hf_key"],
            repo_id=repo_id,
        )
        
        # Set known context window for Mistral-7B-Instruct-v0.2 (8192 tokens) 
        context_window = 8192
        self.llm.__dict__['_context_window'] = context_window
           
        logger.info(f"Using the cloud Mistral model on HuggingFace with context window of {context_window} tokens.")

    def load_vector_database(self):
        """
        Load documents and create the vector database for retrieval.
        """
        if not self.docs_path or not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"The documents directory was not found at: {self.docs_path}")

        pdf_path = os.path.join(self.docs_path, "AIStudioDoc.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file 'AIStudioDoc.pdf' was not found at: {pdf_path}")

        logger.info(f"Reading and processing the PDF file: {pdf_path}")

        try:
            # Load PDF documents
            logger.info("Loading PDF data...")
            pdf_loader = PyPDFLoader(pdf_path)
            pdf_data = pdf_loader.load()

            # Ensure all content is string type
            for doc in pdf_data:
                if not isinstance(doc.page_content, str):
                    doc.page_content = str(doc.page_content)

            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            splits = text_splitter.split_documents(pdf_data)
            logger.info(f"PDF split into {len(splits)} parts.")

            logger.info("Using embedding model initialized during service initialization.")

            # Create vector database
            logger.info("Creating vector database...")
            self.vectordb = Chroma.from_documents(documents=splits, embedding=self.embedding)
            
            self.retriever = self.vectordb.as_retriever()

            logger.info("Vector database created successfully.")
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

    def load_prompt(self) -> None:
        """Load the prompt template for the chatbot."""
        self.prompt_str = """You are a chatbot assistant for a Data Science platform created by HP, called 'Z by HP AI Studio'. 
            Do not hallucinate and answer questions only if they are related to 'Z by HP AI Studio'. 
            Now, answer the question perfectly based on the following context:

            {context}

            Question: {query}
            """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

    def load_chain(self) -> None:
        """Create the RAG chain using the loaded model, retriever, and prompt."""
        if not self.retriever:
            raise ValueError("Retriever must be initialized before creating the chain")
            
        # Get model context window
        context_window = get_context_window(self.llm)
        logger.info(f"Using model with context window of {context_window} tokens")
        
        input_normalizer = RunnableLambda(lambda x: {"input": x} if isinstance(x, str) else x)
        
        # Use dynamic retriever based on context window
        def context_aware_retrieval(x):
            return dynamic_retriever(
                x["input"], 
                collection=self.vectordb, 
                context_window=context_window
            )
        
        # Use adaptive context formatter
        def adaptive_format(docs):
            return format_docs_with_adaptive_context(
                docs, 
                context_window=context_window
            )
            
        retriever_runnable = RunnableLambda(context_aware_retrieval)
        format_docs_r = RunnableLambda(adaptive_format)
        extract_input = RunnableLambda(lambda x: x["input"])

        self.chain = (
            input_normalizer
            | RunnableMap({
                "context": retriever_runnable | format_docs_r,
                "query": extract_input
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def load_context(self, context) -> None:
        """
        Load context for the model, including configuration, model, vector database and chains.

        Args:
            context: MLflow model context
        """
        # Load configuration
        self.load_config(context)

        # Set up environment
        self.setup_environment()

        try:
            # Load model, vector database, prompt, and chain
            self.load_model(context)
            self.load_vector_database()
            self.load_prompt()
            self.load_chain()

            # Set up Galileo integration
            self.setup_protection()
            self.setup_monitoring()

            logger.info(f"{self.__class__.__name__} successfully loaded and configured.")
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            raise

    def add_pdf(self, base64_pdf):
        """
        Add a new PDF to the vector database.

        Args:
            base64_pdf: Base64-encoded PDF content

        Returns:
            Dictionary with status information
        """
        try:
            pdf_bytes = base64.b64decode(base64_pdf)
            temp_pdf_path = f"/tmp/{uuid.uuid4()}.pdf"

            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            pdf_loader = PyPDFLoader(temp_pdf_path)
            pdf_data = pdf_loader.load()

            # Ensure all content is string type
            for doc in pdf_data:
                if not isinstance(doc.page_content, str):
                    doc.page_content = str(doc.page_content)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            new_splits = text_splitter.split_documents(pdf_data)

            # Create a new vector database with the new document using the already initialized embedding model
            vectordb = Chroma.from_documents(documents=new_splits, embedding=self.embedding)
            self.retriever = vectordb.as_retriever()

            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": "New PDF successfully added to the knowledge base.",
                "success": True
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error adding PDF: {str(e)}",
                "success": False
            }

    def get_prompt_template(self):
        """
        Get the current prompt template.

        Returns:
            Dictionary containing the prompt template
        """
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def set_prompt_template(self, new_prompt):
        """
        Update the prompt template.

        Args:
            new_prompt: New prompt template string

        Returns:
            Dictionary with status information
        """
        try:
            self.prompt_str = new_prompt
            self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

            # Rebuild the chain with the new prompt
            self.load_chain()
            self.setup_protection()

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": "Prompt template updated successfully.",
                "success": True
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error updating prompt template: {str(e)}",
                "success": False
            }

    def reset_history(self):
        """
        Reset the conversation history.

        Returns:
            Dictionary with status information
        """
        self.memory = []
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "Conversation history has been reset.",
            "success": True
        }
    
    def inference(self, context, user_query):
        """
        Process a user query and generate a response.
        
        Args:
            context: MLflow model context
            user_query: User's query string
            
        Returns:
            Dictionary with response data
        """
        try:
            # Get the model context window for optimized retrieval
            context_window = get_context_window(self.llm)
            
            # Run the query through the protected chain with monitoring
            response = self.protected_chain.invoke(
                {"input": user_query, "output": ""},
                config={"callbacks": [self.monitor_handler]}
            )
            
            # Get relevant documents using context-aware retrieval
            relevant_docs = dynamic_retriever(user_query, self.vectordb, context_window=context_window)
            chunks = [doc.page_content for doc in relevant_docs]
            
            # Update conversation history
            self.memory.append({"role": "User", "content": user_query})
            self.memory.append({"role": "Assistant", "content": response})
            
            return {
                "chunks": chunks,
                "history": [f"<{m['role']}> {m['content']}\n" for m in self.memory],
                "prompt": self.prompt_str,
                "output": response,
                "success": True
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": error_msg,
                "success": False
            }
    
    def predict(self, context, model_input, params=None):
        """
        Process inputs and generate appropriate responses based on parameters.
        
        Args:
            context: MLflow model context
            model_input: Input data dictionary with keys like 'query', 'prompt', 'document'
            params: Dictionary of parameters to control behavior
            
        Returns:
            Pandas DataFrame with the response data
        """
        if params is None:
            params = {
                "add_pdf": False,
                "get_prompt": False,
                "set_prompt": False,
                "reset_history": False,
                "get_model_info": False
            }
            
        try:
            # Return early for various special operations
            if params.get("get_model_info", False):
                result = self.get_model_info()
            elif params.get("get_prompt", False):
                result = self.get_prompt_template()
            elif params.get("set_prompt", False) and "prompt" in model_input:
                result = self.set_prompt_template(model_input["prompt"][0])
            elif params.get("reset_history", False):
                result = self.reset_history()
            elif params.get("add_pdf", False) and "document" in model_input:
                result = self.add_pdf(model_input["document"][0])
            # Standard query operation
            elif "query" in model_input:
                result = self.inference(context, model_input["query"][0])
            else:
                result = {
                    "chunks": [],
                    "history": [],
                    "prompt": self.prompt_str,
                    "output": "Error: No valid operation specified in the request.",
                    "success": False
                }
        except Exception as e:
            import traceback
            result = {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str if hasattr(self, 'prompt_str') else "",
                "output": f"Error: {str(e)}\nTraceback: {traceback.format_exc()}",
                "success": False
            }
            
        return pd.DataFrame([result])

    @classmethod
    def log_model(cls, artifact_path, secrets_path, config_path, docs_path, model_path=None, demo_folder=None):
        """
        Log the model to MLflow.
        
        Args:
            artifact_path: Path to store the model artifacts
            secrets_path: Path to the secrets file
            config_path: Path to the configuration file
            docs_path: Path to the documents directory
            model_path: Path to the model file (optional)
            demo_folder: Path to the demo folder (optional)
            
        Returns:
            None
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec
        
        # Create demo folder if specified and doesn't exist
        if demo_folder and not os.path.exists(demo_folder):
            os.makedirs(demo_folder, exist_ok=True)
        
        # Define model input/output schema
        input_schema = Schema([
            ColSpec("string", "query"),
            ColSpec("string", "prompt"),
            ColSpec("string", "document")
        ])
        output_schema = Schema([
            ColSpec("string", "chunks"),
            ColSpec("string", "history"),
            ColSpec("string", "prompt"),
            ColSpec("string", "output"),
            ColSpec("boolean", "success")
        ])
        param_schema = ParamSchema([
            ParamSpec("add_pdf", "boolean", False),
            ParamSpec("get_prompt", "boolean", False),
            ParamSpec("set_prompt", "boolean", False),
            ParamSpec("reset_history", "boolean", False),
            ParamSpec("get_model_info", "boolean", False)
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)
        
        # Prepare artifacts
        artifacts = {
            "secrets": secrets_path, 
            "config": config_path, 
            "docs": docs_path
        }
        
        if demo_folder:
            artifacts["demo"] = demo_folder
        
        if model_path:
            artifacts["models"] = model_path
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            code_paths=["../core", "../../src"],
            pip_requirements=[
                "PyPDF",
                "pyyaml",
                "tokenizers>=0.13.0",
                "httpx>=0.24.0",
            ]
        )
        logger.info("Model and artifacts successfully registered in MLflow.")

    def get_model_info(self):
        """
        Get information about the model, including context window size.
        
        Returns:
            Dictionary containing model information
        """
        try:
            context_window = get_context_window(self.llm)
            model_type = type(self.llm).__name__
            
            # Get additional info based on model type
            additional_info = {}
            if hasattr(self.llm, 'model_path'):
                additional_info['model_path'] = self.llm.model_path
            if hasattr(self.llm, 'repo_id'):
                additional_info['repo_id'] = self.llm.repo_id
                
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Model type: {model_type}, Context window: {context_window} tokens",
                "additional_info": additional_info,
                "success": True
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error retrieving model info: {str(e)}",
                "success": False
            }