"""
Standalone Model class.

Business Logic Layer
- Handles RAG-based question answering with document retrieval
- Manages model initialization, embeddings, vector database, and prediction logic
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import uuid
import base64
import logging
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fix for Pydantic model rebuild issue
if hasattr(LlamaCpp, "model_rebuild"):
    LlamaCpp.model_rebuild()
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    HuggingFaceEndpoint,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Add the src directory to the path to import utilities
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import (
    get_context_window,
    dynamic_retriever,
    format_docs_with_adaptive_context,
    load_secrets_to_env,
)

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles RAG-based question answering with document retrieval.
    """

    def __init__(
        self, config: dict, docs_path: str, model_path: str = None, secrets: dict = None
    ):
        """
        Initialize the Model with configuration and artifacts.

        Args:
            config: Model configuration dictionary
            docs_path: Path to documents directory
            model_path: Path to local model file (optional)
            secrets: Dictionary containing secrets (optional)
        """
        self.model_config = config
        self.docs_path = docs_path
        self.model_path = model_path
        self.secrets = secrets

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
            self._load_embeddings()
            self._load_vectordb()
            self._load_model()
            self._load_prompt()
            self._load_chain()

            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Model: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

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

    def _load_embeddings(self) -> None:
        """Load HuggingFace embeddings model."""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": False}

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            logger.info(f"Embeddings model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise

    def _load_vectordb(self) -> None:
        """Load vector database from documents."""
        try:
            # Load documents from the docs path
            documents = []
            for filename in os.listdir(self.docs_path):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(self.docs_path, filename)
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {filename}")

            if not documents:
                raise ValueError(f"No PDF documents found in {self.docs_path}")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            doc_splits = text_splitter.split_documents(documents)
            logger.info(f"Documents split into {len(doc_splits)} chunks")

            # Create vector database
            vectordb_path = os.path.join(os.path.dirname(self.docs_path), "vectordb")
            self.vectordb = Chroma.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                persist_directory=vectordb_path,
            )
            self.vectordb.persist()

            # Create retriever
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})
            logger.info("Vector database and retriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load the appropriate model based on configuration."""
        try:
            model_source = self.model_config.get("model_source", "local")
            logger.info(f"Loading model with source: {model_source}")

            from src.utils import initialize_llm, DEFAULT_MODELS

            # Extract secrets and model path based on configuration
            secrets = self.secrets if self.secrets else {}
            # Use model_path from notebook if provided, otherwise fall back to default
            local_model_path = (
                self.model_path if self.model_path else DEFAULT_MODELS["local"]
            )
            logger.info(f"Using local_model_path: {local_model_path}")

            hf_repo_id = self.model_config.get("hf_repo_id", "")

            # Use the shared initialize_llm function
            self.llm = initialize_llm(
                model_source=model_source,
                secrets=secrets,
                local_model_path=local_model_path,
                hf_repo_id=hf_repo_id,
            )

            if self.llm is None:
                logger.error("Model failed to initialize - llm is None after loading")
                raise RuntimeError("Model initialization failed - llm is None")

            logger.info(f"Model of type {type(self.llm).__name__} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_prompt(self) -> None:
        """Load the prompt template for the service."""
        try:
            self.prompt_str = """
            You are a friendly and professional AI assistant specialized in answering questions about Z by HP AI Studio. Your primary source of information is the provided documentation. Follow these guidelines:

            1. **Base your answers primarily on the provided context** from the Z by HP AI Studio documentation
            2. **Be helpful and informative** while maintaining accuracy
            3. **If information isn't in the context**, politely mention that and provide general guidance if appropriate
            4. **Use a conversational and professional tone**
            5. **Structure your responses clearly** with bullet points or numbered lists when helpful
            6. **Reference specific features or capabilities** mentioned in the documentation when relevant

            Context from Z by HP AI Studio documentation:
            {context}

            Human: {query}

            Assistant: Based on the Z by HP AI Studio documentation provided, I'll help answer your question.
            """

            self.prompt = ChatPromptTemplate.from_template(self.prompt_str)
            logger.info("Prompt template loaded successfully")

        except Exception as e:
            logger.error(f"Error loading prompt: {str(e)}")
            raise

    def _load_chain(self) -> None:
        """Create the RAG chain using the loaded model, retriever, and prompt."""
        try:
            if not self.retriever:
                raise ValueError(
                    "Retriever must be initialized before creating the chain"
                )

            # Get model context window
            context_window = get_context_window(self.llm)
            logger.info(f"Using model with context window of {context_window} tokens")

            input_normalizer = RunnableLambda(
                lambda x: {"input": x} if isinstance(x, str) else x
            )

            # Use dynamic retriever based on context window
            def context_aware_retrieval(x):
                return dynamic_retriever(
                    x["input"], collection=self.vectordb, context_window=context_window
                )

            # Use adaptive context formatter
            def adaptive_format(docs):
                return format_docs_with_adaptive_context(
                    docs, context_window=context_window
                )

            retriever_runnable = RunnableLambda(context_aware_retrieval)
            format_docs_r = RunnableLambda(adaptive_format)
            extract_input = RunnableLambda(lambda x: x["input"])

            self.chain = (
                input_normalizer
                | RunnableMap(
                    {
                        "context": retriever_runnable | format_docs_r,
                        "query": extract_input,
                    }
                )
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            logger.info("RAG chain loaded successfully")

        except Exception as e:
            logger.error(f"Error loading chain: {str(e)}")
            raise

    def predict(self, model_input, params=None):
        """
        Process inputs and generate responses.
        Must return pandas.DataFrame matching original signature.

        Args:
            model_input: Input data containing query, prompt, or document
            params: Optional parameters for different operations

        Returns:
            pandas.DataFrame with columns: chunks, history, prompt, output, success
        """
        try:
            # Initialize default parameters
            if params is None:
                params = {}

            # Handle different operations based on params
            if params.get("get_prompt", False):
                result = self._get_prompt_template()
            elif params.get("set_prompt", False) and "prompt" in model_input:
                result = self._set_prompt_template(model_input["prompt"][0])
            elif params.get("reset_history", False):
                result = self._reset_history()
            elif params.get("add_pdf", False) and "document" in model_input:
                result = self._add_pdf(model_input["document"][0])
            elif params.get("get_model_info", False):
                result = self._get_model_info()
            # Standard query operation
            elif "query" in model_input:
                result = self._inference(model_input["query"][0])
            else:
                result = {
                    "chunks": [],
                    "history": [],
                    "prompt": self.prompt_str,
                    "output": "Error: No valid operation specified in the request.",
                    "success": False,
                }
        except Exception as e:
            import traceback

            result = {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str if hasattr(self, "prompt_str") else "",
                "output": f"Error: {str(e)}\nTraceback: {traceback.format_exc()}",
                "success": False,
            }

        return pd.DataFrame([result])

    def _inference(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and generate a response."""
        try:
            logger.info(f"Processing query: {user_query}")

            # Get response from chain
            response = self.chain.invoke(user_query)

            # Add to memory
            self.memory.append({"role": "user", "content": user_query})
            self.memory.append({"role": "assistant", "content": response})

            # Get retrieved chunks for transparency
            retrieved_docs = dynamic_retriever(
                user_query,
                collection=self.vectordb,
                context_window=get_context_window(self.llm),
            )
            chunks = [doc.page_content for doc in retrieved_docs]

            return {
                "chunks": chunks,
                "history": [f"<{m['role']}> {m['content']}\n" for m in self.memory],
                "prompt": self.prompt_str,
                "output": response,
                "success": True,
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": error_msg,
                "success": False,
            }

    def _add_pdf(self, base64_pdf: str) -> Dict[str, Any]:
        """Add a PDF document to the knowledge base."""
        try:
            logger.info("Adding PDF to knowledge base")

            # Decode base64 PDF
            pdf_data = base64.b64decode(base64_pdf)

            # Save to temporary file
            temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
            temp_path = os.path.join(self.docs_path, temp_filename)

            with open(temp_path, "wb") as f:
                f.write(pdf_data)

            # Load and process the new document
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            doc_splits = text_splitter.split_documents(docs)

            # Add to vector database
            self.vectordb.add_documents(doc_splits)
            self.vectordb.persist()

            logger.info(f"Successfully added PDF with {len(doc_splits)} chunks")

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Successfully added PDF document with {len(doc_splits)} chunks to the knowledge base.",
                "success": True,
            }

        except Exception as e:
            error_msg = f"Error adding PDF: {str(e)}"
            logger.error(error_msg)
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": error_msg,
                "success": False,
            }

    def _get_prompt_template(self) -> Dict[str, Any]:
        """Get the current prompt template."""
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": f"Current prompt template: {self.prompt_str}",
            "success": True,
        }

    def _set_prompt_template(self, new_prompt: str) -> Dict[str, Any]:
        """Set a new prompt template."""
        try:
            self.prompt_str = new_prompt
            self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

            # Rebuild the chain with new prompt
            self._load_chain()

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Prompt template updated successfully.",
                "success": True,
            }
        except Exception as e:
            error_msg = f"Error updating prompt template: {str(e)}"
            logger.error(error_msg)
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": error_msg,
                "success": False,
            }

    def _reset_history(self) -> Dict[str, Any]:
        """Reset the conversation history."""
        self.memory = []
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "Conversation history has been reset.",
            "success": True,
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        try:
            context_window = get_context_window(self.llm)
            model_type = type(self.llm).__name__

            # Get additional info based on model type
            additional_info = {}
            if hasattr(self.llm, "model_path"):
                additional_info["model_path"] = self.llm.model_path
            if hasattr(self.llm, "repo_id"):
                additional_info["repo_id"] = self.llm.repo_id

            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Model type: {model_type}, Context window: {context_window} tokens",
                "additional_info": additional_info,
                "success": True,
            }
        except Exception as e:
            return {
                "chunks": [],
                "history": [],
                "prompt": self.prompt_str,
                "output": f"Error retrieving model info: {str(e)}",
                "success": False,
            }
