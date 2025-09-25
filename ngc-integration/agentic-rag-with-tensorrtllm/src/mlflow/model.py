"""
Standalone Model class for Agentic RAG with TensorRT-LLM.

Business Logic Layer
- Handles complex agentic RAG workflow using LangGraph state machine
- Manages TensorRT-LLM integration, embeddings, vector database, and memory systems
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import json
import logging
import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import tensorrt_llm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

# Add the src directory to the path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from trt_llm_langchain import TensorRTLangchain

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles complex agentic RAG workflow with TensorRT-LLM.
    """

    TOPIC: str = "AI Studio"

    class SimpleKVMemory:
        """Very small persistent key-value store (JSON on disk)."""

        def __init__(self, file_path: Path) -> None:
            self.file_path: Path = file_path
            self._store: Dict[str, str] = self._load()

        # ---------- public ----------------------------------------------------
        def get(self, key: str) -> Optional[str]:
            """Return answer if present, else None."""
            return self._store.get(key)

        def set(self, key: str, value: str) -> None:
            """Save answer and flush to disk."""
            self._store[key] = value
            self._dump()

        # ---------- private ---------------------------------------------------
        def _load(self) -> Dict[str, str]:
            if self.file_path.exists():
                try:
                    with self.file_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load memory (%s). Starting fresh.", exc)
            return {}

        def _dump(self) -> None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)

    class RAGState(TypedDict, total=False):
        topic: str
        query: str
        is_relevant: Optional[bool]
        rewritten_query: Optional[str]
        retrieved_chunks: List[str]
        answer: Optional[str]
        from_memory: Optional[bool]
        messages: List[Dict[str, Any]]  # full conversation with LLM

    def __init__(self, config: dict, chroma_dir: str, memory_path: str):
        """
        Direct dependency injection - no MLflow context.
        Extract all initialization logic from original service.
        """
        self.config = config
        self.TOPIC = Model.TOPIC

        # Set up logger
        self._logger = logging.getLogger("AgenticRAGModel")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        # 1. Load embedding model
        try:
            self._embed_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={"normalize_embeddings": True},
            )
        except:
            self._embed_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={"normalize_embeddings": True},
            )

        # 2. Load persisted Chroma vectorstore
        chroma_dir_path = Path(chroma_dir)
        self._vectorstore = Chroma(
            collection_name="-".join(self.TOPIC.split()),
            persist_directory=str(chroma_dir_path),
            embedding_function=self._embed_model,
        )

        # 3. Load LLM via TensorRTLangchain
        sampling_params = tensorrt_llm.SamplingParams(
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.2,
            stop_token_ids=[128009],
        )
        self._llm = TensorRTLangchain(
            model_path="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
            sampling_params=sampling_params,
        )

        # 4. Initialize persistent memory
        memory_path_obj = Path(memory_path)
        memory_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not memory_path_obj.exists():
            memory_path_obj.write_text("{}", encoding="utf-8")
        self._memory = Model.SimpleKVMemory(memory_path_obj)

        # 5. Define a simple Response namedtuple
        self._LLMResponse = namedtuple("Response", ["content"])

        # 6. Build and compile the LangGraph state graph
        self._build_state_graph()

    # ----------------------------------------
    # Node Functions (each mirrors the notebook)
    # ----------------------------------------
    def ingest_query(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Log the incoming user query and record it in the message history.
        """
        user_query = state["query"]
        self._logger.info("Received user query: %s", user_query)
        previous_messages = state.get("messages", [])
        new_messages = previous_messages + [{"role": "user", "content": user_query}]
        return {"messages": new_messages}

    def check_relevance(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Ask the LLM whether the query relates to our topic.
        If not relevant, include a default apology answer.
        """
        topic = state["topic"]
        user_query = state["query"]

        system_prompt = (
            'You are a strict classifier. Only respond with either "yes" or "no". '
            "Do not include any additional words, explanations, or punctuation. "
            "Answer based solely on whether the user's query is about the specified topic."
        )
        user_prompt = (
            f'The topic is: "{topic}"\n\n'
            f'User query: "{user_query}"\n\n'
            "Is this query related to the topic above? Respond with only 'yes' or 'no'."
            "Answer: "
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt)
        is_relevant = "yes" in resp.strip().lower()
        self._logger.info("Relevance check result: %s", is_relevant)

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Relevance check result:"},
            {"role": "assistant", "content": resp},
        ]
        result: Dict[str, Any] = {"is_relevant": is_relevant, "messages": messages}
        if not is_relevant:
            result["answer"] = f"Sorry, I can only answer questions related to {topic}."
        return result

    def check_memory(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Look up the exact user query in memory and return the cached answer if found.
        """
        raw_query = state["query"]
        key = raw_query.strip().lower()
        cached_answer = self._memory.get(key)
        if cached_answer is not None:
            self._logger.info("Cache hit for query: %s", raw_query)
            return {"answer": cached_answer, "from_memory": True}
        self._logger.info("Cache miss for query: %s", raw_query)
        return {"from_memory": False}

    def rewrite_query(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Correct any grammar in the question and rewrite it as a clear statement
        without altering its meaning, to improve retrieval.
        """
        original = state["query"]
        system_prompt = (
            "You are a rewriting assistant. Your only task is to convert a question into a "
            "grammatically correct statement. Do not change its meaning. "
            "Output only the corrected statement—no explanations or extra text."
        )
        user_prompt = (
            "Convert the following question into a grammatically correct statement "
            "that preserves the original meaning exactly:\n\n"
            "Note: Output only the corrected statement—no explanations or extra text.\n"
            f'Question: "{original}"\n\n'
            "Corrected Statement:"
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt).strip()
        self._logger.info("Rewritten query: %s", resp)

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Rewritten query:"},
            {"role": "assistant", "content": resp},
        ]
        return {"rewritten_query": resp, "messages": messages}

    def retrieve_chunks(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Fetch the top-k most relevant chunks for the rewritten query.
        """
        statement = state["rewritten_query"]
        docs = self._vectorstore.similarity_search(statement, k=5)
        chunks = [doc.page_content for doc in docs]
        self._logger.info("Retrieved %d chunks for query.", len(chunks))
        return {"retrieved_chunks": chunks}

    def generate_answer(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Use the LLM to generate an answer based solely on retrieved context.
        """
        topic = state["topic"]
        user_query = state["query"]
        context = "\n\n---\n\n".join(state["retrieved_chunks"])

        system_prompt = (
            f"You are a knowledgeable assistant specialized in {topic}. Your task is to answer "
            "the user query using only the information found within the <context> block. "
            'Ignore any external knowledge. If the context does not contain the answer, reply exactly with: "I don\'t know." '
            "Do not assume, infer, or add any extra information. "
            "Respond with only the answer—do not include any introductory or explanatory text."
        )
        user_prompt = (
            f"<context>\n{context}\n</context>\n\n"
            f'User query: "{user_query}"\n\n'
            "Based only on the context above, provide the exact answer to the query. "
            'If the context does not contain the answer, respond exactly with: "I don\'t know." '
            "Give only the answer—do not include any intro phrases such as 'The answer is' or 'Here it is'."
            "Answer: "
        )

        resp = self._get_response_from_llm(system_prompt, user_prompt).strip()
        self._logger.info("Generated answer (%d chars)", len(resp))

        messages = state.get("messages", []) + [
            {"role": "developer", "content": "Generated answer:"},
            {"role": "assistant", "content": resp},
        ]
        return {"answer": resp, "messages": messages}

    def update_memory(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        Store new query-answer pairs in memory for faster future lookup.
        """
        if state.get("from_memory"):
            return {}
        raw_query = state["query"]
        key = raw_query.strip().lower()
        answer = state["answer"]
        if answer is not None:
            self._memory.set(key, answer)
            self._logger.info("Stored query-answer in memory for key: %s", key)
        return {}

    def output_answer(self, state: "Model.RAGState") -> Dict[str, Any]:
        """
        The final node. We do not print to STDOUT when serving via MLflow.
        Just return an empty dict as this node does not add new state.
        """
        return {}

    # ----------------------------------------
    # Helper Methods
    # ----------------------------------------
    def _get_response_from_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Wrap the LLM call into the meta-prompt format and return the .content string.
        """
        meta_llama_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        raw = self._llm(meta_llama_prompt)
        # TensorRTLangchain returns a raw string; we can wrap into Response if needed
        return raw

    def _route_relevance(self, state: "Model.RAGState") -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def _route_memory(self, state: "Model.RAGState") -> str:
        return "cached" if state.get("from_memory") else "not_cached"

    def _build_state_graph(self) -> None:
        """
        Construct and compile the LangGraph state graph exactly as in the notebook.
        """
        rag_graph = StateGraph(Model.RAGState)

        # Add nodes
        rag_graph.add_node("ingest_query", self.ingest_query)
        rag_graph.add_node("check_relevance", self.check_relevance)
        rag_graph.add_node("rewrite_query", self.rewrite_query)
        rag_graph.add_node("check_memory", self.check_memory)
        rag_graph.add_node("retrieve_chunks", self.retrieve_chunks)
        rag_graph.add_node("generate_answer", self.generate_answer)
        rag_graph.add_node("update_memory", self.update_memory)
        rag_graph.add_node("output_answer", self.output_answer)

        # Add edges
        rag_graph.add_edge(START, "ingest_query")
        rag_graph.add_edge("ingest_query", "check_relevance")

        rag_graph.add_conditional_edges(
            "check_relevance",
            self._route_relevance,
            {
                "irrelevant": "output_answer",
                "relevant": "check_memory",
            },
        )

        rag_graph.add_conditional_edges(
            "check_memory",
            self._route_memory,
            {
                "cached": "output_answer",
                "not_cached": "rewrite_query",
            },
        )

        rag_graph.add_edge("rewrite_query", "retrieve_chunks")
        rag_graph.add_edge("retrieve_chunks", "generate_answer")
        rag_graph.add_edge("generate_answer", "update_memory")
        rag_graph.add_edge("update_memory", "output_answer")
        rag_graph.add_edge("output_answer", END)

        # Compile graph
        self._compiled_graph = rag_graph.compile()

    def predict(self, model_input, params=None):
        """
        Core business logic extracted from original service predict method.
        Remove context parameter - use instance variables instead.
        Must return same pandas.DataFrame structure as original.
        """
        # Handle pandas DataFrame input
        if isinstance(model_input, pd.DataFrame):
            if "query" not in model_input.columns:
                raise Exception("DataFrame must contain a 'query' column.")
            # Take the first record in that column
            raw_query = model_input["query"].iloc[0]
        else:
            # Could be a plain dict or something else
            if not isinstance(model_input, dict):
                raise Exception(
                    f"Unexpected input type: {type(model_input)}. "
                    "Expected pandas.DataFrame or dict with 'query'."
                )
            # If it's a dict, accept either string or single-element list
            if "query" not in model_input:
                raise Exception("Input dict must contain key 'query'.")
            raw_query = model_input["query"]

        # Initialize state with topic, query, and empty messages
        initial_state: Model.RAGState = {
            "topic": self.TOPIC,
            "query": raw_query.strip(),
            "messages": [],
        }

        # Invoke the compiled LangGraph
        final_state = self._compiled_graph.invoke(input=initial_state)

        # Extract elements to return
        answer = final_state.get("answer", "")
        retrieved_chunks = final_state.get("retrieved_chunks", [])
        messages = final_state.get("messages", [])

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "messages": messages,
        }
