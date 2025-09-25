from typing import TypedDict, Optional, List, Dict, Any
from typing import Annotated
import operator
import numpy as np
from langgraph.graph import StateGraph, END
from src.segment_audio_embeddings import (
    retrieve_audio_segments_from_index,
    rerank_hits_mmr,
)

Messages = Annotated[List[Dict[str, Any]], operator.add]


class AudioState(TypedDict, total=False):
    question: str
    file_id: str
    memory: Any
    audio_llm: Any
    is_relevant: bool
    from_memory: bool
    hits_raw: List[Dict[str, Any]]
    hits: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    answer: str
    messages: Messages


def build_audio_agentic_graph(
    relevance_threshold: float,
    fetch_k: int,
    top_k: int,
    vecs: np.ndarray,
    metas: list[dict],
    audio_index,
    clap_processor,
    clap_model,
):
    def _mem_get(mem, key):
        if isinstance(mem, dict):
            return mem.get(key)
        return mem.get(key) if hasattr(mem, "get") else None

    def _mem_put(mem, key, value):
        if isinstance(mem, dict):
            mem[key] = value
            return
        if hasattr(mem, "set"):
            mem.set(key, value)
            return
        if hasattr(mem, "put"):
            mem.put(key, value)
            return
        raise RuntimeError("Unsupported memory object (no set/put)")

    def node_ingest_question(state: AudioState) -> AudioState:
        q = (state.get("question") or "").strip()
        if not q:
            raise ValueError("Empty question")
        return {
            "messages": [
                {"role": "developer", "content": "Ingested question"},
                {"role": "user", "content": q},
            ]
        }

    def node_check_relevance_audio(state: AudioState) -> AudioState:
        q = state["question"]
        probe = retrieve_audio_segments_from_index(
            audio_index, clap_processor, clap_model, q, top_k=5, fetch_k=8
        )
        max_score = max([h.get("score", 0.0) for h in probe], default=0.0)
        is_rel = bool(max_score >= relevance_threshold)
        updates: AudioState = {
            "is_relevant": is_rel,
            "messages": [
                {
                    "role": "developer",
                    "content": f"Relevance: max_score={max_score:.3f}->{'relevant' if is_rel else 'irrelevant'}",
                }
            ],
        }
        if not is_rel:
            updates["answer"] = (
                "🚫 Sorry, I can’t find anything relevant to that question in this media."
            )
        return updates

    def node_check_memory(state: AudioState) -> AudioState:
        q = (state.get("question") or "").strip().lower()
        fid = state.get("file_id", "global")
        key = f"{fid} :: {q}"
        cached = _mem_get(state.get("memory"), key)
        if cached:
            return {
                "from_memory": True,
                "answer": cached.get("answer", ""),
                "evidence": cached.get("evidence", []),
                "messages": [{"role": "developer", "content": f"Cache hit for {key}"}],
            }
        else:
            return {
                "from_memory": False,
                "messages": [{"role": "developer", "content": f"Cache miss for {key}"}],
            }

    def node_retrieve(state: AudioState) -> AudioState:
        hits_raw = retrieve_audio_segments_from_index(
            audio_index,
            clap_processor,
            clap_model,
            state["question"],
            top_k=top_k,
            fetch_k=fetch_k,
        )
        return {"hits_raw": hits_raw}

    def node_rerank(state: AudioState) -> AudioState:
        hits = rerank_hits_mmr(
            clap_processor,
            clap_model,
            state["question"],
            state.get("hits_raw", []),
            top_k=top_k,
            fetch_k=fetch_k,
            lam=0.6,
        )
        return {"hits": hits}

    def node_generate_audio_only(state: AudioState) -> AudioState:
        hits = state.get("hits", [])
        llm = state.get("audio_llm")
        if llm is None:
            raise RuntimeError(
                "Missing `audio_llm` (QwenOmniAgent). Pass it in the graph state."
            )
        out = llm.answer(state["question"], hits, return_audio=False)
        ev = out.get("evidence", [])
        for e in ev:
            if "score_mmr" in e:
                e["score"] = e["score_mmr"]
        return {"answer": out.get("answer", ""), "evidence": ev}

    def node_update_memory(state: AudioState) -> AudioState:
        q = state["question"].strip().lower()
        fid = state.get("file_id", "global")
        key = f"{fid} :: {q}"
        # write
        mem = state.get("memory")
        if mem is not None:
            val = {
                "answer": state.get("answer", ""),
                "evidence": state.get("evidence", []),
            }
            _mem_put(mem, key, val)
        return {}

    def node_output(state: AudioState) -> AudioState:
        return {}

    g = StateGraph(AudioState)
    g.add_node("ingest_question", node_ingest_question)
    g.add_node("check_relevance_audio", node_check_relevance_audio)
    g.add_node("check_memory", node_check_memory)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("generate_audio", node_generate_audio_only)
    g.add_node("update_memory", node_update_memory)
    g.add_node("output_answer", node_output)

    g.set_entry_point("ingest_question")
    g.add_edge("ingest_question", "check_relevance_audio")

    def after_relevance(state: AudioState):
        return "check_memory" if state.get("is_relevant") else "output_answer"

    g.add_conditional_edges(
        "check_relevance_audio",
        after_relevance,
        {"check_memory": "check_memory", "output_answer": "output_answer"},
    )

    def after_memory(state: AudioState):
        return "output_answer" if state.get("from_memory") else "retrieve"

    g.add_conditional_edges(
        "check_memory",
        after_memory,
        {"output_answer": "output_answer", "retrieve": "retrieve"},
    )

    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate_audio")
    g.add_edge("generate_audio", "update_memory")
    g.add_edge("update_memory", "output_answer")
    g.add_edge("output_answer", END)

    return g.compile()
