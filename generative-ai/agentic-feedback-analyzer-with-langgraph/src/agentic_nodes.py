# ─────── Standard Library Imports ───────
import logging  # Logging system for status and debugging output
import time  # Time tracking and delays
from datetime import datetime  # Handling date and time objects
from pathlib import Path  # Filesystem path abstraction
from typing import Any, Dict, List, Literal, Optional, TypedDict  # Type annotations for clarity and safety

# ─────── Third-Party Package Imports ───────
from tqdm import tqdm  # Visual progress bar for iterables
from langchain.docstore.document import Document  # Standardized document format
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking utility

# ─────── Local Application-Specific Imports ───────
from src.agentic_state import AgenticState  # Manages shared state across agent workflow
from src.utils import get_response_from_llm, log_timing, logger  # Core utilities: LLM calls, logging, and timers

@log_timing
def ingest_question(state: AgenticState) -> Dict[str, Any]:
    """
    Record the user's input question and append it to the system message history.

    Returns:
        Dict[str, Any]: A dictionary with the updated message history.
    """
    question = state["question"].strip()
    logger.info("🗣️ Ingested user question: %s", question)

    # Build updated message history
    messages = state.get("messages", [])
    messages += [
        {"role": "developer", "content": "User submitted a question."},
        {"role": "user", "content": question},
    ]

    return {"messages": messages}


@log_timing
def check_relevance(state: AgenticState) -> Dict[str, Any]:
    """
    Determines whether the user's question is relevant to the given topic using the LLM.

    If irrelevant, returns a polite default answer and flags the result.
    """
    topic = state["topic"]
    question = state["question"]
    llm = state["llm"]

    # Define strict classification prompts
    system_prompt = (
        "You are a binary classification assistant designed to evaluate the relevance of user questions "
        "to a specified topic in a document intelligence system.\n\n"
        "Your task is to determine whether a user's question is relevant to the topic — either directly or indirectly — "
        "based on whether the question could help support, relate to, or expand a conversation or analysis about that topic.\n\n"
        "Rules:\n"
        "- Only respond with 'yes' or 'no' — no punctuation, no elaboration, and no additional words.\n"
        "- If the question is clearly about the topic, answer 'yes'.\n"
        "- If the question does not mention the topic directly but can logically contribute to answering, clarifying, or deepening understanding about the topic, answer 'yes'.\n"
        "- Only answer 'no' if the question is completely unrelated or disconnected from the topic and cannot be used in any meaningful way to explore it.\n\n"
        "Your judgment should be inclusive — if there's any reasonable connection or utility, classify it as relevant ('yes')."
    )


    user_prompt = (
        f"Topic: \"{topic}\"\n"
        f"User's Question: \"{question}\"\n\n"
        "Determine if the question is relevant to the topic.\n"
        "A question is considered relevant if it:\n"
        "- Directly asks about the topic, or\n"
        "- Could be used to inform, support, or guide a discussion, analysis, or answer related to the topic.\n\n"
        "If there's any logical connection between the question and the topic — even if implicit — respond with 'yes'.\n"
        "Respond strictly with one word: 'yes' or 'no'.\n"
        "Answer:"
    )


    # Get LLM response
    response = get_response_from_llm(llm, system_prompt, user_prompt).strip().lower()
    is_relevant = response == "yes"

    logger.info("🧠 Relevance response: %s → %s", response, "Relevant" if is_relevant else "Irrelevant")

    # Append LLM trace and result
    messages = state.get("messages", [])
    messages += [
        {"role": "developer", "content": "🧠 Relevance check result:"},
        {"role": "assistant", "content": response},
    ]

    result: Dict[str, Any] = {
        "is_relevant": is_relevant,
        "messages": messages,
    }

    if not is_relevant:
        result["answer"] = f"🚫 Sorry, I can only answer questions related to '{topic}'."

    return result


@log_timing
def check_memory(state: AgenticState) -> Dict[str, Any]:
    """
    Check if the user's question has been previously answered and cached in memory.

    If found, return the cached answer along with a `from_memory` flag.
    """
    question = state["question"]
    memory = state["memory"]

    key = question.strip().lower()
    cached_answer = memory.get(key)

    messages = state.get("messages", [])

    if cached_answer:
        logger.info("💾 Cache hit for question: %s", question)
        messages.append({
            "role": "developer",
            "content": f"💾 Retrieved cached answer for question: '{question}'"
        })
        return {
            "answer": cached_answer,
            "from_memory": True,
            "messages": messages
        }

    logger.info("🧭 Cache miss for question: %s", question)
    messages.append({
        "role": "developer",
        "content": f"🧭 No cached answer found for question: '{question}'"
    })
    return {
        "from_memory": False,
        "messages": messages
    }


@log_timing
def rewrite_question(state: AgenticState) -> Dict[str, Any]:
    """
    Refines the user's original question into a clear, specific, and LLM-optimized form.

    Returns:
        Dict[str, Any]: Contains the rewritten question and updated message history.
    """
    original_question = state["question"].strip()
    llm = state["llm"]

    # Prompt engineering
    system_prompt = (
        "You are a professional assistant that rewrites vague or ambiguous questions "
        "into clear, focused, and LLM-friendly formats.\n"
        "The rewritten question must:\n"
        "- Be specific to document analysis\n"
        "- Be grammatically correct\n"
        "- Remain a QUESTION (not a statement)\n"
        "- Avoid ambiguity or conversational phrasing\n"
        "Do NOT include explanations or formatting—just return the cleaned question."
    )

    user_prompt = (
        f"Original user question:\n\"{original_question}\"\n\n"
        "Rewrite the question above as a clear and concise instruction for an AI to answer using document content. "
        "Ensure it remains in question form, not declarative."
    )

    # Run LLM
    response = get_response_from_llm(llm, system_prompt, user_prompt)
    rewritten = response.strip()

    # Log and message updates
    logger.info("✏️ Rewritten user question:\n→ %s", rewritten)

    messages = state.get("messages", [])
    messages += [
        {"role": "developer", "content": "✏️ Rewritten user question:"},
        {"role": "assistant", "content": rewritten},
    ]

    return {
        "rewritten_question": rewritten,
        "messages": messages,
    }


@log_timing
def create_chunks(state: AgenticState) -> Dict[str, Any]:
    """
    Split all loaded documents into semantically coherent, overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter to preserve context boundaries
    and control token limits for downstream LLM usage.

    Returns:
        Dict[str, Any]: Contains a "chunks" key with the resulting split Document list.
    """
    docs = state["docs"]
    logger.info("📑 Starting chunking for %d loaded documents", len(docs))
    CHUNK_SIZE = 4096
    CHUNK_OVERLAP = 256

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # Order: most to least semantic
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    logger.info("🧩 Created %d total chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    # Append developer message
    messages = state.get("messages", [])
    messages.append({
        "role": "developer",
        "content": f"🧩 Chunked {len(docs)} documents into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    })

    return {
        "chunks": chunks,
        "messages": messages,
    }


@log_timing
def generate_answer_per_chunks(state: AgenticState) -> Dict[str, Any]:
    """
    Generate an answer for each document chunk based on the rewritten question.

    Each chunk is independently evaluated by the LLM using a shared system/user prompt pair.
    Only the information inside the chunk may be used—no inference or guessing allowed.
    """
    rewritten_question = state["rewritten_question"].strip()
    chunks = state["chunks"]
    llm = state["llm"]
    topic = state["topic"]
    
    logger.info("🧩 Generating answers for %d chunks using rewritten question: '%s'", len(chunks), rewritten_question)

    # 🔒 System Prompt (invariant per chunk)
    system_prompt = (
        "You are a document chunk analysis assistant for a feedback intelligence system.\n\n"
        "Your task is to help answer a specific user question by analyzing **one chunk** of a larger document.\n\n"
        "Instructions:\n"
        "- You are only given one chunk at a time.\n"
        "- Use only the information in that chunk to answer.\n"
        "- Do NOT guess, infer, or draw conclusions from missing context.\n"
        "- If the chunk does not provide enough information to answer, reply exactly with: Not mentioned in this chunk.\n\n"
        "Output Guidelines:\n"
        "- Be factual, clear, and complete.\n"
        "- Use relevant details from the chunk if available.\n"
        "- If the question is not answerable from the chunk, respond accordingly and concisely.\n"
    )

    # 🧾 User Prompt Template
    user_prompt_template = (
        f"User question:\n"
        f"\"{rewritten_question}\"\n\n"
        f"This chunk is part of a document about the topic: \"{topic}\".\n"
        f"Read the chunk carefully and answer the question using only what is written below:\n\n"
        f"--- START OF CHUNK ---\n"
        f"{{chunk}}\n"
        f"--- END OF CHUNK ---\n\n"
        f"If the answer is not found in this chunk, respond with:\n"
        f"Not mentioned in this chunk."
    )

    chunk_responses = []
    messages = state.get("messages", [])

    progress_bar = tqdm(chunks, desc="🔁 Processing each chunk")

    for i, chunk in enumerate(progress_bar):
        chunk_text = chunk.page_content.strip()
        user_prompt = user_prompt_template.replace("{chunk}", chunk_text)

        try:
            response = get_response_from_llm(llm, system_prompt, user_prompt).strip()
            progress_bar.set_postfix({"group": f"✅ Chunk {i + 1} response length: {len(response)} chars"})
        except Exception as e:
            response = f"[ERROR in chunk {i+1}]: {e}"
            progress_bar.set_postfix({"group": f"❌ Error processing chunk {i + 1}: {e}"})

        chunk_responses.append(response)

    logger.info("🧠 Finished generating %d chunk-level responses.", len(chunk_responses))

    # Add summary message
    messages.append({
        "role": "developer",
        "content": f"🧠 Processed {len(chunks)} chunks for question: '{rewritten_question}'"
    })

    return {
        "chunk_responses": chunk_responses,
        "messages": messages,
    }

@log_timing
def generate_synthetized_answer(state: AgenticState) -> Dict[str, Any]:
    """
    Synthesizes a final, comprehensive answer to the user's question
    based on the collected per-chunk LLM responses.
    """
    chunk_answers = state["chunk_responses"]
    rewritten_question = state["rewritten_question"]
    topic = state["topic"]
    llm = state["llm"]

    if not chunk_answers:
        logger.warning("🚫 No chunk-level responses available to synthesize.")
        return {"answer": "No information available to synthesize a final answer."}

    max_context_tokens = getattr(llm, "context_window", 8192)
    chunk_token_budget = max_context_tokens // 2

    # 🧱 Split formatted chunk answers into token-safe groups
    def chunk_by_token_limit(answers: List[str], max_tokens: int) -> List[List[str]]:
        groups, current, current_len = [], [], 0
        for a in answers:
            a = a.strip()
            tokens = int(a.count(" ") * 1.5)
            if current_len + tokens > max_tokens and current:
                groups.append(current)
                current, current_len = [], 0
            current.append(a)
            current_len += tokens
        if current:
            groups.append(current)
        return groups

    grouped_chunks = chunk_by_token_limit(chunk_answers, chunk_token_budget)

    logger.info("🧠 Synthesizing across %d chunk groups", len(grouped_chunks))

    logger.info("🧠 Synthesizing final answer from %d chunk responses", len(chunk_answers))

    # 🧠 System Prompt (for synthesis agent)
    synthesis_system_prompt = (
        "You are a synthesis assistant in a document reasoning system.\n\n"
        "Your job is to produce a final, complete, and precise answer to the user's question "
        "based on multiple intermediate answers derived from different chunks of a document.\n\n"
        "Instructions:\n"
        "- Use ONLY the information provided in the chunk responses.\n"
        "- Do NOT hallucinate, invent, or infer beyond what's included.\n"
        "- Eliminate redundancy and merge overlapping information.\n"
        "- Combine details and structure them clearly.\n"
        "- Be detailed, factual, and coherent.\n"
        "- Avoid repeating redundant or identical statements from the chunks; instead, consolidate and rephrase them concisely.\n"
        "- Format the answer using valid and clean Markdown for headings, lists, and emphasis.\n"
        "- **Return the final answer in clean and well-formatted Markdown.**\n"
    )

    progress_bar = tqdm(grouped_chunks, desc="🔁 Processing each grouped chunk answers")

    partial_summaries = [] 

    for i, chunk_group in enumerate(progress_bar):
        formatted_chunks = "\n".join(f"- Chunk {j+1}: {a}" for j, a in enumerate(chunk_group))
        # 💬 User Prompt Template
        synthesis_user_prompt = (
            f"The user asked the following question:\n"
            f"\"{rewritten_question}\"\n\n"
            f"The topic of the document is: \"{topic}\"\n\n"
            f"Below are the LLM-generated answers for each chunk:\n\n"
            f"{formatted_chunks}\n\n"
            f"Please now synthesize a final, complete, non-redundant answer to the user's question. "
            f"Make sure your answer is factual, logically structured, and clearly written."
            f"Avoid repeating redundant or identical statements from the chunks; instead, consolidate and rephrase them concisely."
            f"If the chunks provide conflicting answers, prioritize the most consistent and complete one. "
            f"\n\n➡️ **Return the final answer in clean and well-formatted Markdown.**"
        )
        summary = get_response_from_llm(
            llm=llm,
            system_prompt=synthesis_system_prompt,
            user_prompt=synthesis_user_prompt,
            ).strip()
        
        progress_bar.set_postfix({"group": f"🧠 Synthesized partial answer ({i + 1}/{len(grouped_chunks)})"})
        
        partial_summary = f"# 🧠 Synthesized partial answer ({i + 1}/{len(grouped_chunks)})\n\n" + summary
        partial_summaries.append(partial_summary)
        
    logger.info(f"✅ Synthesized {len(partial_summaries)} group-level summaries.")

    final_answer = "\n---\n".join(partial_summaries)

    messages = state.get("messages", [])
    messages += [
        {"role": "developer", "content": f"✅ Synthesized {len(partial_summaries)} group-level summaries."},
        {"role": "assistant", "content": final_answer}
    ]

    return {
        "answer": final_answer,
        "messages": messages,
    }


@log_timing
def update_memory(state: AgenticState) -> Dict[str, Any]:
    """
    Persist the current question-answer pair to memory if it was not served from cache.

    Returns:
        Dict[str, Any]: Updated message history (if applicable).
    """
    if state.get("from_memory"):
        logger.info("⏩ Skipping memory update (already served from cache).")
        return {}

    question = state["question"].strip().lower()
    answer = state["answer"]
    memory = state["memory"]

    memory.set(question, answer)
    logger.info("💾 Stored question-answer pair in memory (key: %s)", question)

    messages = state.get("messages", [])
    messages.append({
        "role": "developer",
        "content": f"💾 Stored answer in memory for question key: '{question}'"
    })

    return {"messages": messages}


@log_timing
def output_answer(state: AgenticState) -> Dict[str, Any]:
    """
    Display the final synthesized answer and record the action in the developer trace.

    Returns:
        Dict[str, Any]: Contains updated message history.
    """
    answer = state.get("answer", "").strip()

    # Display output in console (or could be adapted for Streamlit, CLI, etc.)
    print("\n🔚 === Final Answer ===\n")
    print(answer)
    print("\n========================\n")

    logger.info("📤 Delivered final answer (%d characters)", len(answer))

    # Append developer message to trace
    messages = state.get("messages", [])
    messages.append({
        "role": "developer",
        "content": f"📤 Final answer delivered: {answer}"
    })

    return {"messages": messages}
