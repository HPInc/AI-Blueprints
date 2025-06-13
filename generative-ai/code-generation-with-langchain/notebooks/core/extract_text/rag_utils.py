"""
RAG utilities for code repository analysis.

This module extends the base utilities with retrieval and context handling
functions designed specifically for code repositories. Key features include:

1. Multi-document retrieval with metadata filtering
2. Hybrid search combining semantic and keyword matching
3. Query expansion for specific question types
4. File type prioritization based on question intent
5. Dynamic multi-document formatting for improved context
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File type classifications
FILE_TYPE_PRIORITIES = {
    "dependency": [
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "Pipfile",
        "environment.yml",
        "package.json",
        "package-lock.json",
        "yarn.lock"
    ],
    "configuration": [
        "config.yaml",
        "config.json",
        ".env",
        ".ini",
        "settings.py"
    ],
    "documentation": [
        "README.md",
        "docs/",
        ".md",
        ".rst"
    ],
    "code": [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".go",
        ".rs"
    ]
}

# Question type patterns for classification
QUESTION_TYPE_PATTERNS = {
    "dependency": [
        r"depend[ea]nc(y|ies)",
        r"(requires|required|requirement)",
        r"(package|module|library)",
        r"(install|setup)",
        r"(pip|npm|conda)"
    ],
    "usage": [
        r"(how to use|usage|example)",
        r"(call|invoke|execute)",
        r"(interface|api)"
    ],
    "implementation": [
        r"(implement|code|function)",
        r"(algorithm|logic|flow)",
        r"(architecture|design|pattern)"
    ],
    "error": [
        r"(error|exception|issue|bug|problem)",
        r"(fix|resolve|handle|debug)"
    ],
    "concept": [
        r"(what is|explain|concept)",
        r"(why|reason|purpose)",
        r"(difference|compare)"
    ]
}

def identify_question_type(query: str) -> List[str]:
    """
    Identify the type of question being asked.
    
    Args:
        query: The user's question
        
    Returns:
        List of matched question types, ordered by confidence
    """
    matched_types = {}
    query_lower = query.lower()
    
    for q_type, patterns in QUESTION_TYPE_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                matches += 1
        
        if matches > 0:
            matched_types[q_type] = matches
    
    # Sort by match count (descending)
    sorted_types = sorted(matched_types.items(), key=lambda x: x[1], reverse=True)
    return [q_type for q_type, _ in sorted_types]

def expand_query(query: str, question_types: List[str]) -> str:
    """
    Expand the query to improve retrieval based on question type.
    
    Args:
        query: Original user query
        question_types: List of identified question types
        
    Returns:
        Expanded query with additional relevant terms
    """
    expansions = []
    
    if "dependency" in question_types:
        expansions.extend([
            "requirements.txt", "setup.py", "dependencies", 
            "packages", "libraries", "imports", "installation"
        ])
    
    if "usage" in question_types:
        expansions.extend([
            "example", "usage", "how to use", "code sample", 
            "demonstration", "tutorial"
        ])
        
    if "implementation" in question_types:
        expansions.extend([
            "implementation", "code", "function", "class", 
            "method", "algorithm"
        ])
        
    if "error" in question_types:
        expansions.extend([
            "error", "exception", "bug", "issue", "fix", 
            "troubleshoot", "handle"
        ])
    
    # Add the most relevant expansions (avoid query explosion)
    max_expansions = 3
    selected_expansions = expansions[:max_expansions]
    
    # Create expanded query with original query first for weighting
    expanded_query = f"{query} {' '.join(selected_expansions)}"
    
    logger.info(f"Expanded query: '{expanded_query}'")
    return expanded_query

def classify_file_by_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance metadata with file type classification and other useful attributes.
    
    Args:
        metadata: Original document metadata
        
    Returns:
        Enhanced metadata with file_type and other attributes
    """
    enhanced_metadata = metadata.copy()
    
    # Extract filename from metadata
    filename = metadata.get("filename", "")
    if not filename and "filenames" in metadata:
        filename = metadata.get("filenames", "")
    
    # Default classification
    file_type = "unknown"
    contains_dependencies = False
    language = "unknown"
    
    # Classify file type
    if filename:
        # Check for dependency files
        for dep_file in FILE_TYPE_PRIORITIES["dependency"]:
            if dep_file in filename or filename.endswith(dep_file):
                file_type = "dependency"
                contains_dependencies = True
                break
                
        # Check for configuration files
        if file_type == "unknown":
            for config_file in FILE_TYPE_PRIORITIES["configuration"]:
                if config_file in filename or filename.endswith(config_file):
                    file_type = "configuration"
                    break
        
        # Check for documentation files
        if file_type == "unknown":
            for doc_file in FILE_TYPE_PRIORITIES["documentation"]:
                if doc_file in filename or filename.endswith(doc_file):
                    file_type = "documentation"
                    break
        
        # Identify language from extension
        extension = Path(filename).suffix
        if extension:
            if extension == '.py':
                language = 'python'
                # Check for setup.py which indicates dependencies
                if 'setup.py' in filename:
                    contains_dependencies = True
            elif extension in ['.js', '.ts']:
                language = 'javascript'
            elif extension in ['.java']:
                language = 'java'
            elif extension in ['.c', '.cpp', '.h']:
                language = 'c/c++'
            elif extension in ['.go']:
                language = 'go'
            elif extension in ['.rs']:
                language = 'rust'
            elif extension in ['.md', '.rst']:
                language = 'markdown'
            elif extension in ['.json', '.yaml', '.yml', '.toml']:
                language = 'data'
                file_type = 'configuration'
        
        # If we couldn't classify it but it has a code extension
        if file_type == "unknown" and extension in ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs']:
            file_type = "code"
    
    # Detect dependencies from content if available
    if 'page_content' in metadata:
        content = metadata['page_content']
        if isinstance(content, str):
            # Check for Python import statements
            if re.search(r'(import\s+\w+|from\s+\w+\s+import)', content):
                contains_dependencies = True
            # Check for JavaScript require/import statements
            elif re.search(r'(require\s*\(|import\s+\w+\s+from)', content):
                contains_dependencies = True
    
    # Add enhanced metadata
    enhanced_metadata["file_type"] = file_type
    enhanced_metadata["contains_dependencies"] = contains_dependencies
    enhanced_metadata["language"] = language
    
    return enhanced_metadata

def retriever(
    query: str, 
    collection, 
    top_n: int = 10, 
    context_window: int = None,
    metadata_filters: Dict[str, Any] = None
) -> List[Document]:
    """
    Retrieval function with metadata filtering and hybrid search.
    
    Args:
        query: The search query
        collection: Vector database collection to search in
        top_n: Maximum number of documents to retrieve
        context_window: Size of the model's context window in tokens
        metadata_filters: Optional filters to apply to the search
        
    Returns:
        List of Document objects containing relevant content
    """
    from langchain.schema import Document
    
    # Determine question types for query expansion
    question_types = identify_question_type(query)
    logger.info(f"Question types: {question_types}")
    
    # Expand query based on question type
    if question_types:
        expanded_query = expand_query(query, question_types)
    else:
        expanded_query = query
    
    # Retrieve more documents than needed for post-filtering (2x)
    initial_top_n = min(top_n * 2, 20)  # Limit to avoid excessive retrieval
    
    # Determine if we need to prioritize certain file types
    prioritize_file_types = []
    if "dependency" in question_types:
        prioritize_file_types = ["dependency", "configuration"]
    
    # Dynamically determine how many documents to retrieve based on context window
    if top_n is None:
        if context_window:
            # Larger context windows can handle more documents
            # Using a heuristic: 1 document per 1000 tokens of context
            # with a minimum of 5 and maximum of 15
            suggested_top_n = max(5, min(15, context_window // 1000))
            top_n = suggested_top_n
        else:
            # Default if we can't determine context window
            top_n = 8
    
    try:
        # Try direct retrieval from collection
        if hasattr(collection, 'as_retriever'):
            # It's a LangChain Chroma vector store
            retriever = collection.as_retriever(search_kwargs={"k": initial_top_n})
            documents = retriever.get_relevant_documents(expanded_query)
        elif hasattr(collection, '_collection'):
            # It's a direct ChromaDB collection
            results = collection._collection.query(
                query_texts=[expanded_query],
                n_results=initial_top_n,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to Document objects with distance scores
            documents = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i].copy() if results['metadatas'][0][i] else {}
                # Add distance score to metadata for re-ranking
                metadata['distance_score'] = results['distances'][0][i] if 'distances' in results else 1.0
                
                # Add the document
                documents.append(
                    Document(
                        page_content=str(results['documents'][0][i]),
                        metadata=metadata
                    )
                )
        else:
            # Try direct query as a fallback
            results = collection.query(
                query_texts=[expanded_query],
                n_results=initial_top_n,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to Document objects with distance scores
            documents = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i].copy() if results['metadatas'][0][i] else {}
                metadata['distance_score'] = results['distances'][0][i] if 'distances' in results else 1.0
                
                documents.append(
                    Document(
                        page_content=str(results['documents'][0][i]),
                        metadata=metadata
                    )
                )
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        # If retrieval fails, try to fall back to the original dynamic_retriever
        from src.utils import dynamic_retriever
        try:
            documents = dynamic_retriever(query, collection, top_n=initial_top_n, context_window=context_window)
        except Exception:
            # If all methods fail, return empty list
            logger.error("All retrieval methods failed")
            return []
    
    # Enhance metadata with file type classification
    for i, doc in enumerate(documents):
        documents[i].metadata = classify_file_by_metadata(doc.metadata)
    
    # Apply metadata filters if provided
    if metadata_filters:
        filtered_documents = []
        for doc in documents:
            matches_all = True
            for key, value in metadata_filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    matches_all = False
                    break
            if matches_all:
                filtered_documents.append(doc)
        documents = filtered_documents
    
    # Re-rank documents based on:
    # 1. File type priority for the specific question type
    # 2. Original semantic similarity (distance_score)
    # 3. Contains_dependencies flag for dependency questions
    # 4. Exact keyword matches in content
    
    # Helper function to compute ranking score (lower is better)
    def compute_ranking_score(doc):
        score = doc.metadata.get('distance_score', 1.0)  # Base score from vector search
        
        # Prioritize by file type if needed
        if prioritize_file_types and doc.metadata.get('file_type') in prioritize_file_types:
            priority_index = prioritize_file_types.index(doc.metadata.get('file_type'))
            score -= 0.3 * (1.0 / (priority_index + 1))  # Higher priority = lower score
        
        # Boost dependency files for dependency questions
        if "dependency" in question_types and doc.metadata.get('contains_dependencies', False):
            score -= 0.2
            
        # Exact keyword match bonus (case insensitive)
        if any(keyword.lower() in doc.page_content.lower() for keyword in query.split()):
            score -= 0.1
            
        return score
    
    # Re-rank documents
    documents = sorted(documents, key=compute_ranking_score)
    
    # Limit to top_n documents after re-ranking
    documents = documents[:top_n]
    
    # Group documents by file type for better context organization
    grouped_docs = {}
    for doc in documents:
        file_type = doc.metadata.get('file_type', 'unknown')
        if file_type not in grouped_docs:
            grouped_docs[file_type] = []
        grouped_docs[file_type].append(doc)
    
    # Flatten grouped documents while maintaining priority order
    priority_order = ["dependency", "configuration", "code", "documentation", "unknown"]
    sorted_docs = []
    
    for file_type in priority_order:
        if file_type in grouped_docs:
            sorted_docs.extend(grouped_docs[file_type])
    
    # Add any remaining document types not in priority list
    for file_type, docs in grouped_docs.items():
        if file_type not in priority_order:
            sorted_docs.extend(docs)
    
    # Log retrieval results
    logger.info(f"Retrieved {len(sorted_docs)} documents after filtering and re-ranking")
    
    return sorted_docs

def format_multi_doc_context(
    docs: List[Document], 
    question_types: List[str],
    context_window: int = None
) -> str:
    """
    Format multiple documents into a structured context that emphasizes key information.
    
    Args:
        docs: List of Document objects
        question_types: Identified question types to guide formatting
        context_window: Size of model's context window
        
    Returns:
        Formatted context string with hierarchical organization
    """
    if not docs:
        return "No relevant documents found."
    
    # Average tokens per character (approximation)
    chars_per_token = 4
    
    # Determine the maximum character budget based on context window
    if context_window:
        # Reserve 30% for the prompt template and response
        available_tokens = int(context_window * 0.7)
        max_total_chars = available_tokens * chars_per_token
    else:
        # Default conservative estimate if we don't know the context window
        max_total_chars = 10000
    
    # Allocate budget for file type sections
    # File type weight adjustments based on question type
    file_type_weights = {
        "dependency": 1.0,
        "configuration": 1.0,
        "code": 1.0,
        "documentation": 1.0,
        "unknown": 0.5
    }
    
    # Adjust weights based on question type
    if "dependency" in question_types:
        file_type_weights["dependency"] = 2.0
        file_type_weights["configuration"] = 1.5
    elif "implementation" in question_types:
        file_type_weights["code"] = 2.0
    elif "concept" in question_types:
        file_type_weights["documentation"] = 2.0
    
    # Group documents by file type
    file_type_docs = {}
    for doc in docs:
        file_type = doc.metadata.get("file_type", "unknown")
        if file_type not in file_type_docs:
            file_type_docs[file_type] = []
        file_type_docs[file_type].append(doc)
    
    # Calculate total weights for budget allocation
    total_weight = sum(
        file_type_weights.get(file_type, 0.5) * len(docs)
        for file_type, docs in file_type_docs.items()
    )
    
    # Build the formatted context with sections by file type
    formatted_sections = []
    total_chars = 0
    
    # Process file types in order of importance
    priority_order = ["dependency", "configuration", "code", "documentation", "unknown"]
    
    for file_type in priority_order:
        if file_type not in file_type_docs:
            continue
            
        type_docs = file_type_docs[file_type]
        
        # Skip if no documents of this type
        if not type_docs:
            continue
            
        # Calculate budget for this file type
        weight = file_type_weights.get(file_type, 0.5) * len(type_docs)
        type_budget = int((weight / total_weight) * max_total_chars)
        
        # Format header for this section
        section_name = file_type.capitalize()
        section_header = f"## {section_name} Files"
        
        # Format documents of this type
        file_contents = []
        used_chars = len(section_header) + 2  # Account for header
        
        for i, doc in enumerate(type_docs):
            # Get content and metadata
            content = doc.page_content
            metadata = doc.metadata
            filename = metadata.get("filename", "unknown_file")
            
            # Calculate budget for this document based on relevance ranking
            # First document gets more budget, then exponentially declining
            doc_weight = 1.0 / (i + 1)
            doc_budget = min(
                int(type_budget * doc_weight),  # Relevance-based allocation
                len(content) + len(filename) + 10,  # Don't allocate more than needed
                type_budget - used_chars  # Don't exceed remaining type budget
            )
            
            # Skip if no budget left
            if doc_budget <= 0:
                continue
                
            # Create document header with filename and metadata
            file_header = f"### {filename}"
            if metadata.get("language", "") != "unknown":
                file_header += f" ({metadata.get('language')})"
                
            # Add relevance info if available
            if 'distance_score' in metadata:
                # Lower score is better for distance
                relevance = max(0, min(100, int((1 - metadata['distance_score']) * 100)))
                file_header += f" - Relevance: {relevance}%"
                
            doc_chars = len(file_header) + 2  # Start counting chars
            
            # Truncate content if needed
            if len(content) + doc_chars > doc_budget:
                # Try to break at a logical point like a line break
                truncation_point = doc_budget - doc_chars - 15  # Allow for truncation message
                
                # Find a good break point - prefer newlines, then periods, then spaces
                last_newline = content[:truncation_point].rfind('\n')
                last_period = content[:truncation_point].rfind('.')
                last_space = content[:truncation_point].rfind(' ')
                
                # Use the best break point that's not too far from target (at least 80% of target)
                threshold = truncation_point * 0.8
                if last_newline > threshold:
                    truncation_point = last_newline + 1  # +1 to include the newline
                elif last_period > threshold:
                    truncation_point = last_period + 1  # +1 to include the period
                elif last_space > threshold:
                    truncation_point = last_space + 1  # +1 to include the space
                    
                content = f"{content[:truncation_point]}... (truncated)"
                
            # Create formatted document entry
            file_entry = f"{file_header}\n\n```\n{content}\n```"
            doc_chars += len(content) + 8  # Count content plus code block markers
            
            file_contents.append(file_entry)
            used_chars += doc_chars
            
            # Stop if we've exceeded the type budget
            if used_chars >= type_budget:
                break
                
        # Only add the section if we have content
        if file_contents:
            section_content = "\n\n".join([section_header] + file_contents)
            formatted_sections.append(section_content)
            total_chars += used_chars
            
        # Stop if we've exceeded the total budget
        if total_chars >= max_total_chars:
            break
            
    # Join all sections
    if formatted_sections:
        formatted_text = "\n\n".join(formatted_sections)
    else:
        formatted_text = "No relevant content found."
        
    return formatted_text

def process_repository_question(
    query: str,
    collection,
    context_window: int = None,
    top_n: int = None,
    metadata_filters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a question about a code repository, retrieving and formatting relevant documents.
    
    Args:
        query: The user's question
        collection: Vector database collection
        context_window: Size of model's context window
        top_n: Number of documents to retrieve (determined automatically if None)
        metadata_filters: Optional metadata filters
        
    Returns:
        Dict with question, formatted context, and related metadata
    """
    # Step 1: Identify question types for specialized handling
    question_types = identify_question_type(query)
    
    # Step 2: Retrieve relevant documents
    docs = retriever(
        query=query,
        collection=collection,
        top_n=top_n,
        context_window=context_window,
        metadata_filters=metadata_filters
    )
    
    # Step 3: Format documents into structured context
    formatted_context = format_multi_doc_context(
        docs=docs,
        question_types=question_types,
        context_window=context_window
    )
    
    # Step 4: Extract most relevant document for primary context
    primary_document = docs[0] if docs else None
    
    # Create result dictionary
    result = {
        "question": query,
        "context": formatted_context,
        "document_count": len(docs),
        "question_types": question_types,
    }
    
    # Add information from primary document if available
    if primary_document:
        result["primary_code"] = primary_document.page_content
        
        if primary_document.metadata:
            metadata = primary_document.metadata
            result["primary_filename"] = metadata.get("filename", "unknown_file")
            result["primary_file_type"] = metadata.get("file_type", "unknown")
            result["primary_language"] = metadata.get("language", "unknown")
    
    return result
