"""
RAG Utilities

Common functions for working with RAG systems across the course.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200,
               split_by_paragraph: bool = True) -> List[str]:
    """
    Split text into overlapping chunks with advanced options.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        split_by_paragraph: Whether to try to split at paragraph boundaries

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    if split_by_paragraph:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If paragraph is longer than chunk_size, we need to split it
                if len(para) > chunk_size:
                    para_chunks = chunk_text(para, chunk_size, chunk_overlap, False)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    else:
        # Standard character-based chunking with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 200:  # Skip very small chunks at the end
                break
            chunks.append(chunk)

        return chunks

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace, etc."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove extra whitespace at beginning and end
    text = text.strip()

    return text

def score_chunks(query: str, chunks: List[str], embeddings_model) -> List[Tuple[str, float]]:
    """Score chunks against a query using cosine similarity."""
    query_embedding = embeddings_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embeddings_model.encode(chunks, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((chunks[i], float(similarity)))

    # Sort by similarity (highest first)
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def format_sources(chunks_with_scores: List[Tuple[str, float]], n: int = 3) -> str:
    """Format top chunks as sources."""
    result = "Sources:\n\n"
    for i, (chunk, score) in enumerate(chunks_with_scores[:n]):
        result += f"Source {i+1} (Score: {score:.4f}):\n{chunk}\n\n"
    return result

def hybrid_search(query: str, chunks: List[str], embeddings_model,
                  keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Tuple[str, float]]:
    """
    Perform hybrid search combining keyword and semantic searches.

    Args:
        query: Search query
        chunks: List of text chunks to search
        embeddings_model: Model for creating embeddings
        keyword_weight: Weight for keyword search component
        semantic_weight: Weight for semantic search component

    Returns:
        List of (chunk, score) tuples sorted by combined score
    """
    # Semantic search component
    semantic_results = score_chunks(query, chunks, embeddings_model)
    semantic_scores = {chunk: score for chunk, score in semantic_results}

    # Keyword search component (simple implementation)
    query_terms = set(query.lower().split())
    keyword_scores = {}

    for chunk in chunks:
        chunk_lower = chunk.lower()
        term_matches = sum(1 for term in query_terms if term in chunk_lower)
        keyword_scores[chunk] = term_matches / max(1, len(query_terms))

    # Combine scores
    combined_results = []
    for chunk in chunks:
        combined_score = (
            keyword_weight * keyword_scores[chunk] +
            semantic_weight * semantic_scores.get(chunk, 0)
        )
        combined_results.append((chunk, combined_score))

    # Sort by combined score (highest first)
    return sorted(combined_results, key=lambda x: x[1], reverse=True)
