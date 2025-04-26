"""
Vector Database Utilities

Functions for working with different vector database implementations.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import time

logger = logging.getLogger(__name__)

# FAISS utilities
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, some functionality will be limited")

# Chroma utilities
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available, some functionality will be limited")

# Weaviate utilities
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate not available, some functionality will be limited")


def create_faiss_index(embeddings: np.ndarray, index_type: str = "Flat") -> Any:
    """
    Create a FAISS index with the specified type.

    Args:
        embeddings: Numpy array of embeddings to index
        index_type: Type of FAISS index to create (Flat, IVF, HNSW, etc.)

    Returns:
        FAISS index
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'.")

    dimension = embeddings.shape[1]

    if index_type == "Flat":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IVF":
        # IVF requires training with a subset of vectors
        nlist = min(int(np.sqrt(embeddings.shape[0])), 100)  # Rule of thumb
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is a common M value
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    if not index.is_trained:
        index.train(embeddings)

    index.add(embeddings.astype(np.float32))
    return index


def create_chroma_collection(
    collection_name: str,
    persist_directory: Optional[str] = None,
    embedding_function = None,
) -> Any:
    """
    Create a ChromaDB collection.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist the collection
        embedding_function: Function to create embeddings

    Returns:
        ChromaDB collection
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("ChromaDB is not installed. Please install it with 'pip install chromadb'.")

    client_settings = chromadb.config.Settings()

    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory, settings=client_settings)
    else:
        client = chromadb.Client(client_settings)

    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Loaded existing collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
        logger.info(f"Created new collection: {collection_name}")

    return collection


def add_documents_to_chroma(
    collection,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
):
    """
    Add documents to a ChromaDB collection.

    Args:
        collection: ChromaDB collection
        documents: List of document texts
        metadatas: List of metadata dictionaries
        ids: List of document IDs
    """
    if not ids:
        ids = [f"doc_{i}" for i in range(len(documents))]

    if not metadatas:
        metadatas = [{} for _ in range(len(documents))]

    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        logger.info(f"Added batch {i//batch_size + 1}: documents {i} to {end_idx}")


def query_chroma(
    collection,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query a ChromaDB collection.

    Args:
        collection: ChromaDB collection
        query_text: Query text
        n_results: Number of results to return
        where: Filter on metadata
        where_document: Filter on document content

    Returns:
        Query results
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        where_document=where_document
    )

    return results


def create_weaviate_client(
    url: str = "http://localhost:8080",
    auth_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a Weaviate client.

    Args:
        url: Weaviate URL
        auth_config: Authentication configuration

    Returns:
        Weaviate client
    """
    if not WEAVIATE_AVAILABLE:
        raise ImportError("Weaviate not installed. Please install it with 'pip install weaviate-client'.")

    return weaviate.Client(url=url, auth_client_secret=auth_config)


def create_weaviate_schema(
    client,
    class_name: str,
    properties: List[Dict[str, Any]],
    vector_index_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a Weaviate schema.

    Args:
        client: Weaviate client
        class_name: Name of the class
        properties: List of property configurations
        vector_index_config: Vector index configuration
    """
    if not WEAVIATE_AVAILABLE:
        raise ImportError("Weaviate not installed. Please install it with 'pip install weaviate-client'.")

    # Check if class exists
    schema = client.schema.get()
    classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []

    if class_name in classes:
        logger.info(f"Class {class_name} already exists")
        return

    # Default vector index config
    if vector_index_config is None:
        vector_index_config = {
            "vectorizer": "none",  # We'll provide our own vectors
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "skip": False,
                "ef": 100,
                "efConstruction": 128,
                "maxConnections": 64
            }
        }

    # Create class
    class_obj = {
        "class": class_name,
        "vectorIndexConfig": vector_index_config,
        "properties": properties
    }

    client.schema.create_class(class_obj)
    logger.info(f"Created Weaviate class: {class_name}")


def add_documents_to_weaviate(
    client,
    class_name: str,
    documents: List[str],
    embeddings: np.ndarray,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 100
) -> None:
    """
    Add documents with embeddings to Weaviate.

    Args:
        client: Weaviate client
        class_name: Name of the class
        documents: List of document texts
        embeddings: Numpy array of embeddings
        metadatas: List of metadata dictionaries
        batch_size: Batch size for import
    """
    if not WEAVIATE_AVAILABLE:
        raise ImportError("Weaviate not installed. Please install it with 'pip install weaviate-client'.")

    if not metadatas:
        metadatas = [{} for _ in range(len(documents))]

    # Import in batches
    with client.batch as batch:
        batch.batch_size = batch_size

        for i in range(len(documents)):
            # Prepare data object
            data_object = {
                "content": documents[i],
                **metadatas[i]
            }

            # Add object with vector
            batch.add_data_object(
                data_object=data_object,
                class_name=class_name,
                vector=embeddings[i].tolist()
            )

            if i % batch_size == 0 and i > 0:
                logger.info(f"Added {i} documents to Weaviate")

    logger.info(f"Added {len(documents)} documents to Weaviate class {class_name}")


def query_weaviate(
    client,
    class_name: str,
    query_embedding: np.ndarray,
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_vector: bool = False
) -> List[Dict[str, Any]]:
    """
    Query Weaviate using vector search.

    Args:
        client: Weaviate client
        class_name: Name of the class
        query_embedding: Query embedding vector
        limit: Number of results to return
        filters: Filters to apply to the query
        include_vector: Whether to include vectors in results

    Returns:
        List of query results
    """
    if not WEAVIATE_AVAILABLE:
        raise ImportError("Weaviate not installed. Please install it with 'pip install weaviate-client'.")

    # Build query
    query = client.query.get(class_name, ["content"])

    # Add property fields based on schema (simplified)
    query = query.with_additional(["id", "certainty"])

    if include_vector:
        query = query.with_additional(["vector"])

    # Add vector search
    query = query.with_near_vector({
        "vector": query_embedding.tolist()
    })

    # Add filters if provided
    if filters:
        where_filter = filters  # In real implementation, convert filters to Weaviate filter format
        query = query.with_where(where_filter)

    # Set limit
    query = query.with_limit(limit)

    # Execute query
    result = query.do()

    # Extract results
    if "data" in result and "Get" in result["data"] and class_name in result["data"]["Get"]:
        return result["data"]["Get"][class_name]

    return []


def compare_vector_stores(
    query: str,
    documents: List[str],
    embedding_model,
) -> Dict[str, Any]:
    """
    Compare performance of different vector stores on the same data.

    Args:
        query: Query text
        documents: List of documents
        embedding_model: Model to create embeddings

    Returns:
        Dictionary with performance metrics
    """
    results = {}

    # Create embeddings
    doc_embeddings = embedding_model.encode(documents)
    query_embedding = embedding_model.encode(query)

    # Test FAISS
    if FAISS_AVAILABLE:
        try:
            start_time = time.time()

            index = create_faiss_index(doc_embeddings)
            D, I = index.search(np.array([query_embedding]).astype(np.float32), 5)

            results["faiss"] = {
                "time": time.time() - start_time,
                "results": [documents[i] for i in I[0]],
                "scores": D[0].tolist()
            }
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            results["faiss"] = {"error": str(e)}

    # Test ChromaDB
    if CHROMA_AVAILABLE:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                start_time = time.time()

                # Create a simple embedding function
                class EmbeddingFunction:
                    def __call__(self, texts):
                        return embedding_model.encode(texts).tolist()

                collection = create_chroma_collection(
                    "test_collection",
                    persist_directory=tmp_dir,
                    embedding_function=EmbeddingFunction()
                )

                add_documents_to_chroma(collection, documents)
                query_results = query_chroma(collection, query, 5)

                results["chroma"] = {
                    "time": time.time() - start_time,
                    "results": query_results["documents"][0],
                    "scores": [float(d) for d in query_results["distances"][0]]
                }
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            results["chroma"] = {"error": str(e)}

    # Test Weaviate
    if WEAVIATE_AVAILABLE:
        try:
            # This is a simplified test - in practice, you'd need a running Weaviate instance
            logger.info("Weaviate tests require a running instance - skipping")
            results["weaviate"] = {"status": "skipped - requires running instance"}
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            results["weaviate"] = {"error": str(e)}

    return results


def benchmark_embedding_models(
    query: str,
    documents: List[str],
    embedding_models: Dict[str, Any],
    vector_db: str = "faiss"
) -> Dict[str, Any]:
    """
    Benchmark different embedding models on the same data.

    Args:
        query: Query text
        documents: List of documents
        embedding_models: Dictionary of embedding models
        vector_db: Vector database to use (faiss or chroma)

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    for model_name, model in embedding_models.items():
        try:
            start_time = time.time()

            # Create embeddings
            doc_embeddings = model.encode(documents)
            query_embedding = model.encode(query)

            encoding_time = time.time() - start_time

            # Search using specified vector DB
            if vector_db == "faiss" and FAISS_AVAILABLE:
                index = create_faiss_index(doc_embeddings)
                search_start = time.time()
                D, I = index.search(np.array([query_embedding]).astype(np.float32), 5)
                search_time = time.time() - search_start

                results[model_name] = {
                    "encoding_time": encoding_time,
                    "search_time": search_time,
                    "total_time": encoding_time + search_time,
                    "results": [documents[i] for i in I[0]],
                    "scores": D[0].tolist(),
                    "embedding_dimension": len(query_embedding)
                }

            elif vector_db == "chroma" and CHROMA_AVAILABLE:
                # Implementation for ChromaDB benchmarking
                # Similar to the compare_vector_stores function
                pass

            else:
                raise ValueError(f"Unsupported vector database: {vector_db}")

        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results
