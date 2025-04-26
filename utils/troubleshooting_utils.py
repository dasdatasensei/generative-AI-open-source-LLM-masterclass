"""
Troubleshooting Utilities

Functions for diagnosing and resolving common issues with LLMs and RAG systems.
"""

import os
import sys
import logging
import json
import psutil
import platform
from typing import List, Dict, Any, Optional, Union, Tuple
import time

logger = logging.getLogger(__name__)

def check_system_resources():
    """
    Check available system resources.

    Returns:
        Dictionary of system resource information
    """
    # CPU info
    cpu_info = {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=1),
    }

    # Memory info
    memory = psutil.virtual_memory()
    memory_info = {
        "total_memory_gb": memory.total / (1024 ** 3),
        "available_memory_gb": memory.available / (1024 ** 3),
        "memory_percent_used": memory.percent,
    }

    # Disk info
    disk = psutil.disk_usage('/')
    disk_info = {
        "total_disk_gb": disk.total / (1024 ** 3),
        "free_disk_gb": disk.free / (1024 ** 3),
        "disk_percent_used": disk.percent,
    }

    # GPU info (if available)
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            }

            # Get memory info for each GPU
            devices_info = []
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory / (1024 ** 3)

                # Get current memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)

                devices_info.append({
                    "index": i,
                    "name": device_props.name,
                    "total_memory_gb": total_memory,
                    "allocated_memory_gb": allocated,
                    "reserved_memory_gb": reserved,
                    "free_memory_gb": total_memory - allocated,
                })

            gpu_info["devices"] = devices_info
        else:
            gpu_info = {"cuda_available": False}
    except ImportError:
        gpu_info = {"cuda_available": False, "error": "torch module not available"}
    except Exception as e:
        gpu_info = {"cuda_available": False, "error": str(e)}

    # Environment info
    env_info = {
        "os": platform.system(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Put everything together
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "disk": disk_info,
        "gpu": gpu_info,
        "environment": env_info,
        "timestamp": time.time(),
    }


def diagnose_context_window_issue(
    model_name: str,
    input_text: str,
    max_context_length: int
):
    """
    Diagnose potential context window issues.

    Args:
        model_name: Name of the model
        input_text: Input text to check
        max_context_length: Maximum context length for the model

    Returns:
        Dictionary with diagnosis results
    """
    try:
        # Try to estimate token count
        # This is a rough approximation; actual tokenization depends on the specific model
        words = len(input_text.split())
        chars = len(input_text)

        # Rough token estimation: ~1.3 tokens per word for English
        estimated_tokens = int(words * 1.3)

        # For more accurate counting, we'd need the actual tokenizer for the model
        # For example, with a Hugging Face tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            actual_tokens = len(tokenizer.encode(input_text))
        except:
            actual_tokens = None

        # Results
        result = {
            "input_length_chars": chars,
            "input_length_words": words,
            "estimated_tokens": estimated_tokens,
            "actual_tokens": actual_tokens,
            "max_context_length": max_context_length,
        }

        # Determine if there's a potential issue
        if actual_tokens:
            if actual_tokens > max_context_length:
                result["diagnosis"] = "EXCEEDS_CONTEXT_WINDOW"
                result["recommended_action"] = "Reduce input length or chunk text"
                result["severity"] = "HIGH"
            elif actual_tokens > 0.9 * max_context_length:
                result["diagnosis"] = "NEAR_CONTEXT_LIMIT"
                result["recommended_action"] = "Consider reducing input length"
                result["severity"] = "MEDIUM"
            else:
                result["diagnosis"] = "WITHIN_CONTEXT_LIMITS"
                result["severity"] = "LOW"
                result["utilization"] = f"{(actual_tokens / max_context_length) * 100:.1f}%"
        else:
            if estimated_tokens > max_context_length:
                result["diagnosis"] = "LIKELY_EXCEEDS_CONTEXT_WINDOW"
                result["recommended_action"] = "Reduce input length or chunk text"
                result["severity"] = "HIGH"
            elif estimated_tokens > 0.9 * max_context_length:
                result["diagnosis"] = "LIKELY_NEAR_CONTEXT_LIMIT"
                result["recommended_action"] = "Consider reducing input length"
                result["severity"] = "MEDIUM"
            else:
                result["diagnosis"] = "LIKELY_WITHIN_CONTEXT_LIMITS"
                result["severity"] = "LOW"
                result["utilization"] = f"{(estimated_tokens / max_context_length) * 100:.1f}%"

        return result

    except Exception as e:
        return {
            "error": str(e),
            "diagnosis": "ERROR",
            "severity": "HIGH",
            "recommended_action": "Check model name and input text"
        }


def diagnose_memory_issues(model_config: Dict[str, Any]):
    """
    Diagnose potential memory issues for a given model configuration.

    Args:
        model_config: Dictionary with model configuration

    Returns:
        Dictionary with diagnosis results
    """
    # Get system resources
    resources = check_system_resources()

    # Extract model parameters
    model_name = model_config.get("name", "Unknown")
    model_size_b = model_config.get("parameters", 0)
    if isinstance(model_size_b, str):
        # Parse strings like "7B" to numeric
        try:
            model_size_b = float(model_size_b.replace("B", "")) * 1e9
        except:
            model_size_b = 0

    quantization = model_config.get("quantization", "none")

    # Estimate memory requirements
    # This is a rough approximation; actual requirements depend on many factors
    memory_multiplier = {
        "none": 2.0,  # FP32 (no quantization)
        "fp16": 1.0,  # FP16
        "8bit": 0.5,  # INT8
        "4bit": 0.25  # INT4
    }.get(quantization, 1.0)

    # Base memory requirement for model weights
    estimated_model_memory = (model_size_b * memory_multiplier) / (1024 ** 3)  # Convert to GB

    # Additional memory for activations, KV cache, etc. (rough estimate)
    activation_memory = estimated_model_memory * 0.2  # 20% of model size

    # Total estimated memory
    total_estimated_memory = estimated_model_memory + activation_memory

    # Available memory
    if resources["gpu"]["cuda_available"]:
        available_memory = resources["gpu"]["devices"][0]["free_memory_gb"]
        memory_type = "GPU"
    else:
        available_memory = resources["memory"]["available_memory_gb"]
        memory_type = "CPU"

    # Diagnosis
    result = {
        "model_name": model_name,
        "model_size": f"{model_size_b / 1e9:.1f}B parameters",
        "quantization": quantization,
        "estimated_model_memory_gb": estimated_model_memory,
        "estimated_total_memory_gb": total_estimated_memory,
        "available_memory_gb": available_memory,
        "memory_type": memory_type
    }

    # Determine if there's a potential issue
    if total_estimated_memory > available_memory:
        result["diagnosis"] = f"INSUFFICIENT_{memory_type}_MEMORY"
        result["severity"] = "HIGH"

        # Recommendations
        recommendations = []
        if memory_type == "GPU" and resources["memory"]["available_memory_gb"] > total_estimated_memory:
            recommendations.append("Use CPU instead of GPU")

        if quantization != "4bit":
            recommendations.append("Use 4-bit quantization")

        if not recommendations:
            recommendations.append("Use a smaller model")

        result["recommended_actions"] = recommendations

    elif total_estimated_memory > 0.8 * available_memory:
        result["diagnosis"] = f"LIMITED_{memory_type}_MEMORY"
        result["severity"] = "MEDIUM"
        result["recommended_actions"] = ["Consider using more aggressive quantization",
                                        "Close other applications to free memory"]
    else:
        result["diagnosis"] = "SUFFICIENT_MEMORY"
        result["severity"] = "LOW"
        result["memory_utilization"] = f"{(total_estimated_memory / available_memory) * 100:.1f}%"

    return result


def diagnose_rag_retrieval_issues(
    query: str,
    retrieved_documents: List[str],
    similarity_scores: List[float],
    threshold: float = 0.5
):
    """
    Diagnose issues with RAG retrieval.

    Args:
        query: The query text
        retrieved_documents: List of retrieved documents
        similarity_scores: List of similarity scores for the documents
        threshold: Similarity threshold for relevance

    Returns:
        Dictionary with diagnosis results
    """
    # Basic checks
    if not retrieved_documents:
        return {
            "diagnosis": "NO_DOCUMENTS_RETRIEVED",
            "severity": "HIGH",
            "recommended_actions": ["Check vector store configuration",
                                   "Verify document ingestion process"]
        }

    # Check similarity scores
    low_relevance_count = sum(1 for score in similarity_scores if score < threshold)

    # Calculate query complexity
    query_words = len(query.split())
    query_complexity = "HIGH" if query_words > 15 else "MEDIUM" if query_words > 8 else "LOW"

    # Results
    result = {
        "query_length": len(query),
        "query_word_count": query_words,
        "query_complexity": query_complexity,
        "documents_retrieved": len(retrieved_documents),
        "avg_similarity_score": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        "min_similarity_score": min(similarity_scores) if similarity_scores else 0,
        "max_similarity_score": max(similarity_scores) if similarity_scores else 0,
        "low_relevance_count": low_relevance_count
    }

    # Diagnosis
    if low_relevance_count == len(retrieved_documents) and len(retrieved_documents) > 0:
        result["diagnosis"] = "ALL_DOCUMENTS_LOW_RELEVANCE"
        result["severity"] = "HIGH"
        result["recommended_actions"] = [
            "Reformulate the query",
            "Check if relevant documents exist in the collection",
            "Consider using hybrid search (dense + sparse)",
            "Use query expansion techniques"
        ]
    elif low_relevance_count > 0:
        result["diagnosis"] = "SOME_DOCUMENTS_LOW_RELEVANCE"
        result["severity"] = "MEDIUM"
        result["recommended_actions"] = [
            "Consider adjusting the number of retrieved documents",
            "Implement re-ranking to prioritize relevant documents"
        ]
    elif len(retrieved_documents) == 1 and similarity_scores[0] > 0.9:
        result["diagnosis"] = "SINGLE_HIGH_RELEVANCE_DOCUMENT"
        result["severity"] = "LOW"
        # This might be good or bad depending on the use case
        result["note"] = "Single highly relevant document found. This may be appropriate for specific queries."
    else:
        result["diagnosis"] = "NORMAL_RETRIEVAL_PATTERN"
        result["severity"] = "LOW"

    return result


def create_troubleshooting_guide(issues: List[Dict[str, Any]], format: str = "markdown"):
    """
    Create a troubleshooting guide from a list of known issues.

    Args:
        issues: List of issue dictionaries
        format: Output format (markdown or json)

    Returns:
        Troubleshooting guide as a string
    """
    if format == "json":
        return json.dumps(issues, indent=2)

    # Create markdown guide
    guide = "# LLM and RAG Troubleshooting Guide\n\n"
    guide += "This guide addresses common issues with Large Language Models and RAG systems.\n\n"

    # Group issues by category
    categories = {}
    for issue in issues:
        category = issue.get("category", "General")
        if category not in categories:
            categories[category] = []
        categories[category].append(issue)

    # Generate guide by category
    for category, category_issues in categories.items():
        guide += f"## {category}\n\n"

        for i, issue in enumerate(category_issues):
            guide += f"### {i+1}. {issue.get('title', 'Issue')}\n\n"

            if "symptoms" in issue:
                guide += "**Symptoms:**\n\n"
                for symptom in issue["symptoms"]:
                    guide += f"- {symptom}\n"
                guide += "\n"

            if "causes" in issue:
                guide += "**Possible Causes:**\n\n"
                for cause in issue["causes"]:
                    guide += f"- {cause}\n"
                guide += "\n"

            if "solutions" in issue:
                guide += "**Solutions:**\n\n"
                for j, solution in enumerate(issue["solutions"]):
                    guide += f"{j+1}. {solution}\n"
                guide += "\n"

            if "code_example" in issue:
                guide += "**Example Code:**\n\n"
                guide += "```python\n"
                guide += issue["code_example"]
                guide += "\n```\n\n"

            if "notes" in issue:
                guide += "**Notes:**\n\n"
                guide += f"{issue['notes']}\n\n"

    return guide


def create_diagnostic_report(model_name: str, diagnostic_tests: List[Dict[str, Any]]):
    """
    Create a comprehensive diagnostic report.

    Args:
        model_name: Name of the model
        diagnostic_tests: List of diagnostic test results

    Returns:
        Diagnostic report as a dictionary
    """
    # System resources
    resources = check_system_resources()

    # Summarize test results
    test_summaries = []
    for test in diagnostic_tests:
        summary = {
            "test_name": test.get("test_name", "Unknown test"),
            "diagnosis": test.get("diagnosis", "UNKNOWN"),
            "severity": test.get("severity", "LOW"),
            "timestamp": test.get("timestamp", time.time())
        }
        test_summaries.append(summary)

    # Count issues by severity
    severity_counts = {
        "HIGH": sum(1 for test in test_summaries if test["severity"] == "HIGH"),
        "MEDIUM": sum(1 for test in test_summaries if test["severity"] == "MEDIUM"),
        "LOW": sum(1 for test in test_summaries if test["severity"] == "LOW")
    }

    # Overall health score (simple algorithm)
    health_score = 100
    health_score -= severity_counts["HIGH"] * 20
    health_score -= severity_counts["MEDIUM"] * 5
    health_score = max(0, health_score)  # Don't go below 0

    # Health status
    if health_score >= 90:
        health_status = "EXCELLENT"
    elif health_score >= 70:
        health_status = "GOOD"
    elif health_score >= 50:
        health_status = "FAIR"
    else:
        health_status = "POOR"

    # Assemble the report
    report = {
        "model_name": model_name,
        "timestamp": time.time(),
        "system_resources": resources,
        "diagnostic_summary": {
            "total_tests": len(diagnostic_tests),
            "health_score": health_score,
            "health_status": health_status,
            "severity_counts": severity_counts
        },
        "test_summaries": test_summaries,
        "detailed_results": diagnostic_tests
    }

    return report
