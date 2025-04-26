"""
Evaluation Utilities

Functions for evaluating LLM and RAG system performance.
"""

import os
import logging
import json
import csv
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import ragas
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available, some evaluation functionality will be limited")

try:
    from lm_eval import evaluator
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    logger.warning("LM Evaluation Harness not available, some evaluation functionality will be limited")

try:
    from trulens_eval import TruLlama, Feedback, Tru
    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    logger.warning("TruLens not available, some evaluation functionality will be limited")


def evaluate_rag_with_ragas(
    questions: List[str],
    retrieved_contexts: List[List[str]],
    generated_answers: List[str],
    ground_truths: Optional[List[str]] = None,
):
    """
    Evaluate a RAG system using RAGAS metrics.

    Args:
        questions: List of questions
        retrieved_contexts: List of lists of retrieved contexts for each question
        generated_answers: List of generated answers
        ground_truths: Optional list of ground truth answers

    Returns:
        Dictionary of evaluation results
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS is not installed. Please install it with 'pip install ragas'.")

    import pandas as pd
    from datasets import Dataset

    # Prepare evaluation data
    eval_data = {
        "question": questions,
        "contexts": retrieved_contexts,
        "answer": generated_answers,
    }

    if ground_truths:
        eval_data["ground_truths"] = [[gt] for gt in ground_truths]

    # Convert to RAGAS Dataset format
    dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

    # Run evaluation
    results = {}

    # Always evaluate these metrics (don't require ground truth)
    faithfulness_score = ragas.evaluate(dataset, [faithfulness])
    answer_relevancy_score = ragas.evaluate(dataset, [answer_relevancy])
    context_precision_score = ragas.evaluate(dataset, [context_precision])

    results["faithfulness"] = faithfulness_score["faithfulness"]
    results["answer_relevancy"] = answer_relevancy_score["answer_relevancy"]
    results["context_precision"] = context_precision_score["context_precision"]

    # Evaluate these metrics if ground truth is provided
    if ground_truths:
        context_recall_score = ragas.evaluate(dataset, [context_recall])
        answer_correctness_score = ragas.evaluate(dataset, [answer_correctness])

        results["context_recall"] = context_recall_score["context_recall"]
        results["answer_correctness"] = answer_correctness_score["answer_correctness"]

    # Calculate overall score
    if ground_truths:
        results["overall_score"] = (
            results["faithfulness"] +
            results["answer_relevancy"] +
            results["context_precision"] +
            results["context_recall"] +
            results["answer_correctness"]
        ) / 5
    else:
        results["overall_score"] = (
            results["faithfulness"] +
            results["answer_relevancy"] +
            results["context_precision"]
        ) / 3

    return results


def evaluate_llm_with_harness(
    model,
    tokenizer=None,
    model_type="llama",
    tasks=["gsm8k", "mmlu", "truthfulqa"],
    num_samples=100
):
    """
    Evaluate an LLM using the LM Evaluation Harness.

    Args:
        model: The model to evaluate
        tokenizer: Optional tokenizer for Hugging Face models
        model_type: Type of model (e.g., "llama", "phi", "mistral")
        tasks: List of tasks to evaluate on
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary of evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError("LM Evaluation Harness not installed. Please install it with 'pip install lm-eval'.")

    # Prepare model for evaluation (implementation depends on model type)
    # This is a simplified example
    if tokenizer:
        # For Hugging Face models
        from lm_eval.models import HuggingFaceModel

        eval_model = HuggingFaceModel(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=1
        )
    else:
        # For other model types, custom implementation would be needed
        # This is just a placeholder
        raise NotImplementedError("Custom model adapter not implemented")

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit=num_samples
    )

    return results


def evaluate_llm_with_trulens(
    llm,
    questions: List[str],
    retrieve_fn=None,
):
    """
    Evaluate an LLM or RAG system using TruLens.

    Args:
        llm: The LLM to evaluate
        questions: List of questions to evaluate on
        retrieve_fn: Optional retrieval function for RAG evaluation

    Returns:
        Dictionary of evaluation results
    """
    if not TRULENS_AVAILABLE:
        raise ImportError("TruLens not installed. Please install it with 'pip install trulens-eval'.")

    # Initialize TruLens
    tru = Tru()

    # Create feedback functions
    feedbacks = [
        Feedback("relevance").on_input_output(),
        Feedback("correctness").on_input_output(),
        Feedback("coherence").on_output(),
        Feedback("fluency").on_output()
    ]

    if retrieve_fn:
        # RAG-specific feedbacks
        feedbacks.extend([
            Feedback("groundedness").on_retrieval_generation(),
            Feedback("context_relevance").on_retrieval()
        ])

    # Create TruLlama wrapper
    tru_llm = TruLlama(
        llm,
        app_id="llm_evaluation",
        feedbacks=feedbacks
    )

    # Evaluate
    results = []
    for question in questions:
        with tru_llm as recording:
            if retrieve_fn:
                context = retrieve_fn(question)
                response = llm(context + "\n" + question)
            else:
                response = llm(question)

        results.append(recording)

    # Process results
    evaluation_results = tru.get_records_and_feedback(app_ids=["llm_evaluation"])

    return evaluation_results


def detect_hallucinations(
    answers: List[str],
    contexts: List[List[str]],
    model=None
):
    """
    Detect hallucinations in generated answers.

    Args:
        answers: List of generated answers
        contexts: List of lists of contexts used for generation
        model: Optional model for advanced hallucination detection

    Returns:
        List of hallucination scores and explanations
    """
    # Simple rule-based hallucination detection
    # A more sophisticated approach would use a model specifically trained for this
    results = []

    for answer, context_list in zip(answers, contexts):
        # Combine all contexts
        full_context = " ".join(context_list)

        # Simple token overlap-based detection
        answer_tokens = set(answer.lower().split())
        context_tokens = set(full_context.lower().split())

        # Calculate token overlap
        overlap_tokens = answer_tokens.intersection(context_tokens)
        hallucination_score = 1.0 - (len(overlap_tokens) / len(answer_tokens)) if len(answer_tokens) > 0 else 0.0

        # Determine potential hallucinated content
        potential_hallucinations = []
        for sentence in answer.split('.'):
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                sentence_tokens = set(sentence.lower().split())
                sentence_overlap = sentence_tokens.intersection(context_tokens)
                sentence_score = 1.0 - (len(sentence_overlap) / len(sentence_tokens)) if len(sentence_tokens) > 0 else 0.0

                if sentence_score > 0.7:  # High hallucination threshold
                    potential_hallucinations.append({
                        "sentence": sentence,
                        "score": sentence_score
                    })

        results.append({
            "overall_hallucination_score": hallucination_score,
            "potential_hallucinations": potential_hallucinations
        })

    return results


def create_evaluation_report(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, float],
    examples: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None
):
    """
    Create a detailed evaluation report.

    Args:
        model_name: Name of the evaluated model
        dataset_name: Name of the evaluation dataset
        metrics: Dictionary of evaluation metrics
        examples: Optional list of example evaluations
        output_file: Optional file to save the report

    Returns:
        Report as a dictionary
    """
    report = {
        "model": model_name,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "examples": examples or []
    }

    # Calculate summary statistics
    report["summary"] = {
        "average_score": sum(metrics.values()) / len(metrics) if metrics else 0,
        "num_examples": len(examples) if examples else 0
    }

    # Save report if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_file}")

    return report


def export_evaluation_results(
    results: Dict[str, Any],
    format: str = "json",
    output_file: Optional[str] = None
):
    """
    Export evaluation results in various formats.

    Args:
        results: Evaluation results
        format: Output format (json, csv, or markdown)
        output_file: Optional output file

    Returns:
        Exported results as a string
    """
    if format == "json":
        output = json.dumps(results, indent=2)

    elif format == "csv":
        # Flatten metrics for CSV export
        rows = []
        metrics = results.get("metrics", {})

        # Header row
        header = ["model", "dataset", "timestamp"] + list(metrics.keys())
        rows.append(header)

        # Data row
        row = [
            results.get("model", ""),
            results.get("dataset", ""),
            results.get("timestamp", "")
        ] + [metrics.get(k, "") for k in header[3:]]

        rows.append(row)

        # Write to string
        import io
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        writer.writerows(rows)
        output = output_buffer.getvalue()

    elif format == "markdown":
        # Create markdown table
        output = f"# Evaluation Report: {results.get('model', 'Unknown Model')}

"
        output += f"**Dataset**: {results.get('dataset', 'Unknown')}
"
        output += f"**Date**: {results.get('timestamp', '')}

"

        output += "## Metrics

"
        output += "| Metric | Value |
"
        output += "|--------|-------|
"

        for metric, value in results.get("metrics", {}).items():
            output += f"| {metric} | {value:.4f} |
"

        output += f"
**Overall Score**: {results.get('summary', {}).get('average_score', 0):.4f}
"

        if results.get("examples"):
            output += "
## Example Evaluations

"
            for i, example in enumerate(results["examples"][:5]):  # Show up to 5 examples
                output += f"### Example {i+1}

"
                output += f"**Question**: {example.get('question', '')}

"
                output += f"**Answer**: {example.get('answer', '')}

"
                output += f"**Score**: {example.get('score', 0):.4f}

"
                if example.get("explanation"):
                    output += f"**Explanation**: {example['explanation']}

"

    else:
        raise ValueError(f"Unsupported format: {format}")

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)

        logger.info(f"Evaluation results exported to {output_file}")

    return output
