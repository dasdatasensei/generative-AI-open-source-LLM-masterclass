"""
Monitoring Utilities

Functions for monitoring LLM application performance and usage.
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)


class InferenceTracker:
    """
    Tracks inference statistics for monitoring performance.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the inference tracker.

        Args:
            log_file: Optional file to log inference data
        """
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "requests_by_model": {},
            "errors": 0,
            "requests_by_hour": {},
        }
        self.log_file = log_file

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

    def log_request(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an inference request.

        Args:
            model_name: Name of the model used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency: Request latency in seconds
            success: Whether the request succeeded
            metadata: Additional metadata
        """
        # Update general stats
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += prompt_tokens + completion_tokens
        self.stats["total_latency"] += latency

        if not success:
            self.stats["errors"] += 1

        # Update model-specific stats
        if model_name not in self.stats["requests_by_model"]:
            self.stats["requests_by_model"][model_name] = {
                "count": 0,
                "tokens": 0,
                "latency": 0,
                "errors": 0
            }

        model_stats = self.stats["requests_by_model"][model_name]
        model_stats["count"] += 1
        model_stats["tokens"] += prompt_tokens + completion_tokens
        model_stats["latency"] += latency

        if not success:
            model_stats["errors"] += 1

        # Update hourly stats
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if hour not in self.stats["requests_by_hour"]:
            self.stats["requests_by_hour"][hour] = 0
        self.stats["requests_by_hour"][hour] += 1

        # Log to file if specified
        if self.log_file:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency": latency,
                "success": success
            }

            if metadata:
                log_entry["metadata"] = metadata

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def get_stats(self):
        """
        Get current statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()

        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
            stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["total_requests"]
            stats["error_rate"] = stats["errors"] / stats["total_requests"]

        for model, model_stats in stats["requests_by_model"].items():
            if model_stats["count"] > 0:
                model_stats["avg_latency"] = model_stats["latency"] / model_stats["count"]
                model_stats["avg_tokens"] = model_stats["tokens"] / model_stats["count"]
                model_stats["error_rate"] = model_stats["errors"] / model_stats["count"]

        return stats

    def save_stats(self, file_path: str):
        """
        Save statistics to a JSON file.

        Args:
            file_path: Path to save the statistics
        """
        with open(file_path, "w") as f:
            json.dump(self.get_stats(), f, indent=2)


def measure_with_timeout(func: Callable, timeout: float = 30.0, *args, **kwargs):
    """
    Measure function execution time with timeout.

    Args:
        func: Function to execute
        timeout: Timeout in seconds
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (result, execution_time)
    """
    import threading
    import concurrent.futures

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            execution_time = time.time() - start_time
            return result, execution_time
        except concurrent.futures.TimeoutError:
            logger.warning(f"Function execution timed out after {timeout} seconds")
            return None, timeout


class FallbackChain:
    """
    Chain of models with automatic fallback on failure or timeout.
    """

    def __init__(self, models: List[Dict[str, Any]], timeout: float = 30.0):
        """
        Initialize the fallback chain.

        Args:
            models: List of model configurations, from highest to lowest priority
                Each model should have:
                - "name": Model name
                - "model": Model object
                - "tokenizer": Optional tokenizer
            timeout: Timeout in seconds for each model
        """
        self.models = models
        self.timeout = timeout
        self.tracker = InferenceTracker()

    def generate(self, prompt: str, **kwargs):
        """
        Generate a response, falling back to next model on failure.

        Args:
            prompt: Prompt text
            **kwargs: Additional arguments to pass to generation

        Returns:
            Generated response and metadata
        """
        result = None
        metadata = {
            "attempts": [],
            "success": False,
            "fallback_used": False
        }

        for i, model_config in enumerate(self.models):
            model_name = model_config["name"]
            model = model_config["model"]
            tokenizer = model_config.get("tokenizer")

            try:
                start_time = time.time()

                if tokenizer:
                    # Transformer model
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **kwargs)
                    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                    prompt_tokens = inputs.input_ids.shape[1]
                    completion_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                else:
                    # llama.cpp model
                    output = model(prompt=prompt, **kwargs)
                    result = output["choices"][0]["text"]

                    prompt_tokens = model.n_tokens(prompt)
                    completion_tokens = len(output["choices"][0]["tokens"])

                latency = time.time() - start_time
                success = True

                # Log the request
                self.tracker.log_request(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency=latency,
                    success=True
                )

                # Record attempt
                metadata["attempts"].append({
                    "model": model_name,
                    "success": True,
                    "latency": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                })

                metadata["success"] = True
                metadata["fallback_used"] = i > 0

                # Successfully generated, break the loop
                break

            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")

                # Log the failed request
                self.tracker.log_request(
                    model_name=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency=time.time() - start_time,
                    success=False,
                    metadata={"error": str(e)}
                )

                # Record attempt
                metadata["attempts"].append({
                    "model": model_name,
                    "success": False,
                    "error": str(e)
                })

        if not metadata["success"]:
            result = "I'm sorry, but all available models failed to generate a response. Please try again later."

        return result, metadata
