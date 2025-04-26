"""
LLM Utilities

Common functions for working with LLMs across the course.
"""

import os
import time
import logging
import torch
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

def load_model_with_timing(model_load_fn, *args, **kwargs):
    """Load a model and log timing information."""
    start_time = time.time()
    try:
        model = model_load_fn(*args, **kwargs)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        return model, load_time
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def format_prompt(system_prompt: str, user_message: str, model_type: str = "phi") -> str:
    """Format a prompt according to the model's expected format."""
    if model_type.lower() in ["phi", "phi-3"]:
        return f"<|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
"
    elif model_type.lower() in ["llama", "llama3"]:
        return f"<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"
    elif model_type.lower() in ["mistral", "mixtral"]:
        return f"<s>[INST] {system_prompt}

{user_message} [/INST]"
    else:
        return f"{system_prompt}

{user_message}"

def log_gpu_info():
    """Log information about available GPUs."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        logger.warning("CUDA not available, using CPU")
        return False

def measure_inference_speed(model, input_text, tokenizer=None, generate_kwargs=None, num_runs=3):
    """Measure inference speed of a model."""
    generate_kwargs = generate_kwargs or {"max_new_tokens": 50}

    if tokenizer:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    else:
        inputs = input_text

    # Warmup
    if tokenizer:
        with torch.no_grad():
            _ = model.generate(**inputs, **generate_kwargs)
    else:
        _ = model(inputs, **generate_kwargs)

    # Measure
    times = []
    for _ in range(num_runs):
        start_time = time.time()

        if tokenizer:
            with torch.no_grad():
                outputs = model.generate(**inputs, **generate_kwargs)
            output_length = outputs.shape[1] - inputs.input_ids.shape[1]
        else:
            output = model(inputs, **generate_kwargs)
            output_length = len(output["choices"][0]["text"].split())

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    tokens_per_second = output_length / avg_time

    return {
        "avg_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "output_length": output_length
    }
