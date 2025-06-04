"""
Utility functions and helper classes for the cognitive system
"""
from .logging import setup_logging, get_cognitive_logger
from .helpers import (
    generate_memory_id,
    calculate_memory_strength,
    apply_forgetting_curve,
    calculate_attention_score,
    safe_json_serialize,
    validate_embedding_vector
)

__all__ = [
    "setup_logging",
    "get_cognitive_logger",
    "generate_memory_id",
    "calculate_memory_strength", 
    "apply_forgetting_curve",
    "calculate_attention_score",
    "safe_json_serialize",
    "validate_embedding_vector"
]