"""
Core utilities and helper functions for the cognitive system
"""
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np

def generate_memory_id(content: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique memory ID based on content and timestamp
    
    Args:
        content: The memory content
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Unique memory identifier
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create hash from content and timestamp
    hash_input = f"{content}{timestamp.isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

def calculate_memory_strength(
    importance: float,
    recency: float,
    frequency: int = 1,
    emotional_valence: float = 0.0
) -> float:
    """
    Calculate memory strength using biologically-inspired factors
    
    Args:
        importance: Importance score (0.0 to 1.0)
        recency: Recency score (0.0 to 1.0, 1.0 = very recent)
        frequency: Number of times accessed/reinforced
        emotional_valence: Emotional weight (-1.0 to 1.0)
    
    Returns:
        Combined memory strength score
    """
    # Base strength from importance and recency
    base_strength = (importance * 0.6) + (recency * 0.3)
    
    # Frequency bonus (logarithmic to avoid unlimited growth)
    frequency_bonus = np.log(1 + frequency) * 0.1
    
    # Emotional enhancement (absolute value, strong emotions strengthen memory)
    emotional_bonus = abs(emotional_valence) * 0.1
    
    return min(1.0, base_strength + frequency_bonus + emotional_bonus)

def apply_forgetting_curve(
    initial_strength: float,
    time_delta: timedelta,
    decay_rate: float = 0.1
) -> float:
    """
    Apply Ebbinghaus forgetting curve to memory strength
    
    Args:
        initial_strength: Initial memory strength
        time_delta: Time elapsed since memory formation
        decay_rate: Rate of forgetting (higher = faster decay)
    
    Returns:
        Decayed memory strength
    """
    hours_elapsed = time_delta.total_seconds() / 3600
    decay_factor = np.exp(-decay_rate * hours_elapsed)
    return initial_strength * decay_factor

def calculate_attention_score(
    relevance: float,
    novelty: float,
    emotional_salience: float = 0.0,
    current_fatigue: float = 0.0
) -> float:
    """
    Calculate attention score for cognitive processing
    
    Args:
        relevance: How relevant the stimulus is to current goals
        novelty: How novel/unexpected the stimulus is
        emotional_salience: Emotional weight of the stimulus
        current_fatigue: Current cognitive fatigue level (0.0 to 1.0)
    
    Returns:
        Attention score (0.0 to 1.0)
    """
    # Base attention from relevance and novelty
    base_attention = (relevance * 0.7) + (novelty * 0.3)
    
    # Emotional enhancement
    emotional_boost = abs(emotional_salience) * 0.2
    
    # Fatigue penalty
    fatigue_penalty = current_fatigue * 0.3
    
    return max(0.0, min(1.0, base_attention + emotional_boost - fatigue_penalty))

def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize objects to JSON, handling datetime and other types
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string representation
    """
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
        return str(obj)
    
    return json.dumps(obj, default=json_serializer, indent=2)

def validate_embedding_vector(vector: Union[List[float], np.ndarray], expected_dim: int) -> bool:
    """
    Validate that an embedding vector has the correct dimensions and format
    
    Args:
        vector: The embedding vector to validate
        expected_dim: Expected dimensionality
    
    Returns:
        True if valid, False otherwise
    """
    if isinstance(vector, list):
        vector = np.array(vector)
    
    return (
        isinstance(vector, np.ndarray) and
        vector.shape == (expected_dim,) and
        np.isfinite(vector).all() and
        not np.allclose(vector, 0)  # Not all zeros
    )
