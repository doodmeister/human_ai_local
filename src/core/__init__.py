"""
Core cognitive architecture components
"""
from .config import CognitiveConfig, MemoryConfig, AttentionConfig, ProcessingConfig, AWSConfig, LLMConfig
from .cognitive_agent import CognitiveAgent

__all__ = [
    "CognitiveConfig",
    "MemoryConfig", 
    "AttentionConfig",
    "ProcessingConfig",
    "AWSConfig",
    "LLMConfig",
    "CognitiveAgent"
]