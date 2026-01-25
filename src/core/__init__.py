"""
Core cognitive architecture components
"""
from __future__ import annotations

from .config import (
    AWSConfig,
    AttentionConfig,
    CognitiveConfig,
    LLMConfig,
    MemoryConfig,
    ProcessingConfig,
)

__all__ = [
    "CognitiveConfig",
    "MemoryConfig", 
    "AttentionConfig",
    "ProcessingConfig",
    "AWSConfig",
    "LLMConfig",
    "CognitiveAgent",
]


def __getattr__(name: str):  # pragma: no cover
    if name == "CognitiveAgent":
        import importlib

        return importlib.import_module("src.orchestration.cognitive_agent").CognitiveAgent
    raise AttributeError(name)