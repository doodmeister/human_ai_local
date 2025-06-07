"""
Episodic Memory System Module

This module implements episodic memory for the Human-AI Cognition Framework.
Episodic memory stores contextually rich memories of specific experiences
with temporal organization and cross-references to STM/LTM systems.

Key Components:
- EpisodicMemorySystem: Main episodic memory management system
- EpisodicMemory: Individual episodic memory data structure
- EpisodicContext: Rich contextual metadata for episodes
- EpisodicSearchResult: Search result with relevance scoring
"""

from .episodic_memory import (
    EpisodicMemory,
    EpisodicContext,
    EpisodicSearchResult,
    EpisodicMemorySystem
)

__all__ = [
    'EpisodicMemory',
    'EpisodicContext', 
    'EpisodicSearchResult',
    'EpisodicMemorySystem'
]

__version__ = '1.0.0'
