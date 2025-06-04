"""
Human-AI Cognition Framework

A biologically-inspired cognitive architecture that simulates human-like memory, 
attention, and reasoning capabilities in AI systems.

This package implements:
- Memory Systems (STM, LTM, Prospective, Procedural)
- Cognitive Processing Layers (Sensory Buffer, Attention, Meta-Cognition)
- Executive Functions (Cognitive Agent Orchestrator, Executive Planner)
- Neural Architectures (Hopfield Networks, DPAD, LSHN)
"""

__version__ = "0.1.0"
__author__ = "Human-AI Cognition Team"

# Core cognitive components
from .core import CognitiveAgent, CognitiveConfig
from .utils import get_cognitive_logger, setup_logging

__all__ = [
    "CognitiveAgent",
    "CognitiveConfig",
    "get_cognitive_logger", 
    "setup_logging"
]