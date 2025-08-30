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

# Core cognitive components (lazy import pattern to avoid heavy deps during lightweight tests)
try:  # pragma: no cover - defensive import guard
    from .core import CognitiveAgent, CognitiveConfig  # type: ignore
except Exception:  # Avoid failing when optional deps (e.g., openai) missing for simple submodule imports
    CognitiveAgent = None  # type: ignore
    CognitiveConfig = None  # type: ignore
try:  # pragma: no cover
    from .utils import get_cognitive_logger, setup_logging  # type: ignore
except Exception:
    def get_cognitive_logger(*args, **kwargs):  # type: ignore
        return None
    def setup_logging(*args, **kwargs):  # type: ignore
        return None

__all__ = [
    "CognitiveAgent",
    "CognitiveConfig",
    "get_cognitive_logger",
    "setup_logging",
]