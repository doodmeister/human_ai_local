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

# Core cognitive components — fully lazy to avoid pulling in torch/transformers
# (which add 10-30s of startup time).  Access via src.CognitiveAgent etc.
CognitiveAgent = None  # type: ignore
CognitiveConfig = None  # type: ignore

try:  # pragma: no cover
    from .core.config import CognitiveConfig  # type: ignore  (lightweight, no heavy deps)
except Exception:
    pass

try:  # pragma: no cover
    from .utils import get_cognitive_logger, setup_logging  # type: ignore
except Exception:
    def get_cognitive_logger(*args, **kwargs):  # type: ignore
        return None
    def setup_logging(*args, **kwargs):  # type: ignore
        return None


def __getattr__(name: str):
    """Lazy-load CognitiveAgent on first access to avoid slow torch import at startup."""
    if name == "CognitiveAgent":
        global CognitiveAgent
        from .orchestration.cognitive_agent import CognitiveAgent  # type: ignore
        return CognitiveAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CognitiveAgent",
    "CognitiveConfig",
    "get_cognitive_logger",
    "setup_logging",
]


# ---- Backward-compatible module aliases (Phase 6 normalization) ----
# We keep the old import paths working without reintroducing the old top-level
# directories in `src/`.
import importlib
import sys


def _alias_module(old: str, new: str) -> None:  # pragma: no cover
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception:
        # Keep imports failing loudly later if the target module is broken.
        return


# NOTE: We intentionally do NOT alias legacy modules like `src.chat` / `src.processing`
# here anymore. Instead, we provide explicit compatibility shim packages under
# `src/chat/`, `src/processing/`, etc., so that legacy imports continue to work
# while emitting targeted DeprecationWarnings.