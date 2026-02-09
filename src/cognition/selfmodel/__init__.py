"""Phase 2, Layer 4: Self-Model (Partial, Biased).

The agent's theory of itself — what it believes about its own
patterns, drives, strengths, weaknesses, and values.  Crucially,
this model is **incomplete and biased** by current mood and
limited self-knowledge.

Public API
----------
* ``SelfModel`` — the self-model data structure
* ``SelfDiscovery`` — a self-discovery moment
* ``SelfModelBuilder`` — constructs / rebuilds the self-model
* ``SelfModelConfig`` — tuning knobs
"""

from .self_model_config import SelfModelConfig
from .self_model_state import SelfDiscovery, SelfModel
from .self_model_builder import SelfModelBuilder

__all__ = [
    "SelfModel",
    "SelfDiscovery",
    "SelfModelBuilder",
    "SelfModelConfig",
]
