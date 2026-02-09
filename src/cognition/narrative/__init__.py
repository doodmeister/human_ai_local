"""Phase 2, Layer 5: Narrative — the agent's story of itself.

The narrative is the highest-level cognitive layer.  It synthesises
lower-layer data (drives, patterns, self-model, relationships) into
a coherent autobiographical account that can be injected into the
LLM system prompt.

Public API
----------
* ``SelfNarrative`` — the narrative data structure
* ``NarrativeConstructor`` — constructs / rebuilds the narrative
* ``NarrativeConfig`` — tuning knobs
"""

from .narrative_config import NarrativeConfig
from .narrative_state import SelfNarrative
from .narrative_constructor import NarrativeConstructor

__all__ = [
    "SelfNarrative",
    "NarrativeConstructor",
    "NarrativeConfig",
]
