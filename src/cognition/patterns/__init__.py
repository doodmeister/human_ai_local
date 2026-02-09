"""Phase 2, Layer 3: Emergent Patterns — tendencies that arise from experience.

Instead of predefined personality traits, patterns *emerge* from:

- Drive satisfaction history
- Relational experiences
- Felt-sense tendencies
- Internal conflict resolutions

Big Five becomes a **description layer** — how we describe emergent
patterns to ourselves and others, not the underlying truth.

Public API
----------
EmergentPattern     – a single behavioral/cognitive pattern
PatternField        – collection of all emergent patterns
PatternConfig       – tuning knobs
PatternDetector     – heuristic pattern detection from experience data
BIG_FIVE_DIMENSIONS – tuple of the five personality dimensions
"""

from .pattern_state import EmergentPattern, PatternField, BIG_FIVE_DIMENSIONS
from .pattern_config import PatternConfig
from .pattern_detector import PatternDetector

__all__ = [
    "EmergentPattern",
    "PatternField",
    "PatternConfig",
    "PatternDetector",
    "BIG_FIVE_DIMENSIONS",
]
