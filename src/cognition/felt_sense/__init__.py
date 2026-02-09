"""Phase 2, Layer 1: Felt Sense — pre-conceptual emotional texture.

The felt sense exists *before* a labeled mood.  It captures the raw
qualitative character of the agent's current experience: metaphorical
body sensations, movement qualities, and intensity.  The ``MoodLabeler``
then derives a named mood from the felt sense (with explicit confidence).

Public API
----------
FeltSense        – current pre-conceptual texture
FeltSenseHistory – ring buffer of recent felt-sense snapshots
Mood             – labeled mood derived from felt sense
FeltSenseConfig  – tuning knobs
FeltSenseGenerator – creates felt sense from drives + recent experience
MoodLabeler      – converts felt sense → labeled Mood
"""

from .felt_sense_state import FeltSense, FeltSenseHistory
from .mood import Mood, MoodLabeler
from .felt_sense_config import FeltSenseConfig
from .felt_sense_generator import FeltSenseGenerator

__all__ = [
    "FeltSense",
    "FeltSenseHistory",
    "Mood",
    "MoodLabeler",
    "FeltSenseConfig",
    "FeltSenseGenerator",
]
