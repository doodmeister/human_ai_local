"""Drive system — Layer 0 of the Phase 2 cognitive architecture.

Drives are unsatisfied needs that create motivational pressure. They are
always running, always influencing cognition.  Unlike goals (explicit
objectives), drives are underlying tensions that *generate* goals and shape
all processing.

Public API:
    DriveState      – current drive levels
    DriveProcessor  – update drives from experience / time
    DriveConfig     – tuning knobs
    DRIVE_NAMES     – canonical list of drive identifiers
"""

from .drive_state import DriveState, DRIVE_NAMES
from .drive_processor import DriveProcessor
from .drive_config import DriveConfig

__all__ = [
    "DriveState",
    "DriveProcessor",
    "DriveConfig",
    "DRIVE_NAMES",
]
