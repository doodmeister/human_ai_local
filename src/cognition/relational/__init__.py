"""Phase 2, Layer 2: Relational Field — how relationships feel and shape us.

Personality develops *through* relationships.  This layer tracks not just
facts about relationships, but their **felt quality** — the warmth, tension,
or comfort a particular person evokes — and how those felt dynamics
influence drives and emergent patterns over time.

Public API
----------
RelationalModel    – deep model of one significant relationship
RelationalField    – container of all relationships + current interlocutor
RelationalConfig   – tuning knobs (decay, thresholds, limits)
RelationalProcessor – updates relationships from turn-level data
"""

from .relational_state import RelationalModel, RelationalField
from .relational_config import RelationalConfig
from .relational_processor import RelationalProcessor

__all__ = [
    "RelationalModel",
    "RelationalField",
    "RelationalConfig",
    "RelationalProcessor",
]
