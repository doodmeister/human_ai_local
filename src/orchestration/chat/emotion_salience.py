from __future__ import annotations

from typing import Tuple

from src.cognition.attention.attention_manager import get_attention_manager


def estimate_salience_and_valence(text: str) -> Tuple[float, float]:
    """Compatibility wrapper.

    Phase 3: salience/valence computation is centralized in AttentionManager.
    """
    return get_attention_manager().estimate_salience_and_valence(text)
