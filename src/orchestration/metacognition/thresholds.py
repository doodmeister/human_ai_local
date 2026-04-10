from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CognitiveThresholds", "DEFAULT_COGNITIVE_THRESHOLDS"]


@dataclass(frozen=True, slots=True)
class CognitiveThresholds:
    """Shared default thresholds for metacognitive planning and scheduling."""

    uncertainty_threshold: float = 0.45
    cognitive_load_threshold: float = 0.75

    def __post_init__(self) -> None:
        for name, value in (
            ("uncertainty_threshold", self.uncertainty_threshold),
            ("cognitive_load_threshold", self.cognitive_load_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")


DEFAULT_COGNITIVE_THRESHOLDS = CognitiveThresholds()