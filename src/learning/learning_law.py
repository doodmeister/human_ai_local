"""Phase 7 — Learning Law Unification.

Chosen law: utility-based.

All adaptive/learning effects should reduce to maximizing a single scalar utility:

  U = benefit - cost

Where both benefit and cost are normalized to [0, 1] (best-effort).

This module intentionally stays tiny: it is a shared primitive, not a new framework.
"""

from __future__ import annotations

from typing import Any


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def utility_score(
    *,
    benefit: Any,
    cost: Any,
    benefit_weight: float = 1.0,
    cost_weight: float = 1.0,
) -> float:
    """Compute utility as weighted benefit minus weighted cost.

    Inputs are clamped to [0, 1] as a safety guard.
    """
    b = clamp01(benefit)
    c = clamp01(cost)
    return (benefit_weight * b) - (cost_weight * c)
