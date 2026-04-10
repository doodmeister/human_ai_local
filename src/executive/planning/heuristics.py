"""
Heuristic functions for GOAP planning.

Heuristics guide A* search toward the goal by estimating the cost to reach
the goal from any given state. Good heuristics are:

1. **Admissible**: Never overestimate the true cost (ensures optimal plans)
2. **Consistent**: h(n) ≤ cost(n, n') + h(n') for any neighbors n and n'
3. **Informative**: Close to true cost (reduces search space)

Reference: Pearl, J. (1984). "Heuristics: Intelligent Search Strategies"
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

from .world_state import WorldState

__all__ = [
    "Heuristic",
    "goal_distance_heuristic",
    "weighted_goal_distance_heuristic",
    "relaxed_plan_heuristic",
    "zero_heuristic",
    "max_heuristic",
    "CompositeHeuristic",
    "DEFAULT_HEURISTIC",
    "HEURISTICS",
    "get_heuristic",
]

_PRIORITY_PREFIX_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("critical_", 3.0),
    ("important_", 2.0),
    ("optional_", 0.5),
)
_NON_ADMISSIBLE_HEURISTIC_NAMES = frozenset({"weighted_goal_distance"})


def _goal_key_is_satisfied(current: WorldState, key: str, goal_value: object) -> bool:
    return current.has(key) and current.get(key) == goal_value


def _count_unsatisfied_goal_keys(current: WorldState, goal: WorldState) -> int:
    unsatisfied = 0
    for key, goal_value in goal.state.items():
        if not _goal_key_is_satisfied(current, key, goal_value):
            unsatisfied += 1
    return unsatisfied


def _weight_for_goal_key(key: str) -> float:
    for prefix, weight in _PRIORITY_PREFIX_WEIGHTS:
        if key.startswith(prefix):
            return weight
    return 1.0


@runtime_checkable
class Heuristic(Protocol):
    """Protocol for heuristic functions."""

    def __call__(self, current: WorldState, goal: WorldState) -> float:
        """
        Estimate cost from current state to goal.

        Args:
            current: Current world state
            goal: Goal world state

        Returns:
            Estimated cost (must be admissible for optimal plans)
        """
        ...


def goal_distance_heuristic(current: WorldState, goal: WorldState) -> float:
    """
    Count unsatisfied goal keys with presence-aware comparisons.

    This is a conservative ranking heuristic for GOAP states. It correctly
    treats a missing key as distinct from a key explicitly set to None.

    Example:
        current = WorldState({"has_data": False, "has_document": False})
        goal = WorldState({"has_data": True, "has_document": True})
        h = goal_distance_heuristic(current, goal)  # Returns 2.0

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Number of goal keys not yet satisfied
    """
    return float(_count_unsatisfied_goal_keys(current, goal))


def weighted_goal_distance_heuristic(
    current: WorldState, goal: WorldState
) -> float:
    """
    Priority-weighted goal distance heuristic.

    Uses key prefixes to determine importance:
    - "critical_": weight 3.0 (e.g., critical_task_complete)
    - "important_": weight 2.0 (e.g., important_data_gathered)
    - "optional_": weight 0.5 (e.g., optional_documentation_created)
    - default: weight 1.0

    This heuristic is intentionally non-admissible in the general case and
    should only be used when ranking speed matters more than A* optimality.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Weighted sum of unsatisfied goals
    """
    total_cost = 0.0
    for key, goal_value in goal.state.items():
        if not _goal_key_is_satisfied(current, key, goal_value):
            total_cost += _weight_for_goal_key(key)
    return total_cost


def relaxed_plan_heuristic(current: WorldState, goal: WorldState) -> float:
    """
    Conservative placeholder for a relaxed-planning heuristic.

    A true relaxed-plan heuristic needs access to the action library so it can
    build a delete-relaxed plan. This function does not have that information,
    so it falls back to the same presence-aware goal counting used by
    goal_distance_heuristic.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Conservative lower-complexity proxy for a relaxed plan estimate
    """
    return goal_distance_heuristic(current, goal)


def zero_heuristic(current: WorldState, goal: WorldState) -> float:
    """
    Zero heuristic: always returns 0.

    This turns A* into Dijkstra's algorithm (uniform cost search).
    Useful for debugging or when no good heuristic is available.

    Always admissible but not informative (explores all nodes).

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Always 0.0
    """
    return 0.0


def max_heuristic(current: WorldState, goal: WorldState) -> float:
    """
    Maximum of the currently safe built-in heuristics.

    This keeps the helper aligned with the conservative heuristics exported by
    this module instead of combining in the explicitly non-admissible weighted
    heuristic.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Maximum of goal_distance and relaxed_plan
    """
    h1 = goal_distance_heuristic(current, goal)
    h2 = relaxed_plan_heuristic(current, goal)
    return max(h1, h2)


class CompositeHeuristic:
    """
    Composite heuristic that combines multiple heuristics.

    Supports:
    - max: Take maximum (admissible if all components admissible)
    - avg: Take average (not admissible in general)
    - weighted: Weighted sum (not admissible in general)
    """

    def __init__(
        self,
        heuristics: list[Heuristic],
        mode: str = "max",
        weights: list[float] | None = None,
    ):
        """
        Initialize composite heuristic.

        Args:
            heuristics: List of heuristic functions
            mode: Combination mode ("max", "avg", "weighted")
            weights: Weights for "weighted" mode (must sum to 1.0)
        """
        if not heuristics:
            raise ValueError("heuristics must be non-empty")
        if not all(callable(heuristic) for heuristic in heuristics):
            raise TypeError("heuristics must contain only callable heuristics")
        if mode not in {"max", "avg", "weighted"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode != "weighted" and weights is not None:
            raise ValueError("weights are only supported in weighted mode")

        self.heuristics = list(heuristics)
        self.mode = mode
        self.weights = list(weights) if weights is not None else None

        if mode in {"avg", "weighted"}:
            warnings.warn(
                f"CompositeHeuristic mode '{mode}' is non-admissible in general.",
                UserWarning,
                stacklevel=2,
            )

        if mode == "weighted":
            if self.weights is None:
                raise ValueError("weights required for weighted mode")
            if len(self.weights) != len(self.heuristics):
                raise ValueError("weights must match number of heuristics")
            if any(weight < 0.0 for weight in self.weights):
                raise ValueError("weights must be non-negative")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError("weights must sum to 1.0")

    def __call__(self, current: WorldState, goal: WorldState) -> float:
        """Compute composite heuristic value."""
        values = [h(current, goal) for h in self.heuristics]

        if self.mode == "max":
            return max(values)
        if self.mode == "avg":
            return sum(values) / len(values)
        if self.mode == "weighted":
            assert self.weights is not None
            return sum(value * weight for value, weight in zip(values, self.weights, strict=True))
        raise ValueError(f"Unknown mode: {self.mode}")


# Default heuristic for general use
DEFAULT_HEURISTIC = goal_distance_heuristic

# Heuristic registry for easy lookup
HEURISTICS = {
    "goal_distance": goal_distance_heuristic,
    "weighted_goal_distance": weighted_goal_distance_heuristic,
    "relaxed_plan": relaxed_plan_heuristic,
    "zero": zero_heuristic,
    "max": max_heuristic,
}


def get_heuristic(name: str) -> Heuristic:
    """
    Get heuristic function by name.

    Args:
        name: Heuristic name (see HEURISTICS dict)

    Returns:
        Heuristic function

    Raises:
        ValueError: If heuristic name not found
    """
    if name == "default":
        return DEFAULT_HEURISTIC
    if name not in HEURISTICS:
        available = ", ".join([*HEURISTICS.keys(), "default"])
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {available}"
        )
    if name in _NON_ADMISSIBLE_HEURISTIC_NAMES:
        warnings.warn(
            f"Heuristic '{name}' is non-admissible and may produce suboptimal A* plans.",
            UserWarning,
            stacklevel=2,
        )
    return HEURISTICS[name]
