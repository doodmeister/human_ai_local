"""
Heuristic functions for GOAP planning.

Heuristics guide A* search toward the goal by estimating the cost to reach
the goal from any given state. Good heuristics are:

1. **Admissible**: Never overestimate the true cost (ensures optimal plans)
2. **Consistent**: h(n) ≤ cost(n, n') + h(n') for any neighbors n and n'
3. **Informative**: Close to true cost (reduces search space)

Reference: Pearl, J. (1984). "Heuristics: Intelligent Search Strategies"
"""

from typing import Protocol

from .world_state import WorldState


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
    Simple goal distance heuristic: count unsatisfied goal keys.

    This is admissible if each action costs at least 1.0 and can satisfy
    at most one goal key. In practice, actions may satisfy multiple keys,
    so this heuristic underestimates (which is good for A*).

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
    unsatisfied = 0
    for key, goal_value in goal.state.items():
        if current.get(key) != goal_value:
            unsatisfied += 1
    return float(unsatisfied)


def weighted_goal_distance_heuristic(
    current: WorldState, goal: WorldState
) -> float:
    """
    Weighted goal distance: assign different weights to different goals.

    Uses key prefixes to determine importance:
    - "critical_": weight 3.0 (e.g., critical_task_complete)
    - "important_": weight 2.0 (e.g., important_data_gathered)
    - "optional_": weight 0.5 (e.g., optional_documentation_created)
    - default: weight 1.0

    This is still admissible if action costs reflect these priorities.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Weighted sum of unsatisfied goals
    """
    total_cost = 0.0
    for key, goal_value in goal.state.items():
        if current.get(key) != goal_value:
            # Determine weight based on key prefix
            if key.startswith("critical_"):
                weight = 3.0
            elif key.startswith("important_"):
                weight = 2.0
            elif key.startswith("optional_"):
                weight = 0.5
            else:
                weight = 1.0
            total_cost += weight
    return total_cost


def relaxed_plan_heuristic(current: WorldState, goal: WorldState) -> float:
    """
    Relaxed planning heuristic: estimate cost by ignoring preconditions.

    This heuristic builds a "relaxed plan" where actions can be applied
    without checking preconditions. This gives a lower bound on the true
    cost, making it admissible.

    Algorithm:
    1. Start with current state
    2. Find cheapest action that makes progress toward any goal
    3. Apply action (ignoring preconditions)
    4. Repeat until all goals satisfied
    5. Return total cost

    This is more expensive to compute but more informative than simple
    goal distance.

    Note: This implementation uses a simplified version that doesn't
    require the full action library. For a complete implementation,
    inject the action library during planner initialization.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Estimated cost of relaxed plan
    """
    # Simplified version: assume average action cost of 1.5 per goal
    # (better than goal_distance which assumes 1.0)
    unsatisfied = 0
    for key, goal_value in goal.state.items():
        if current.get(key) != goal_value:
            unsatisfied += 1

    # Average action cost slightly higher than minimum
    avg_action_cost = 1.5
    return unsatisfied * avg_action_cost


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
    Maximum of multiple heuristics.

    Taking the max of multiple admissible heuristics is still admissible
    and often more informative than any single heuristic.

    Args:
        current: Current world state
        goal: Goal world state

    Returns:
        Maximum of goal_distance and weighted_goal_distance
    """
    h1 = goal_distance_heuristic(current, goal)
    h2 = weighted_goal_distance_heuristic(current, goal)
    return max(h1, h2)


class CompositeHeuristic:
    """
    Composite heuristic that combines multiple heuristics.

    Supports:
    - max: Take maximum (admissible if all components admissible)
    - avg: Take average (may not be admissible, use with caution)
    - weighted: Weighted sum (may not be admissible, use with caution)
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
        self.heuristics = heuristics
        self.mode = mode
        self.weights = weights

        if mode == "weighted":
            if weights is None:
                raise ValueError("weights required for weighted mode")
            if len(weights) != len(heuristics):
                raise ValueError("weights must match number of heuristics")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("weights must sum to 1.0")

    def __call__(self, current: WorldState, goal: WorldState) -> float:
        """Compute composite heuristic value."""
        values = [h(current, goal) for h in self.heuristics]

        if self.mode == "max":
            return max(values)
        elif self.mode == "avg":
            return sum(values) / len(values)
        elif self.mode == "weighted":
            return sum(v * w for v, w in zip(values, self.weights))  # type: ignore
        else:
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
    "default": DEFAULT_HEURISTIC,
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
    if name not in HEURISTICS:
        available = ", ".join(HEURISTICS.keys())
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {available}"
        )
    return HEURISTICS[name]
