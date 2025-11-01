"""
GOAP Planning Module - Goal-Oriented Action Planning

This module implements Goal-Oriented Action Planning (GOAP) for automated
task planning. GOAP uses A* search over a state space to find optimal action
sequences that achieve goals.

Components:
- world_state: State representation for planning
- action_library: Predefined actions with preconditions/effects
- goap_planner: A* search-based planner
- heuristics: Planning heuristics for efficient search

Reference: Orkin, J. (2006). "Three States and a Plan: The A.I. of F.E.A.R."
"""

from .world_state import WorldState, merge_states
from .action_library import Action, ActionLibrary, create_default_action_library
from .goap_planner import GOAPPlanner, Plan, PlanStep, create_planner
from . import heuristics
from .heuristics import (
    goal_distance_heuristic,
    weighted_goal_distance_heuristic,
    relaxed_plan_heuristic,
    zero_heuristic,
    max_heuristic,
    CompositeHeuristic,
    get_heuristic,
    Heuristic,
    DEFAULT_HEURISTIC,
)
from .constraints import (
    Constraint,
    ResourceConstraint,
    TemporalConstraint,
    DependencyConstraint,
    StateConstraint,
    ConstraintChecker,
    create_resource_constraint,
    create_deadline_constraint,
    create_time_window_constraint,
    create_dependency_constraint,
    create_state_constraint,
)
from .replanning import (
    ReplanningEngine,
    FailureReason,
    PlanFailure,
)

__all__ = [
    "WorldState",
    "merge_states",
    "Action",
    "ActionLibrary",
    "create_default_action_library",
    "GOAPPlanner",
    "Plan",
    "PlanStep",
    "create_planner",
    "heuristics",
    "goal_distance_heuristic",
    "weighted_goal_distance_heuristic",
    "relaxed_plan_heuristic",
    "zero_heuristic",
    "max_heuristic",
    "CompositeHeuristic",
    "get_heuristic",
    "Heuristic",
    "DEFAULT_HEURISTIC",
    "Constraint",
    "ResourceConstraint",
    "TemporalConstraint",
    "DependencyConstraint",
    "StateConstraint",
    "ConstraintChecker",
    "create_resource_constraint",
    "create_deadline_constraint",
    "create_time_window_constraint",
    "create_dependency_constraint",
    "create_state_constraint",
    "ReplanningEngine",
    "FailureReason",
    "PlanFailure",
]
