"""
HTN (Hierarchical Task Networks) Goal Management Module.

This module provides hierarchical goal decomposition using HTN planning:
- Goal taxonomy (primitive vs compound goals)
- Decomposition methods (rules for breaking down goals)
- Recursive decomposition algorithm
- Conflict detection and resolution
- Predictive goal analytics

HTN complements GOAP (Phase 2):
- GOAP: Operational planning (how to achieve a specific goal)
- HTN: Strategic planning (which goals to pursue, in what order)

Architecture:
    User/System Goal (high-level)
           ↓
    HTN Decomposition (strategic)
           ↓
    Subtasks/Goals (intermediate)
           ↓
    GOAP Planning (operational)
           ↓
    Action Sequences (executable)
"""

from .goal_taxonomy import GoalType, Goal, GoalStatus, GoalPriority
from .decomposition import Method, DecompositionRule, create_default_methods, OrderingConstraint, SubtaskTemplate
from .htn_manager import HTNManager, DecompositionResult
from .htn_goal_manager_adapter import HTNGoalManagerAdapter
from .htn_goap_bridge import HTNGOAPBridge, PlanningResult, create_default_bridge, plan_goal_with_goap
from .priority_calculator import (
    GoalPriorityCalculator, 
    GoalContext, 
    PriorityWeights, 
    PriorityScore, 
    PriorityFactor
)
from .conflict_detection import (
    ConflictDetector,
    Conflict,
    ConflictReport,
    ConflictType,
    ConflictSeverity
)

__all__ = [
    "GoalType",
    "Goal",
    "GoalStatus",
    "GoalPriority",
    "Method",
    "DecompositionRule",
    "create_default_methods",
    "OrderingConstraint",
    "SubtaskTemplate",
    "HTNManager",
    "DecompositionResult",
    "HTNGoalManagerAdapter",
    "HTNGOAPBridge",
    "PlanningResult",
    "create_default_bridge",
    "plan_goal_with_goap",
    # Week 10: Goal Intelligence
    "GoalPriorityCalculator",
    "GoalContext",
    "PriorityWeights",
    "PriorityScore",
    "PriorityFactor",
    "ConflictDetector",
    "Conflict",
    "ConflictReport",
    "ConflictType",
    "ConflictSeverity",
]
