"""Executive Functioning Module

Phase 2 (Executive Collapse):
        - The single executive owner is `ExecutiveController` in `executive_core.py`.
        - Legacy components in this package (decision engine, controller, planners) are
            advisor-only and must not commit actions.
        - `ExecutiveAgent` is kept as a backwards-compatible facade that delegates
            turn execution to the executive core.
"""

from .goal_manager import GoalManager
from .task_planner import TaskPlanner
from .decision_engine import DecisionEngine
from .cognitive_controller import CognitiveController
from .executive_agent import ExecutiveAgent

__all__ = [
    'GoalManager',
    'TaskPlanner', 
    'DecisionEngine',
    'CognitiveController',
    'ExecutiveAgent'
]
