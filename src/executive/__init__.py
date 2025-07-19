"""
Executive Functioning Module

This module implements the executive control system that orchestrates all cognitive processes.
It provides goal management, task planning, decision making, and cognitive control.

Key Components:
- GoalManager: Hierarchical goal tracking and prioritization
- TaskPlanner: Goal decomposition and task sequencing  
- DecisionEngine: Multi-criteria decision making
- CognitiveController: Resource allocation and process coordination
- ExecutiveAgent: Main orchestrator integrating all components
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
