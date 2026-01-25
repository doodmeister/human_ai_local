"""
HTN-GOAP Bridge.

Connects HTN strategic planning with GOAP operational planning.

Architecture:
    HTN Manager (strategic)
         ↓ (decompose compound goals)
    Primitive Goals
         ↓ (this bridge)
    GOAP Planner (operational)
         ↓ (action sequences)
    Executable Actions

The bridge converts:
- HTN primitive goals → GOAP planning problems
- HTN postconditions → GOAP goal states
- HTN preconditions → GOAP initial state constraints

Usage:
    bridge = HTNGOAPBridge(goap_planner, action_library)
    
    # For each primitive HTN goal
    plan = bridge.plan_primitive_goal(
        goal=primitive_goal,
        current_state=WorldState({'has_data': True})
    )
    
    if plan:
        execute_plan(plan)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

from .goal_taxonomy import Goal
from ..planning.goap_planner import GOAPPlanner, Plan
from ..planning.world_state import WorldState
from ..planning.action_library import ActionLibrary, create_default_action_library

logger = logging.getLogger(__name__)


@dataclass
class PlanningResult:
    """Result of planning for a primitive goal."""
    
    goal_id: str
    goal_description: str
    success: bool
    plan: Optional[Plan] = None
    error: Optional[str] = None
    
    # Metrics
    planning_time_ms: float = 0.0
    nodes_expanded: int = 0
    plan_length: int = 0
    plan_cost: float = 0.0


class HTNGOAPBridge:
    """
    Bridge between HTN planning and GOAP planning.
    
    Responsibilities:
    - Convert HTN primitive goals to GOAP problems
    - Validate preconditions before planning
    - Handle planning failures gracefully
    - Provide unified interface for execution
    """
    
    def __init__(
        self,
        goap_planner: Optional[GOAPPlanner] = None,
        action_library: Optional[ActionLibrary] = None
    ):
        """
        Initialize bridge.
        
        Args:
            goap_planner: GOAP planner (creates default if None)
            action_library: Action library (creates default if None)
        """
        # Use provided or create defaults
        if action_library is None:
            action_library = create_default_action_library()
        
        if goap_planner is None:
            # Create planner with default heuristic (None = uses built-in default)
            goap_planner = GOAPPlanner(
                action_library=action_library,
                heuristic=None  # Uses default goal_distance heuristic
            )
        
        self.goap_planner = goap_planner
        self.action_library = action_library
        
        # Track planning statistics
        self._planning_attempts = 0
        self._planning_successes = 0
        self._planning_failures = 0
    
    # ====================
    # Main Planning API
    # ====================
    
    def plan_primitive_goal(
        self,
        goal: Goal,
        current_state: WorldState,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> PlanningResult:
        """
        Plan how to achieve a primitive HTN goal using GOAP.
        
        Workflow:
        1. Validate goal is primitive
        2. Check preconditions satisfied
        3. Convert HTN postconditions → GOAP goal state
        4. Plan with GOAP
        5. Return result
        
        Args:
            goal: Primitive HTN goal
            current_state: Current world state
            plan_context: Optional context for planning
            
        Returns:
            PlanningResult with plan or error
        """
        self._planning_attempts += 1
        
        # 1. Validate goal is primitive
        if not goal.is_primitive():
            self._planning_failures += 1
            return PlanningResult(
                goal_id=goal.id,
                goal_description=goal.description,
                success=False,
                error=f"Goal is not primitive (type={goal.goal_type})"
            )
        
        # 2. Check preconditions
        if not self._check_preconditions(goal, current_state):
            self._planning_failures += 1
            return PlanningResult(
                goal_id=goal.id,
                goal_description=goal.description,
                success=False,
                error=f"Preconditions not satisfied: {goal.preconditions}"
            )
        
        # 3. Convert HTN postconditions → GOAP goal state
        goal_state = self._create_goal_state(goal)
        
        # 4. Plan with GOAP
        try:
            plan = self.goap_planner.plan(
                initial_state=current_state,
                goal_state=goal_state,
                plan_context=plan_context or {}
            )
            
            if plan:
                self._planning_successes += 1
                return PlanningResult(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    success=True,
                    plan=plan,
                    planning_time_ms=plan.planning_time_ms,
                    nodes_expanded=plan.nodes_expanded,
                    plan_length=len(plan.steps),
                    plan_cost=plan.total_cost
                )
            else:
                self._planning_failures += 1
                return PlanningResult(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    success=False,
                    error="GOAP planner returned no plan"
                )
        
        except Exception as e:
            self._planning_failures += 1
            logger.error(f"GOAP planning failed for goal {goal.id}: {e}")
            return PlanningResult(
                goal_id=goal.id,
                goal_description=goal.description,
                success=False,
                error=f"Planning exception: {str(e)}"
            )
    
    def plan_multiple_goals(
        self,
        goals: List[Goal],
        current_state: WorldState,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> List[PlanningResult]:
        """
        Plan for multiple primitive goals in sequence.
        
        Args:
            goals: List of primitive HTN goals
            current_state: Current world state
            plan_context: Optional context for planning
            
        Returns:
            List of PlanningResults (one per goal)
        """
        results = []
        state = current_state
        
        for goal in goals:
            result = self.plan_primitive_goal(goal, state, plan_context)
            results.append(result)
            
            # Update state if planning succeeded
            if result.success and result.plan:
                # Final state of plan becomes initial state for next goal
                state = result.plan.steps[-1].state_after
        
        return results
    
    # ====================
    # Conversion Helpers
    # ====================
    
    def _check_preconditions(self, goal: Goal, current_state: WorldState) -> bool:
        """
        Check if goal's preconditions are satisfied.
        
        Args:
            goal: HTN goal with preconditions
            current_state: Current world state
            
        Returns:
            True if all preconditions satisfied
        """
        if not goal.preconditions:
            return True  # No preconditions = always satisfied
        
        for key, required_value in goal.preconditions.items():
            current_value = current_state.get(key)
            if current_value != required_value:
                logger.debug(
                    f"Precondition not met for goal {goal.id}: "
                    f"{key}={current_value} (expected {required_value})"
                )
                return False
        
        return True
    
    def _create_goal_state(self, goal: Goal) -> WorldState:
        """
        Create GOAP goal state from HTN postconditions.
        
        Args:
            goal: HTN goal with postconditions
            
        Returns:
            WorldState representing desired outcome
        """
        return WorldState(goal.postconditions)
    
    # ====================
    # Statistics
    # ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return {
            'planning_attempts': self._planning_attempts,
            'planning_successes': self._planning_successes,
            'planning_failures': self._planning_failures,
            'success_rate': (
                self._planning_successes / self._planning_attempts
                if self._planning_attempts > 0 else 0.0
            )
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._planning_attempts = 0
        self._planning_successes = 0
        self._planning_failures = 0


# ====================
# Convenience Functions
# ====================

def create_default_bridge() -> HTNGOAPBridge:
    """Create bridge with default GOAP planner and action library."""
    return HTNGOAPBridge()


def plan_goal_with_goap(
    goal: Goal,
    current_state: Dict[str, Any]
) -> Optional[Plan]:
    """
    Convenience function to plan a single goal.
    
    Args:
        goal: Primitive HTN goal
        current_state: Current state as dict
        
    Returns:
        GOAP Plan or None if planning failed
    """
    bridge = create_default_bridge()
    state = WorldState(current_state)
    result = bridge.plan_primitive_goal(goal, state)
    return result.plan if result.success else None
