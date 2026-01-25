"""
Replanning Engine for GOAP.

Handles plan failures and dynamic replanning during execution:
- Detects when plan becomes invalid (action failure, world state divergence)
- Replans from current state
- Reuses valid plan prefix when possible (plan repair)
- Provides reactive planning for dynamic environments
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

from .world_state import WorldState
from .goap_planner import GOAPPlanner, Plan, PlanStep
from .action_library import Action

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """Reasons for plan failure."""
    ACTION_FAILED = "action_failed"  # Action execution failed
    PRECONDITION_VIOLATED = "precondition_violated"  # Preconditions no longer met
    GOAL_CHANGED = "goal_changed"  # Goal was modified
    STATE_DIVERGED = "state_diverged"  # World state diverged from expected
    CONSTRAINT_VIOLATED = "constraint_violated"  # Constraint no longer satisfied
    TIMEOUT = "timeout"  # Plan execution exceeded time limit


@dataclass
class PlanFailure:
    """Information about a plan failure."""
    reason: FailureReason
    failed_step: Optional[PlanStep] = None
    failed_action: Optional[Action] = None
    current_state: Optional[WorldState] = None
    expected_state: Optional[WorldState] = None
    details: str = ""


class ReplanningEngine:
    """
    Engine for handling plan failures and replanning.
    
    Strategies:
    1. Full Replan: Discard entire plan, replan from current state
    2. Plan Repair: Keep valid prefix, replan from failure point
    3. Action Retry: Retry failed action with adjusted parameters
    """
    
    def __init__(self, planner: GOAPPlanner):
        """
        Initialize replanning engine.
        
        Args:
            planner: GOAP planner instance for replanning
        """
        self.planner = planner
        self.replan_count = 0
        self.repair_count = 0
        self.retry_count = 0
    
    def detect_failure(
        self,
        plan: Plan,
        current_step_index: int,
        current_state: WorldState,
        execution_result: dict
    ) -> Optional[PlanFailure]:
        """
        Detect if plan has failed during execution.
        
        Args:
            plan: Current plan being executed
            current_step_index: Index of step being/just executed
            current_state: Actual current world state
            execution_result: Result of action execution
            
        Returns:
            PlanFailure if failure detected, None otherwise
        """
        if current_step_index >= len(plan.steps):
            return None
        
        current_step = plan.steps[current_step_index]
        
        # Check execution result
        if execution_result.get('failed', False):
            return PlanFailure(
                reason=FailureReason.ACTION_FAILED,
                failed_step=current_step,
                failed_action=current_step.action,
                current_state=current_state,
                details=execution_result.get('error', 'Action execution failed')
            )
        
        # Check if current state diverged from expected
        if not current_state.satisfies(current_step.state_after):
            # Calculate divergence
            expected_keys = set(current_step.state_after.state.keys())
            actual_keys = set(current_state.state.keys())
            
            missing_keys = expected_keys - actual_keys
            diverged_keys = [
                k for k in expected_keys & actual_keys
                if current_state.get(k) != current_step.state_after.get(k)
            ]
            
            return PlanFailure(
                reason=FailureReason.STATE_DIVERGED,
                failed_step=current_step,
                current_state=current_state,
                expected_state=current_step.state_after,
                details=f"State diverged: missing={missing_keys}, changed={diverged_keys}"
            )
        
        # Check next action preconditions (if there is a next step)
        if current_step_index + 1 < len(plan.steps):
            next_step = plan.steps[current_step_index + 1]
            if not next_step.action.is_applicable(current_state):
                return PlanFailure(
                    reason=FailureReason.PRECONDITION_VIOLATED,
                    failed_step=next_step,
                    failed_action=next_step.action,
                    current_state=current_state,
                    details=f"Preconditions for {next_step.action.name} no longer met"
                )
        
        return None
    
    def replan(
        self,
        failure: PlanFailure,
        current_state: WorldState,
        goal_state: WorldState,
        original_plan: Plan,
        max_iterations: int = 1000
    ) -> Optional[Plan]:
        """
        Replan from current state after failure.
        
        Attempts plan repair first (reuse valid prefix), falls back to full replan.
        
        Args:
            failure: Information about the failure
            current_state: Current world state
            goal_state: Desired goal state
            original_plan: The plan that failed
            max_iterations: Maximum planning iterations
            
        Returns:
            New plan, or None if replanning failed
        """
        logger.info(f"Replanning after {failure.reason.value}: {failure.details}")
        
        # Try plan repair first (faster)
        repaired_plan = self._attempt_repair(
            failure, current_state, goal_state, original_plan, max_iterations
        )
        
        if repaired_plan:
            self.repair_count += 1
            logger.info(f"Plan repaired: {len(repaired_plan.steps)} steps")
            return repaired_plan
        
        # Fall back to full replan
        logger.info("Plan repair failed, attempting full replan")
        new_plan = self.planner.plan(
            initial_state=current_state,
            goal_state=goal_state,
            max_iterations=max_iterations
        )
        
        if new_plan:
            self.replan_count += 1
            logger.info(f"Replanned: {len(new_plan.steps)} steps")
        else:
            logger.warning("Replanning failed: no plan found")
        
        return new_plan
    
    def _attempt_repair(
        self,
        failure: PlanFailure,
        current_state: WorldState,
        goal_state: WorldState,
        original_plan: Plan,
        max_iterations: int
    ) -> Optional[Plan]:
        """
        Attempt to repair plan by keeping valid prefix.
        
        Strategy:
        1. Find last valid step before failure
        2. Keep steps up to that point
        3. Replan from failure state to goal
        4. Concatenate valid prefix + new suffix
        
        Args:
            failure: Failure information
            current_state: Current state
            goal_state: Goal state
            original_plan: Original plan
            max_iterations: Max planning iterations
            
        Returns:
            Repaired plan or None
        """
        if not failure.failed_step:
            return None
        
        # Find index of failed step
        failed_index = None
        for i, step in enumerate(original_plan.steps):
            if step == failure.failed_step:
                failed_index = i
                break
        
        if failed_index is None or failed_index == 0:
            # Can't repair if first step failed
            return None
        
        # Valid prefix is steps before failed step
        valid_prefix = original_plan.steps[:failed_index]
        
        # Replan from current state to goal
        new_suffix = self.planner.plan(
            initial_state=current_state,
            goal_state=goal_state,
            max_iterations=max_iterations
        )
        
        if not new_suffix:
            return None
        
        # Concatenate prefix + suffix
        all_steps = valid_prefix + new_suffix.steps
        
        # Renumber steps
        for i, step in enumerate(all_steps):
            step.step_number = i + 1
        
        # Calculate total cost
        total_cost = sum(step.cost for step in all_steps)
        
        # Create repaired plan
        repaired_plan = Plan(
            steps=all_steps,
            initial_state=original_plan.initial_state,
            goal_state=goal_state,
            total_cost=total_cost,
            nodes_expanded=new_suffix.nodes_expanded,  # Only suffix expansion
            planning_time_ms=new_suffix.planning_time_ms
        )
        
        return repaired_plan
    
    def should_retry_action(
        self,
        failure: PlanFailure,
        retry_count: int,
        max_retries: int = 3
    ) -> bool:
        """
        Determine if failed action should be retried.
        
        Args:
            failure: Failure information
            retry_count: Number of retries already attempted
            max_retries: Maximum retry attempts
            
        Returns:
            True if should retry
        """
        if retry_count >= max_retries:
            return False
        
        # Retry only for transient failures (not precondition violations)
        return failure.reason == FailureReason.ACTION_FAILED
    
    def get_statistics(self) -> dict:
        """
        Get replanning statistics.
        
        Returns:
            Dictionary with replan counts
        """
        return {
            'replan_count': self.replan_count,
            'repair_count': self.repair_count,
            'retry_count': self.retry_count,
            'total_adaptations': self.replan_count + self.repair_count + self.retry_count
        }


def create_replanning_engine(planner: GOAPPlanner) -> ReplanningEngine:
    """
    Factory function for creating replanning engine.
    
    Args:
        planner: GOAP planner instance
        
    Returns:
        ReplanningEngine instance
    """
    return ReplanningEngine(planner=planner)
