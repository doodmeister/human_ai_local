"""
Constraints for GOAP planning.

Constraints restrict the feasibility of actions and plans based on:
- Resource availability (memory, energy, tokens)
- Temporal requirements (deadlines, time windows)
- Dependencies (action ordering, state requirements)

Constraints are checked during A* search to prune infeasible paths early.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .world_state import WorldState
from .action_library import Action


class Constraint(ABC):
    """
    Abstract base class for planning constraints.
    
    Constraints define feasibility conditions that must be satisfied
    for actions to be applicable and plans to be valid.
    """
    
    @abstractmethod
    def is_satisfied(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if constraint is satisfied.
        
        Args:
            current_state: Current world state
            action: Action being considered (optional)
            plan_context: Planning context (time, resources used, etc.)
            
        Returns:
            True if constraint satisfied, False otherwise
        """
        pass
    
    @abstractmethod
    def get_violation_reason(self) -> str:
        """Get human-readable reason for constraint violation."""
        pass


@dataclass
class ResourceConstraint(Constraint):
    """
    Resource availability constraint.
    
    Ensures that resource usage doesn't exceed available capacity.
    Examples: memory, energy, API tokens, cognitive load
    """
    
    resource_name: str
    max_capacity: float
    current_usage_key: str = ""  # WorldState key for current usage
    
    def __post_init__(self):
        if not self.current_usage_key:
            self.current_usage_key = f"{self.resource_name}_used"
        self._violation_reason = ""
    
    def is_satisfied(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if resource constraint is satisfied."""
        # Get current resource usage
        current_usage = current_state.get(self.current_usage_key, 0.0)
        
        # If checking action, add its resource cost
        if action:
            # Use getattr to avoid static typing errors if Action doesn't declare attributes
            action_cost = float(getattr(action, 'cost', 0.0) or 0.0)
            # Check if action specifies resource usage in a resources mapping
            resources = getattr(action, 'resources', None)
            if isinstance(resources, dict) and self.resource_name in resources:
                try:
                    action_cost = float(resources[self.resource_name])
                except (TypeError, ValueError):
                    # If resource value is not numeric, fallback to previous action_cost
                    pass
            
            total_usage = current_usage + action_cost
            
            if total_usage > self.max_capacity:
                self._violation_reason = (
                    f"Resource '{self.resource_name}' would exceed capacity: "
                    f"{total_usage:.2f} > {self.max_capacity:.2f}"
                )
                return False
        
        # Check current state only
        if current_usage > self.max_capacity:
            self._violation_reason = (
                f"Resource '{self.resource_name}' exceeds capacity: "
                f"{current_usage:.2f} > {self.max_capacity:.2f}"
            )
            return False
        
        return True
    
    def get_violation_reason(self) -> str:
        return self._violation_reason


@dataclass
class TemporalConstraint(Constraint):
    """
    Temporal constraint (deadline or time window).
    
    Ensures actions complete within required time windows.
    """
    
    deadline: Optional[datetime] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    current_time_key: str = "current_time"
    
    def __post_init__(self):
        self._violation_reason = ""
    
    def is_satisfied(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if temporal constraint is satisfied."""
        # Get current time from state or context
        current_time = current_state.get(self.current_time_key)
        if current_time is None and plan_context:
            current_time = plan_context.get('current_time')
        
        if current_time is None:
            return True  # No time tracking, assume satisfied
        
        # Convert to datetime if needed
        if isinstance(current_time, (int, float)):
            # Assume Unix timestamp
            current_time = datetime.fromtimestamp(current_time)
        
        # Check deadline
        if self.deadline and current_time > self.deadline:
            self._violation_reason = (
                f"Deadline exceeded: {current_time} > {self.deadline}"
            )
            return False
        
        # Check time window
        if self.time_window_start and current_time < self.time_window_start:
            self._violation_reason = (
                f"Before time window: {current_time} < {self.time_window_start}"
            )
            return False
        
        if self.time_window_end and current_time > self.time_window_end:
            self._violation_reason = (
                f"After time window: {current_time} > {self.time_window_end}"
            )
            return False
        
        # If action provided, check if it can complete before deadline
        if action and self.deadline:
            action_duration = getattr(action, 'duration', 0)
            if action_duration > 0:
                completion_time = current_time + timedelta(seconds=action_duration)
                if completion_time > self.deadline:
                    self._violation_reason = (
                        f"Action would miss deadline: "
                        f"{completion_time} > {self.deadline}"
                    )
                    return False
        
        return True
    
    def get_violation_reason(self) -> str:
        return self._violation_reason


@dataclass
class DependencyConstraint(Constraint):
    """
    Dependency constraint (action ordering).
    
    Ensures certain actions must be completed before others.
    Example: "gather_data" must precede "analyze_data"
    """
    
    required_action: str
    dependent_action: str
    required_state_key: Optional[str] = None  # WorldState key indicating completion
    
    def __post_init__(self):
        if not self.required_state_key:
            self.required_state_key = f"{self.required_action}_complete"
        self._violation_reason = ""
    
    def is_satisfied(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if dependency constraint is satisfied."""
        # If checking a specific action
        if action and action.name == self.dependent_action:
            # Check if required action has been completed
            if not current_state.get(self.required_state_key, False):
                self._violation_reason = (
                    f"Action '{self.dependent_action}' requires "
                    f"'{self.required_action}' to be completed first"
                )
                return False
        
        return True
    
    def get_violation_reason(self) -> str:
        return self._violation_reason


@dataclass
class StateConstraint(Constraint):
    """
    State-based constraint.
    
    Ensures world state satisfies certain conditions.
    Example: "cognitive_load" must be below 0.8
    """
    
    state_key: str
    operator: str  # '<', '>', '<=', '>=', '==', '!='
    threshold: float
    
    def __post_init__(self):
        self._violation_reason = ""
        valid_operators = ['<', '>', '<=', '>=', '==', '!=']
        if self.operator not in valid_operators:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of: {valid_operators}"
            )
    
    def is_satisfied(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if state constraint is satisfied."""
        state_value = current_state.get(self.state_key)
        
        if state_value is None:
            # Key not in state - assume unconstrained
            return True
        
        # Evaluate constraint
        try:
            if self.operator == '<':
                satisfied = state_value < self.threshold
            elif self.operator == '>':
                satisfied = state_value > self.threshold
            elif self.operator == '<=':
                satisfied = state_value <= self.threshold
            elif self.operator == '>=':
                satisfied = state_value >= self.threshold
            elif self.operator == '==':
                satisfied = abs(state_value - self.threshold) < 1e-9
            elif self.operator == '!=':
                satisfied = abs(state_value - self.threshold) >= 1e-9
            else:
                return True  # Unknown operator, assume satisfied
            
            if not satisfied:
                self._violation_reason = (
                    f"State constraint violated: {self.state_key} "
                    f"({state_value}) {self.operator} {self.threshold}"
                )
            
            return satisfied
            
        except TypeError:
            # Can't compare values, assume satisfied
            return True
    
    def get_violation_reason(self) -> str:
        return self._violation_reason


class ConstraintChecker:
    """
    Manages and checks multiple constraints.
    
    Used during planning to validate actions and states.
    """
    
    def __init__(self, constraints: Optional[List[Constraint]] = None):
        """
        Initialize constraint checker.
        
        Args:
            constraints: List of constraints to check
        """
        self.constraints = constraints or []
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the checker."""
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint: Constraint) -> None:
        """Remove a constraint from the checker."""
        if constraint in self.constraints:
            self.constraints.remove(constraint)
    
    def check_all(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Check all constraints.
        
        Args:
            current_state: Current world state
            action: Action being considered (optional)
            plan_context: Planning context (optional)
            
        Returns:
            Tuple of (all_satisfied, violation_reasons)
        """
        violations = []
        
        for constraint in self.constraints:
            if not constraint.is_satisfied(current_state, action, plan_context):
                violations.append(constraint.get_violation_reason())
        
        return len(violations) == 0, violations
    
    def is_action_feasible(
        self,
        action: Action,
        current_state: WorldState,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if action is feasible given constraints.
        
        Args:
            action: Action to check
            current_state: Current world state
            plan_context: Planning context (optional)
            
        Returns:
            True if action satisfies all constraints
        """
        satisfied, _ = self.check_all(current_state, action, plan_context)
        return satisfied
    
    def get_violated_constraints(
        self,
        current_state: WorldState,
        action: Optional[Action] = None,
        plan_context: Optional[Dict[str, Any]] = None
    ) -> List[Constraint]:
        """
        Get list of violated constraints.
        
        Args:
            current_state: Current world state
            action: Action being considered (optional)
            plan_context: Planning context (optional)
            
        Returns:
            List of constraints that are violated
        """
        violated = []
        
        for constraint in self.constraints:
            if not constraint.is_satisfied(current_state, action, plan_context):
                violated.append(constraint)
        
        return violated


# Helper functions for creating common constraints

def create_resource_constraint(
    resource_name: str,
    max_capacity: float,
    current_usage_key: Optional[str] = None
) -> ResourceConstraint:
    """Create a resource constraint."""
    return ResourceConstraint(
        resource_name=resource_name,
        max_capacity=max_capacity,
        current_usage_key=current_usage_key or f"{resource_name}_used"
    )


def create_deadline_constraint(
    deadline: datetime,
    current_time_key: str = "current_time"
) -> TemporalConstraint:
    """Create a deadline constraint."""
    return TemporalConstraint(
        deadline=deadline,
        current_time_key=current_time_key
    )


def create_time_window_constraint(
    start: datetime,
    end: datetime,
    current_time_key: str = "current_time"
) -> TemporalConstraint:
    """Create a time window constraint."""
    return TemporalConstraint(
        time_window_start=start,
        time_window_end=end,
        current_time_key=current_time_key
    )


def create_dependency_constraint(
    required_action: str,
    dependent_action: str
) -> DependencyConstraint:
    """Create an action dependency constraint."""
    return DependencyConstraint(
        required_action=required_action,
        dependent_action=dependent_action
    )


def create_state_constraint(
    state_key: str,
    operator: str,
    threshold: float
) -> StateConstraint:
    """Create a state-based constraint."""
    return StateConstraint(
        state_key=state_key,
        operator=operator,
        threshold=threshold
    )
