"""
Action Library for GOAP Planning

Defines actions with preconditions, effects, and costs.
Actions are the building blocks of plans.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from .world_state import WorldState


@dataclass
class Action:
    """
    GOAP Action with preconditions, effects, and cost
    
    An action can be applied to a world state if its preconditions are met.
    When applied, it produces a new state with its effects.
    
    Attributes:
        name: Unique action identifier
        description: Human-readable description
        preconditions: State that must be satisfied to apply action
        effects: State changes that result from action
        cost: Base cost of action (time/effort)
        cognitive_load: Mental effort required (0.0 - 1.0)
        resources_needed: List of required resources
        duration_minutes: Estimated duration in minutes
        cost_function: Optional custom cost calculator
        
    Examples:
        >>> analyze = Action(
        ...     name="analyze_data",
        ...     preconditions=WorldState({'has_data': True}),
        ...     effects=WorldState({'analyzed': True}),
        ...     cost=10.0
        ... )
    """
    
    name: str
    description: str = ""
    preconditions: WorldState = field(default_factory=lambda: WorldState({}))
    effects: WorldState = field(default_factory=lambda: WorldState({}))
    cost: float = 1.0
    cognitive_load: float = 0.5
    resources_needed: List[str] = field(default_factory=list)
    duration_minutes: int = 30
    cost_function: Optional[Callable[[WorldState, WorldState], float]] = None
    
    def __post_init__(self):
        """Validate action parameters"""
        if not self.name:
            raise ValueError("Action name cannot be empty")
        if self.cost < 0:
            raise ValueError("Action cost must be non-negative")
        if not (0.0 <= self.cognitive_load <= 1.0):
            raise ValueError("Cognitive load must be between 0.0 and 1.0")
    
    def is_applicable(self, state: WorldState) -> bool:
        """
        Check if action can be applied in given state
        
        Args:
            state: Current world state
            
        Returns:
            True if all preconditions are satisfied
        """
        return state.satisfies(self.preconditions)
    
    def apply(self, state: WorldState) -> WorldState:
        """
        Apply action to state, producing new state
        
        Args:
            state: Current world state
            
        Returns:
            New world state with action effects applied
            
        Raises:
            ValueError: If action cannot be applied (preconditions not met)
        """
        if not self.is_applicable(state):
            raise ValueError(
                f"Action '{self.name}' cannot be applied: "
                f"preconditions not met. "
                f"Required: {self.preconditions}, Got: {state}"
            )
        
        # Apply effects to create new state
        return state.update(self.effects._state)
    
    def get_cost(self, from_state: WorldState, to_state: WorldState) -> float:
        """
        Calculate actual cost of applying action
        
        Can use custom cost function if provided, otherwise uses base cost.
        Cost may depend on states (e.g., longer path = higher cost).
        
        Args:
            from_state: State before action
            to_state: State after action
            
        Returns:
            Numeric cost (lower is better)
        """
        if self.cost_function:
            return self.cost_function(from_state, to_state)
        return self.cost
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Action({self.name})"
    
    def __str__(self) -> str:
        """Human-readable string"""
        return f"{self.name} (cost: {self.cost}, load: {self.cognitive_load})"


class ActionLibrary:
    """
    Collection of available actions for planning
    
    Manages a library of predefined actions that the planner can use.
    """
    
    def __init__(self):
        """Initialize empty action library"""
        self.actions: Dict[str, Action] = {}
    
    def add_action(self, action: Action) -> None:
        """
        Add action to library
        
        Args:
            action: Action to add
            
        Raises:
            ValueError: If action with same name already exists
        """
        if action.name in self.actions:
            raise ValueError(f"Action '{action.name}' already exists in library")
        self.actions[action.name] = action
    
    def get_action(self, name: str) -> Optional[Action]:
        """Get action by name"""
        return self.actions.get(name)
    
    def get_applicable_actions(self, state: WorldState) -> List[Action]:
        """
        Get all actions that can be applied in given state
        
        Args:
            state: Current world state
            
        Returns:
            List of actions whose preconditions are satisfied
        """
        return [
            action for action in self.actions.values()
            if action.is_applicable(state)
        ]
    
    def remove_action(self, name: str) -> None:
        """Remove action from library"""
        if name in self.actions:
            del self.actions[name]
    
    def clear(self) -> None:
        """Remove all actions"""
        self.actions.clear()
    
    def __len__(self) -> int:
        """Number of actions in library"""
        return len(self.actions)
    
    def __contains__(self, name: str) -> bool:
        """Check if action exists"""
        return name in self.actions
    
    def __iter__(self):
        """Iterate over actions"""
        return iter(self.actions.values())
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ActionLibrary({len(self)} actions)"


def create_default_action_library() -> ActionLibrary:
    """
    Create library with common predefined actions
    
    Returns:
        ActionLibrary with standard actions for cognitive tasks
    """
    library = ActionLibrary()
    
    # Analysis actions
    library.add_action(Action(
        name="analyze_data",
        description="Analyze available data to extract insights",
        preconditions=WorldState({'has_data': True}),
        effects=WorldState({'data_analyzed': True, 'has_insights': True}),
        cost=10.0,
        cognitive_load=0.7,
        duration_minutes=45
    ))
    
    library.add_action(Action(
        name="gather_data",
        description="Collect necessary data from sources",
        preconditions=WorldState({}),
        effects=WorldState({'has_data': True}),
        cost=5.0,
        cognitive_load=0.3,
        duration_minutes=20
    ))
    
    # Creation actions
    library.add_action(Action(
        name="create_document",
        description="Create a new document or artifact",
        preconditions=WorldState({'has_insights': True}),
        effects=WorldState({'document_created': True}),
        cost=15.0,
        cognitive_load=0.8,
        duration_minutes=60
    ))
    
    library.add_action(Action(
        name="draft_outline",
        description="Create an outline or structure",
        preconditions=WorldState({}),
        effects=WorldState({'has_outline': True}),
        cost=8.0,
        cognitive_load=0.6,
        duration_minutes=30
    ))
    
    # Communication actions
    library.add_action(Action(
        name="send_notification",
        description="Notify stakeholders of progress",
        preconditions=WorldState({'document_created': True}),
        effects=WorldState({'stakeholders_notified': True}),
        cost=2.0,
        cognitive_load=0.2,
        duration_minutes=10
    ))
    
    library.add_action(Action(
        name="schedule_meeting",
        description="Arrange a meeting with stakeholders",
        preconditions=WorldState({}),
        effects=WorldState({'meeting_scheduled': True}),
        cost=3.0,
        cognitive_load=0.3,
        duration_minutes=15
    ))
    
    # Verification actions
    library.add_action(Action(
        name="review_work",
        description="Review and verify completed work",
        preconditions=WorldState({'document_created': True}),
        effects=WorldState({'work_verified': True}),
        cost=10.0,
        cognitive_load=0.6,
        duration_minutes=40
    ))
    
    library.add_action(Action(
        name="run_tests",
        description="Execute tests to verify functionality",
        preconditions=WorldState({'document_created': True}),
        effects=WorldState({'tests_passed': True}),
        cost=12.0,
        cognitive_load=0.5,
        duration_minutes=50
    ))
    
    # Planning actions
    library.add_action(Action(
        name="create_plan",
        description="Develop a detailed plan of action",
        preconditions=WorldState({'has_insights': True}),
        effects=WorldState({'plan_created': True}),
        cost=8.0,
        cognitive_load=0.7,
        duration_minutes=35
    ))
    
    library.add_action(Action(
        name="break_down_goal",
        description="Decompose high-level goal into sub-goals",
        preconditions=WorldState({}),
        effects=WorldState({'goals_decomposed': True}),
        cost=6.0,
        cognitive_load=0.6,
        duration_minutes=25
    ))
    
    return library
