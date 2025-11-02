"""
HTN Manager - Hierarchical Task Network Planning.

Manages goal decomposition using HTN planning:
- Recursive decomposition of compound goals
- Method selection and application
- Subtask generation and ordering
- Integration with GOAP for primitive goals

Algorithm:
    1. Start with high-level compound goal
    2. Find applicable methods for decomposition
    3. Select best method based on current state
    4. Generate subtasks from method
    5. Recursively decompose compound subtasks
    6. Continue until all goals are primitive
    7. Return complete task network
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import uuid
import logging

from .goal_taxonomy import Goal, GoalType, GoalStatus
from .decomposition import Method, create_default_methods

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """
    Result of HTN decomposition.
    
    Attributes:
        success: Whether decomposition succeeded
        goals: All generated goals (root + subtasks)
        primitive_goals: Only primitive (executable) goals
        compound_goals: Goals that were further decomposed
        ordering: Execution order constraints
        depth: Maximum depth of goal hierarchy
        error: Error message if decomposition failed
    """
    
    success: bool
    goals: List[Goal] = field(default_factory=list)
    primitive_goals: List[Goal] = field(default_factory=list)
    compound_goals: List[Goal] = field(default_factory=list)
    ordering: Dict[str, List[str]] = field(default_factory=dict)  # goal_id -> depends_on
    depth: int = 0
    error: Optional[str] = None
    
    def get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by its ID."""
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        return None
    
    def get_subtasks(self, parent_id: str) -> List[Goal]:
        """Get all subtasks of a parent goal."""
        return [g for g in self.goals if g.parent_id == parent_id]
    
    def get_ready_goals(self, completed_ids: set) -> List[Goal]:
        """
        Get goals that are ready to execute.
        
        A goal is ready if:
        - It's primitive
        - It's pending
        - All dependencies are completed
        
        Args:
            completed_ids: Set of completed goal IDs
            
        Returns:
            List of ready goals
        """
        ready = []
        
        for goal in self.primitive_goals:
            if goal.status != GoalStatus.PENDING:
                continue
            
            # Check if all dependencies are satisfied
            all_deps_met = all(
                dep_id in completed_ids
                for dep_id in goal.dependencies
            )
            
            if all_deps_met:
                ready.append(goal)
        
        return ready


class HTNManager:
    """
    Hierarchical Task Network manager.
    
    Decomposes high-level goals into executable subtasks using methods.
    """
    
    def __init__(
        self,
        methods: Optional[List[Method]] = None,
        max_depth: int = 10
    ):
        """
        Initialize HTN manager.
        
        Args:
            methods: Decomposition methods (uses defaults if None)
            max_depth: Maximum recursion depth
        """
        self.methods = methods if methods else create_default_methods()
        self.max_depth = max_depth
        self._goal_counter = 0
    
    def _generate_goal_id(self) -> str:
        """Generate unique goal ID."""
        self._goal_counter += 1
        return f"goal_{self._goal_counter}_{uuid.uuid4().hex[:8]}"
    
    def decompose(
        self,
        goal: Goal,
        current_state: Dict[str, Any],
        depth: int = 0
    ) -> DecompositionResult:
        """
        Recursively decompose a goal into primitive subtasks.
        
        Args:
            goal: Goal to decompose (may be primitive or compound)
            current_state: Current world state
            depth: Current recursion depth
            
        Returns:
            DecompositionResult with all goals and ordering
        """
        # Check depth limit
        if depth > self.max_depth:
            return DecompositionResult(
                success=False,
                error=f"Maximum decomposition depth ({self.max_depth}) exceeded"
            )
        
        all_goals = [goal]
        primitive_goals = []
        compound_goals = []
        ordering = {}
        max_depth_seen = depth
        
        # If primitive, no decomposition needed
        if goal.is_primitive():
            primitive_goals.append(goal)
            return DecompositionResult(
                success=True,
                goals=all_goals,
                primitive_goals=primitive_goals,
                compound_goals=compound_goals,
                ordering=ordering,
                depth=max_depth_seen
            )
        
        # Compound goal - find applicable method
        applicable_methods = [
            m for m in self.methods
            if m.is_applicable(goal, current_state)
        ]
        
        if not applicable_methods:
            return DecompositionResult(
                success=False,
                error=f"No applicable methods found for goal: {goal.description}"
            )
        
        # Select best method (highest priority)
        method = max(applicable_methods, key=lambda m: m.priority)
        
        logger.info(
            f"Decomposing '{goal.description}' using method '{method.name}' "
            f"(depth={depth})"
        )
        
        # Generate subtasks
        subtasks = method.decompose(goal, current_state, self._generate_goal_id)
        
        # Update parent goal with subtask IDs
        for subtask in subtasks:
            goal.add_subtask(subtask.id)
        
        compound_goals.append(goal)
        
        # Recursively decompose subtasks
        for subtask in subtasks:
            result = self.decompose(subtask, current_state, depth + 1)
            
            if not result.success:
                return result  # Propagate failure
            
            # Merge results
            all_goals.extend(result.goals)
            primitive_goals.extend(result.primitive_goals)
            compound_goals.extend(result.compound_goals)
            ordering.update(result.ordering)
            max_depth_seen = max(max_depth_seen, result.depth)
        
        # Record ordering constraints
        for subtask in subtasks:
            if subtask.dependencies:
                ordering[subtask.id] = subtask.dependencies.copy()
        
        return DecompositionResult(
            success=True,
            goals=all_goals,
            primitive_goals=primitive_goals,
            compound_goals=compound_goals,
            ordering=ordering,
            depth=max_depth_seen
        )
    
    def add_method(self, method: Method) -> None:
        """Add a decomposition method to the library."""
        self.methods.append(method)
    
    def remove_method(self, method_name: str) -> bool:
        """
        Remove a method by name.
        
        Args:
            method_name: Name of method to remove
            
        Returns:
            True if method was found and removed
        """
        initial_count = len(self.methods)
        self.methods = [m for m in self.methods if m.name != method_name]
        return len(self.methods) < initial_count
    
    def get_method_by_name(self, method_name: str) -> Optional[Method]:
        """Get a method by name."""
        for method in self.methods:
            if method.name == method_name:
                return method
        return None
    
    def find_applicable_methods(
        self,
        goal: Goal,
        current_state: Dict[str, Any]
    ) -> List[Method]:
        """
        Find all methods applicable to a goal.
        
        Args:
            goal: Goal to find methods for
            current_state: Current world state
            
        Returns:
            List of applicable methods (sorted by priority)
        """
        applicable = [
            m for m in self.methods
            if m.is_applicable(goal, current_state)
        ]
        
        # Sort by priority (highest first)
        applicable.sort(key=lambda m: m.priority, reverse=True)
        
        return applicable
