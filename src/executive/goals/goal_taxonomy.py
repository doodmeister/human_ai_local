"""
Goal Taxonomy for HTN Planning.

Defines goal types, goal structure, and goal status tracking.

Key Concepts:
- Primitive goals: Concrete, executable goals (map to GOAP actions)
- Compound goals: High-level goals that decompose into subtasks
- Goal hierarchy: Parent-child relationships
- Goal status: Pending, active, completed, failed, blocked
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class GoalType(Enum):
    """Type of goal in the hierarchy."""
    PRIMITIVE = "primitive"  # Concrete, executable goal
    COMPOUND = "compound"    # Decomposes into subtasks


class GoalStatus(Enum):
    """Current status of a goal."""
    PENDING = "pending"      # Not yet started
    ACTIVE = "active"        # Currently being worked on
    COMPLETED = "completed"  # Successfully achieved
    FAILED = "failed"        # Could not be achieved
    BLOCKED = "blocked"      # Waiting on dependencies


class GoalPriority(Enum):
    """Priority level for goals."""
    CRITICAL = 10
    HIGH = 8
    MEDIUM = 5
    LOW = 3
    OPTIONAL = 1


@dataclass
class Goal:
    """
    Represents a goal in the HTN hierarchy.
    
    Goals can be:
    - Primitive: Map directly to executable actions (via GOAP)
    - Compound: Decompose into subtasks using methods
    
    Attributes:
        id: Unique goal identifier
        description: Human-readable goal description
        goal_type: Primitive or compound
        status: Current goal status
        priority: Goal priority (affects scheduling)
        parent_id: Parent goal (if this is a subtask)
        subtask_ids: Child goals (if compound and decomposed)
        preconditions: Conditions that must be true to start
        postconditions: Expected state after completion
        deadline: Optional deadline for completion
        dependencies: Other goals that must complete first
        metadata: Additional goal-specific data
        created_at: When goal was created
        completed_at: When goal was completed (if completed)
    """
    
    id: str
    description: str
    goal_type: GoalType
    status: GoalStatus = GoalStatus.PENDING
    priority: int = GoalPriority.MEDIUM.value
    
    # Hierarchy
    parent_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    
    # Planning
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def is_primitive(self) -> bool:
        """Check if goal is primitive (executable)."""
        return self.goal_type == GoalType.PRIMITIVE
    
    def is_compound(self) -> bool:
        """Check if goal is compound (needs decomposition)."""
        return self.goal_type == GoalType.COMPOUND
    
    def is_decomposed(self) -> bool:
        """Check if compound goal has been decomposed into subtasks."""
        return self.is_compound() and len(self.subtask_ids) > 0
    
    def is_leaf(self) -> bool:
        """Check if goal is a leaf node (no subtasks)."""
        return len(self.subtask_ids) == 0
    
    def is_root(self) -> bool:
        """Check if goal is a root node (no parent)."""
        return self.parent_id is None
    
    def can_start(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if goal can start given current state.
        
        Args:
            current_state: Current world state
            
        Returns:
            True if all preconditions are met
        """
        if self.status != GoalStatus.PENDING:
            return False
        
        # Check preconditions
        for key, required_value in self.preconditions.items():
            if key not in current_state:
                return False
            if current_state[key] != required_value:
                return False
        
        return True
    
    def mark_started(self) -> None:
        """Mark goal as active."""
        self.status = GoalStatus.ACTIVE
    
    def mark_completed(self) -> None:
        """Mark goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self) -> None:
        """Mark goal as failed."""
        self.status = GoalStatus.FAILED
    
    def mark_blocked(self) -> None:
        """Mark goal as blocked."""
        self.status = GoalStatus.BLOCKED
    
    def add_subtask(self, subtask_id: str) -> None:
        """Add a subtask to this goal."""
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)
    
    def remove_subtask(self, subtask_id: str) -> None:
        """Remove a subtask from this goal."""
        if subtask_id in self.subtask_ids:
            self.subtask_ids.remove(subtask_id)
    
    def add_dependency(self, goal_id: str) -> None:
        """Add a goal dependency."""
        if goal_id not in self.dependencies:
            self.dependencies.append(goal_id)
    
    def __repr__(self) -> str:
        return (
            f"Goal(id={self.id}, type={self.goal_type.value}, "
            f"status={self.status.value}, priority={self.priority})"
        )
