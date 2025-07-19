"""
Goal Manager - Hierarchical goal tracking and prioritization

This module manages the agent's goal hierarchy, tracking progress, priorities,
and dependencies between goals. It supports both long-term strategic goals
and short-term tactical objectives.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json

class GoalStatus(Enum):
    """Status of a goal"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GoalPriority(Enum):
    """Priority levels for goals"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class Goal:
    """
    Represents a goal with hierarchical structure and progress tracking
    
    Attributes:
        id: Unique identifier
        title: Short descriptive title
        description: Detailed description
        priority: Priority level
        status: Current status
        parent_id: Parent goal ID (for hierarchical goals)
        dependencies: Set of goal IDs this goal depends on
        created_at: Creation timestamp
        updated_at: Last update timestamp
        target_date: Target completion date
        progress: Progress percentage (0.0 to 1.0)
        context: Additional context and metadata
        success_criteria: Criteria for determining completion
        resources_needed: List of required resources
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.CREATED
    parent_id: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    target_date: Optional[datetime] = None
    progress: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate goal parameters"""
        if not self.title:
            raise ValueError("Goal title cannot be empty")
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError("Progress must be between 0.0 and 1.0")
    
    def update_progress(self, progress: float, update_reason: str = "") -> None:
        """Update goal progress with validation"""
        if not (0.0 <= progress <= 1.0):
            raise ValueError("Progress must be between 0.0 and 1.0")
        
        old_progress = self.progress
        self.progress = progress
        self.updated_at = datetime.now()
        
        # Auto-update status based on progress
        if progress >= 1.0 and self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.COMPLETED
        elif progress > 0.0 and self.status == GoalStatus.CREATED:
            self.status = GoalStatus.ACTIVE
        
        # Log progress update
        self.context.setdefault('progress_history', []).append({
            'timestamp': self.updated_at.isoformat(),
            'old_progress': old_progress,
            'new_progress': progress,
            'reason': update_reason
        })
    
    def is_overdue(self) -> bool:
        """Check if goal is overdue"""
        if not self.target_date:
            return False
        return datetime.now() > self.target_date and self.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED]
    
    def days_until_target(self) -> Optional[int]:
        """Get days until target date"""
        if not self.target_date:
            return None
        delta = self.target_date - datetime.now()
        return delta.days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'status': self.status.value,
            'parent_id': self.parent_id,
            'dependencies': list(self.dependencies),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'progress': self.progress,
            'context': self.context,
            'success_criteria': self.success_criteria,
            'resources_needed': self.resources_needed
        }

class GoalManager:
    """
    Manages hierarchical goals with priority tracking and progress monitoring
    """
    
    def __init__(self, max_active_goals: int = 10):
        """
        Initialize goal manager
        
        Args:
            max_active_goals: Maximum number of active goals to track
        """
        self.goals: Dict[str, Goal] = {}
        self.max_active_goals = max_active_goals
        self.goal_hierarchy: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        target_date: Optional[datetime] = None,
        success_criteria: Optional[List[str]] = None,
        resources_needed: Optional[List[str]] = None
    ) -> str:
        """
        Create a new goal
        
        Args:
            title: Goal title
            description: Detailed description
            priority: Priority level
            parent_id: Parent goal ID for hierarchical goals
            target_date: Target completion date
            success_criteria: List of success criteria
            resources_needed: List of required resources
            
        Returns:
            Goal ID
            
        Raises:
            ValueError: If parameters are invalid
        """
        if parent_id and parent_id not in self.goals:
            raise ValueError(f"Parent goal {parent_id} not found")
        
        goal = Goal(
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            target_date=target_date,
            success_criteria=success_criteria or [],
            resources_needed=resources_needed or []
        )
        
        self.goals[goal.id] = goal
        
        # Update hierarchy
        if parent_id:
            self.goal_hierarchy.setdefault(parent_id, []).append(goal.id)
        
        return goal.id
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID"""
        return self.goals.get(goal_id)
    
    def update_goal_progress(self, goal_id: str, progress: float, reason: str = "") -> bool:
        """Update progress for a goal"""
        goal = self.goals.get(goal_id)
        if not goal:
            return False
        
        goal.update_progress(progress, reason)
        
        # Update parent goal progress if applicable
        if goal.parent_id:
            self._update_parent_progress(goal.parent_id)
        
        return True
    
    def _update_parent_progress(self, parent_id: str) -> None:
        """Update parent goal progress based on child goals"""
        parent_goal = self.goals.get(parent_id)
        if not parent_goal:
            return
        
        child_ids = self.goal_hierarchy.get(parent_id, [])
        if not child_ids:
            return
        
        # Calculate average progress of child goals
        total_progress = 0.0
        active_children = 0
        
        for child_id in child_ids:
            child_goal = self.goals.get(child_id)
            if child_goal and child_goal.status in [GoalStatus.ACTIVE, GoalStatus.COMPLETED]:
                total_progress += child_goal.progress
                active_children += 1
        
        if active_children > 0:
            avg_progress = total_progress / active_children
            parent_goal.update_progress(avg_progress, f"Updated based on {active_children} child goals")
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by priority"""
        active_goals = [
            goal for goal in self.goals.values()
            if goal.status == GoalStatus.ACTIVE
        ]
        return sorted(active_goals, key=lambda g: g.priority.value, reverse=True)
    
    def get_overdue_goals(self) -> List[Goal]:
        """Get all overdue goals"""
        return [goal for goal in self.goals.values() if goal.is_overdue()]
    
    def get_child_goals(self, parent_id: str) -> List[Goal]:
        """Get all child goals for a parent"""
        child_ids = self.goal_hierarchy.get(parent_id, [])
        return [self.goals[child_id] for child_id in child_ids if child_id in self.goals]
    
    def get_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """Get goal hierarchy tree starting from a specific goal"""
        goal = self.goals.get(goal_id)
        if not goal:
            return {}
        
        result = goal.to_dict()
        
        # Add children
        child_ids = self.goal_hierarchy.get(goal_id, [])
        if child_ids:
            result['children'] = [
                self.get_goal_hierarchy(child_id) 
                for child_id in child_ids
            ]
        
        return result
    
    def prioritize_goals(self) -> List[Goal]:
        """
        Get goals prioritized by multiple criteria:
        1. Priority level
        2. Target date proximity
        3. Dependencies
        """
        active_goals = self.get_active_goals()
        
        def priority_score(goal: Goal) -> Tuple[int, int, int]:
            # Higher priority value = more important
            priority_val = goal.priority.value
            
            # Days until target (closer = higher priority)
            days_until = goal.days_until_target()
            target_urgency = 1000 - (days_until if days_until is not None else 1000)
            
            # Number of goals depending on this one
            dependents = len([
                g for g in self.goals.values()
                if goal.id in g.dependencies
            ])
            
            return (priority_val, target_urgency, dependents)
        
        return sorted(active_goals, key=priority_score, reverse=True)
    
    def can_start_goal(self, goal_id: str) -> Tuple[bool, List[str]]:
        """
        Check if a goal can be started (all dependencies met)
        
        Returns:
            (can_start, list_of_blocking_dependencies)
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return False, ["Goal not found"]
        
        blocking_deps = []
        for dep_id in goal.dependencies:
            dep_goal = self.goals.get(dep_id)
            if not dep_goal:
                blocking_deps.append(f"Dependency {dep_id} not found")
            elif dep_goal.status != GoalStatus.COMPLETED:
                blocking_deps.append(f"Dependency '{dep_goal.title}' not completed")
        
        return len(blocking_deps) == 0, blocking_deps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get goal management statistics"""
        status_counts = {}
        priority_counts = {}
        
        for goal in self.goals.values():
            status_counts[goal.status.value] = status_counts.get(goal.status.value, 0) + 1
            priority_counts[goal.priority.value] = priority_counts.get(goal.priority.value, 0) + 1
        
        active_goals = self.get_active_goals()
        overdue_goals = self.get_overdue_goals()
        
        goals_near_target = []
        for g in active_goals:
            days_until = g.days_until_target()
            if days_until is not None and days_until <= 7:
                goals_near_target.append(g)
        
        return {
            'total_goals': len(self.goals),
            'active_goals': len(active_goals),
            'overdue_goals': len(overdue_goals),
            'status_distribution': status_counts,
            'priority_distribution': priority_counts,
            'average_progress': sum(g.progress for g in active_goals) / len(active_goals) if active_goals else 0.0,
            'goals_near_target': len(goals_near_target)
        }
    
    def export_goals(self) -> str:
        """Export all goals to JSON string"""
        export_data = {
            'goals': [goal.to_dict() for goal in self.goals.values()],
            'hierarchy': self.goal_hierarchy,
            'exported_at': datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2)
    
    def import_goals(self, json_data: str) -> int:
        """
        Import goals from JSON string
        
        Returns:
            Number of goals imported
        """
        try:
            data = json.loads(json_data)
            imported_count = 0
            
            for goal_data in data.get('goals', []):
                goal = Goal(
                    id=goal_data['id'],
                    title=goal_data['title'],
                    description=goal_data['description'],
                    priority=GoalPriority(goal_data['priority']),
                    status=GoalStatus(goal_data['status']),
                    parent_id=goal_data.get('parent_id'),
                    dependencies=set(goal_data.get('dependencies', [])),
                    created_at=datetime.fromisoformat(goal_data['created_at']),
                    updated_at=datetime.fromisoformat(goal_data['updated_at']),
                    target_date=datetime.fromisoformat(goal_data['target_date']) if goal_data.get('target_date') else None,
                    progress=goal_data['progress'],
                    context=goal_data.get('context', {}),
                    success_criteria=goal_data.get('success_criteria', []),
                    resources_needed=goal_data.get('resources_needed', [])
                )
                self.goals[goal.id] = goal
                imported_count += 1
            
            self.goal_hierarchy = data.get('hierarchy', {})
            return imported_count
            
        except Exception as e:
            raise ValueError(f"Failed to import goals: {e}")
