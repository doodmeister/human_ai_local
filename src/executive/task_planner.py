"""
Task Planner - Goal decomposition and task sequencing

This module breaks down high-level goals into executable tasks, manages
task dependencies, and creates execution plans optimized for cognitive
resources and constraints.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .goal_manager import Goal, GoalManager, GoalPriority

class TaskType(Enum):
    """Types of tasks"""
    ANALYSIS = "analysis"
    CREATION = "creation"
    COMMUNICATION = "communication"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

@dataclass
class Task:
    """
    Individual task with execution details and dependencies
    
    Attributes:
        id: Unique identifier
        title: Task title
        description: Detailed description
        task_type: Type of task
        status: Current execution status
        goal_id: Parent goal ID
        dependencies: Set of task IDs this task depends on
        estimated_duration: Estimated time to complete (minutes)
        actual_duration: Actual time spent (minutes)
        cognitive_load: Estimated cognitive effort (0.0 to 1.0)
        resources_needed: Required resources
        context: Additional context and metadata
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        instructions: Step-by-step instructions
        success_criteria: Criteria for task completion
        priority_score: Calculated priority score
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    task_type: TaskType = TaskType.EXECUTION
    status: TaskStatus = TaskStatus.PENDING
    goal_id: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: int = 30  # minutes
    actual_duration: int = 0
    cognitive_load: float = 0.5
    resources_needed: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    instructions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    priority_score: float = 0.5
    
    def __post_init__(self):
        """Validate task parameters"""
        if not self.title:
            raise ValueError("Task title cannot be empty")
        if not (0.0 <= self.cognitive_load <= 1.0):
            raise ValueError("Cognitive load must be between 0.0 and 1.0")
        if self.estimated_duration <= 0:
            raise ValueError("Estimated duration must be positive")
    
    def start_task(self) -> None:
        """Mark task as started"""
        if self.status != TaskStatus.READY:
            raise ValueError(f"Cannot start task in status {self.status}")
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete_task(self) -> None:
        """Mark task as completed"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task in status {self.status}")
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if self.started_at:
            self.actual_duration = int((self.completed_at - self.started_at).total_seconds() / 60)
    
    def fail_task(self, reason: str = "") -> None:
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.context['failure_reason'] = reason
    
    def is_overdue(self) -> bool:
        """Check if task is overdue based on estimated duration"""
        if self.status != TaskStatus.IN_PROGRESS or not self.started_at:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds() / 60
        return elapsed > self.estimated_duration * 1.5  # 50% buffer
    
    def time_remaining(self) -> Optional[int]:
        """Get estimated time remaining in minutes"""
        if self.status != TaskStatus.IN_PROGRESS or not self.started_at:
            return None
        elapsed = (datetime.now() - self.started_at).total_seconds() / 60
        return max(0, self.estimated_duration - int(elapsed))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type.value,
            'status': self.status.value,
            'goal_id': self.goal_id,
            'dependencies': list(self.dependencies),
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'cognitive_load': self.cognitive_load,
            'resources_needed': self.resources_needed,
            'context': self.context,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'instructions': self.instructions,
            'success_criteria': self.success_criteria,
            'priority_score': self.priority_score
        }

class TaskPlanner:
    """
    Plans and manages tasks derived from goals
    """
    
    def __init__(self, goal_manager: GoalManager):
        """
        Initialize task planner
        
        Args:
            goal_manager: Goal manager instance
        """
        self.goal_manager = goal_manager
        self.tasks: Dict[str, Task] = {}
        self.task_templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default task templates for common goal types"""
        self.task_templates = {
            'research_goal': {
                'tasks': [
                    {'title': 'Define research questions', 'type': TaskType.ANALYSIS, 'duration': 30},
                    {'title': 'Gather information', 'type': TaskType.LEARNING, 'duration': 60},
                    {'title': 'Analyze findings', 'type': TaskType.ANALYSIS, 'duration': 45},
                    {'title': 'Create summary', 'type': TaskType.CREATION, 'duration': 30}
                ]
            },
            'project_goal': {
                'tasks': [
                    {'title': 'Project planning', 'type': TaskType.PLANNING, 'duration': 45},
                    {'title': 'Setup and initialization', 'type': TaskType.EXECUTION, 'duration': 30},
                    {'title': 'Implementation', 'type': TaskType.EXECUTION, 'duration': 120},
                    {'title': 'Testing and validation', 'type': TaskType.VERIFICATION, 'duration': 60},
                    {'title': 'Documentation', 'type': TaskType.CREATION, 'duration': 30}
                ]
            },
            'communication_goal': {
                'tasks': [
                    {'title': 'Prepare message', 'type': TaskType.CREATION, 'duration': 20},
                    {'title': 'Send communication', 'type': TaskType.COMMUNICATION, 'duration': 10},
                    {'title': 'Follow up', 'type': TaskType.COMMUNICATION, 'duration': 15}
                ]
            }
        }
    
    def decompose_goal(self, goal_id: str, template_name: Optional[str] = None) -> List[str]:
        """
        Decompose a goal into tasks
        
        Args:
            goal_id: Goal to decompose
            template_name: Optional template to use
            
        Returns:
            List of created task IDs
        """
        goal = self.goal_manager.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")
        
        task_ids = []
        
        # Use template if provided
        if template_name and template_name in self.task_templates:
            template = self.task_templates[template_name]
            for i, task_def in enumerate(template['tasks']):
                task_id = self.create_task(
                    title=task_def['title'],
                    description=f"Task {i+1} for goal: {goal.title}",
                    task_type=task_def['type'],
                    goal_id=goal_id,
                    estimated_duration=task_def['duration']
                )
                task_ids.append(task_id)
                
                # Add dependency on previous task
                if i > 0:
                    self.tasks[task_id].dependencies.add(task_ids[i-1])
        
        # Auto-decompose based on goal characteristics
        else:
            task_ids = self._auto_decompose_goal(goal)
        
        return task_ids
    
    def _auto_decompose_goal(self, goal: Goal) -> List[str]:
        """
        Automatically decompose a goal into tasks based on its characteristics
        """
        task_ids = []
        
        # Analysis task for complex goals
        if len(goal.description) > 100 or len(goal.success_criteria) > 3:
            task_id = self.create_task(
                title=f"Analyze requirements for: {goal.title}",
                description=f"Break down and analyze the requirements for {goal.title}",
                task_type=TaskType.ANALYSIS,
                goal_id=goal.id,
                estimated_duration=30
            )
            task_ids.append(task_id)
        
        # Main execution task
        exec_task_id = self.create_task(
            title=f"Execute: {goal.title}",
            description=goal.description,
            task_type=TaskType.EXECUTION,
            goal_id=goal.id,
            estimated_duration=60,
            success_criteria=goal.success_criteria
        )
        task_ids.append(exec_task_id)
        
        # Add dependency on analysis if it exists
        if len(task_ids) > 1:
            self.tasks[exec_task_id].dependencies.add(task_ids[0])
        
        # Verification task for goals with success criteria
        if goal.success_criteria:
            verify_task_id = self.create_task(
                title=f"Verify completion: {goal.title}",
                description=f"Verify that {goal.title} has been completed successfully",
                task_type=TaskType.VERIFICATION,
                goal_id=goal.id,
                estimated_duration=15
            )
            task_ids.append(verify_task_id)
            self.tasks[verify_task_id].dependencies.add(exec_task_id)
        
        return task_ids
    
    def create_task(
        self,
        title: str,
        description: str = "",
        task_type: TaskType = TaskType.EXECUTION,
        goal_id: Optional[str] = None,
        estimated_duration: int = 30,
        cognitive_load: float = 0.5,
        resources_needed: Optional[List[str]] = None,
        instructions: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None
    ) -> str:
        """
        Create a new task
        
        Returns:
            Task ID
        """
        task = Task(
            title=title,
            description=description,
            task_type=task_type,
            goal_id=goal_id,
            estimated_duration=estimated_duration,
            cognitive_load=cognitive_load,
            resources_needed=resources_needed or [],
            instructions=instructions or [],
            success_criteria=success_criteria or []
        )
        
        self.tasks[task.id] = task
        self._update_task_priority(task.id)
        
        return task.id
    
    def _update_task_priority(self, task_id: str) -> None:
        """Update task priority based on goal priority and dependencies"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Base priority from goal
        goal_priority = 0.5
        if task.goal_id:
            goal = self.goal_manager.get_goal(task.goal_id)
            if goal:
                goal_priority = goal.priority.value / 5.0  # Normalize to 0-1
        
        # Boost priority for blocking tasks (tasks that others depend on)
        blocking_boost = len([
            t for t in self.tasks.values()
            if task.id in t.dependencies
        ]) * 0.1
        
        # Reduce priority for tasks with many dependencies
        dependency_penalty = len(task.dependencies) * 0.05
        
        # Urgency based on goal target date
        urgency_boost = 0.0
        if task.goal_id:
            goal = self.goal_manager.get_goal(task.goal_id)
            if goal and goal.target_date:
                days_until = (goal.target_date - datetime.now()).days
                if days_until <= 7:
                    urgency_boost = 0.3
                elif days_until <= 30:
                    urgency_boost = 0.1
        
        task.priority_score = min(1.0, goal_priority + blocking_boost - dependency_penalty + urgency_boost)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)"""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                can_start = True
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        can_start = False
                        break
                
                if can_start:
                    task.status = TaskStatus.READY
                    ready_tasks.append(task)
        
        # Sort by priority score
        return sorted(ready_tasks, key=lambda t: t.priority_score, reverse=True)
    
    def get_execution_plan(self, max_duration: int = 480) -> List[Task]:
        """
        Create an execution plan for tasks within time constraints
        
        Args:
            max_duration: Maximum duration in minutes
            
        Returns:
            Ordered list of tasks to execute
        """
        ready_tasks = self.get_ready_tasks()
        
        # Simple greedy scheduling based on priority and duration
        plan = []
        remaining_time = max_duration
        
        for task in ready_tasks:
            if task.estimated_duration <= remaining_time:
                plan.append(task)
                remaining_time -= task.estimated_duration
        
        return plan
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task management statistics"""
        status_counts = {}
        type_counts = {}
        total_estimated = 0
        total_actual = 0
        
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
            type_counts[task.task_type.value] = type_counts.get(task.task_type.value, 0) + 1
            total_estimated += task.estimated_duration
            total_actual += task.actual_duration
        
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        overdue_tasks = [t for t in self.tasks.values() if t.is_overdue()]
        
        return {
            'total_tasks': len(self.tasks),
            'ready_tasks': len(self.get_ready_tasks()),
            'overdue_tasks': len(overdue_tasks),
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'total_estimated_time': total_estimated,
            'total_actual_time': total_actual,
            'estimation_accuracy': total_actual / total_estimated if total_estimated > 0 else 0.0,
            'completion_rate': len(completed_tasks) / len(self.tasks) if self.tasks else 0.0,
            'average_task_duration': total_actual / len(completed_tasks) if completed_tasks else 0.0
        }
    
    def suggest_next_task(self, available_time: int = 60, max_cognitive_load: float = 0.8) -> Optional[Task]:
        """
        Suggest the next best task to work on
        
        Args:
            available_time: Available time in minutes
            max_cognitive_load: Maximum cognitive load willing to handle
            
        Returns:
            Best task to work on or None
        """
        ready_tasks = self.get_ready_tasks()
        
        # Filter by constraints
        suitable_tasks = [
            task for task in ready_tasks
            if task.estimated_duration <= available_time
            and task.cognitive_load <= max_cognitive_load
        ]
        
        if not suitable_tasks:
            return None
        
        # Return highest priority task
        return max(suitable_tasks, key=lambda t: t.priority_score)


# ============================================================================
# GOAP Planning Integration (Phase 2)
# ============================================================================

class GOAPTaskPlannerAdapter:
    """
    Adapter that integrates GOAP (Goal-Oriented Action Planning) with the
    legacy TaskPlanner system.
    
    This adapter:
    - Converts Goals → GOAP WorldState representations
    - Uses GOAP planner to find optimal action sequences
    - Converts GOAP Plans → Task sequences for execution
    - Provides feature flags for gradual rollout
    - Falls back to template-based planning on errors
    
    Phase 2 Integration Pattern (similar to Phase 1 DecisionEngine)
    """
    
    def __init__(self, task_planner: TaskPlanner):
        """
        Initialize GOAP adapter with legacy task planner.
        
        Args:
            task_planner: Legacy TaskPlanner instance for fallback
        """
        self.task_planner = task_planner
        self.goap_available = False
        self.goap_planner = None
        self.action_library = None
        self.feature_flags = None
        
        # Try to import GOAP components
        try:
            from .planning import (
                WorldState,
                GOAPPlanner,
                create_default_action_library
            )
            from .planning.heuristics import get_heuristic
            from .decision.base import get_feature_flags
            
            self.WorldState = WorldState
            self.action_library = create_default_action_library()
            self.goap_planner = GOAPPlanner(
                action_library=self.action_library,
                heuristic=get_heuristic('weighted_goal_distance')
            )
            self.feature_flags = get_feature_flags()
            self.goap_available = True
            
        except ImportError:
            # GOAP not available, will use legacy planning
            self.goap_available = False
    
    def decompose_goal_with_goap(
        self,
        goal_id: str,
        current_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Decompose a goal into tasks using GOAP planning.
        
        Args:
            goal_id: Goal to decompose
            current_context: Current world state context (optional)
            
        Returns:
            List of created task IDs
            
        Raises:
            ValueError: If goal not found or GOAP planning fails
        """
        # Check if GOAP is enabled
        if not self._should_use_goap():
            return self.task_planner.decompose_goal(goal_id)
        
        # Get goal from manager
        goal = self.task_planner.goal_manager.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")
        
        try:
            # Convert goal to GOAP world states
            initial_state, goal_state = self._goal_to_world_states(goal, current_context)
            
            # Plan with GOAP
            plan = self.goap_planner.plan(
                initial_state=initial_state,
                goal_state=goal_state,
                max_iterations=1000
            )
            
            if plan is None:
                # No plan found, fallback to legacy
                if self.feature_flags.fallback_to_legacy:
                    return self.task_planner.decompose_goal(goal_id)
                else:
                    raise ValueError(f"GOAP planning failed for goal {goal_id}")
            
            # Convert GOAP plan to tasks
            task_ids = self._plan_to_tasks(plan, goal)
            
            return task_ids
            
        except Exception:
            # Error in GOAP planning, fallback if enabled
            if self.feature_flags and self.feature_flags.fallback_to_legacy:
                return self.task_planner.decompose_goal(goal_id)
            else:
                raise
    
    def _should_use_goap(self) -> bool:
        """Check if GOAP planning should be used."""
        return (
            self.goap_available
            and self.feature_flags is not None
            and self.feature_flags.use_goap_planning
        )
    
    def _goal_to_world_states(
        self,
        goal: Goal,
        current_context: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Convert a Goal to GOAP WorldStates (initial and goal).
        
        Args:
            goal: Goal to convert
            current_context: Current state context
            
        Returns:
            Tuple of (initial_state, goal_state)
        """
        # Build initial state from current context
        initial_dict = current_context or {}
        
        # Add default initial state if empty
        if not initial_dict:
            initial_dict = {
                'has_goal_defined': True,
                'goal_id': goal.id,
                'goal_priority': goal.priority.value
            }
        
        initial_state = self.WorldState(initial_dict)
        
        # Build goal state from goal's success criteria and context
        goal_dict = {}
        
        # Map goal characteristics to GOAP goal states
        if goal.success_criteria:
            # Use success criteria to define goal state
            for i, criterion in enumerate(goal.success_criteria):
                # Convert success criteria to state keys
                key = self._criterion_to_state_key(criterion, i)
                goal_dict[key] = True
        else:
            # Generic goal completion state
            goal_dict['goal_completed'] = True
            goal_dict['goal_verified'] = True
        
        # Add priority-based flags for weighted heuristic
        if goal.priority == GoalPriority.CRITICAL:
            goal_dict['critical_goal_achieved'] = True
        elif goal.priority == GoalPriority.URGENT or goal.priority == GoalPriority.HIGH:
            goal_dict['important_goal_achieved'] = True
        else:
            goal_dict['optional_goal_achieved'] = True
        
        goal_state = self.WorldState(goal_dict)
        
        return initial_state, goal_state
    
    def _criterion_to_state_key(self, criterion: str, index: int) -> str:
        """
        Convert a success criterion string to a WorldState key.
        
        Args:
            criterion: Success criterion text
            index: Index in criteria list
            
        Returns:
            State key string
        """
        # Simple heuristic: extract key words and create state key
        # Examples:
        # "Data gathered and analyzed" → "has_data_analyzed"
        # "Report completed" → "has_report_completed"
        
        words = criterion.lower().split()
        
        # Remove common articles/connectors
        filtered = [w for w in words if w not in ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are']]
        
        if filtered:
            # Use first 2-3 significant words
            key_words = filtered[:min(3, len(filtered))]
            key = 'has_' + '_'.join(key_words)
        else:
            # Fallback to generic key
            key = f'criterion_{index}_met'
        
        return key
    
    def _plan_to_tasks(self, plan, goal: Goal) -> List[str]:
        """
        Convert a GOAP Plan to a sequence of Tasks.
        
        Args:
            plan: GOAP Plan object
            goal: Original Goal
            
        Returns:
            List of created task IDs
        """
        task_ids = []
        
        for step in plan.steps:
            action = step.action
            
            # Map GOAP action to TaskType
            task_type = self._action_to_task_type(action.name)
            
            # Create task from GOAP action
            task_id = self.task_planner.create_task(
                title=self._action_to_task_title(action.name, goal.title),
                description=f"GOAP action: {action.name}\nFor goal: {goal.description}",
                task_type=task_type,
                goal_id=goal.id,
                estimated_duration=int(action.duration / 60) if action.duration > 0 else 30,
                cognitive_load=action.cognitive_load / 10.0,  # Normalize to 0-1
                resources_needed=action.resources,
                instructions=[f"Execute GOAP action: {action.name}"],
                success_criteria=[f"Action {action.name} completed successfully"]
            )
            
            # Add dependencies (previous task in sequence)
            if task_ids:
                self.task_planner.tasks[task_id].dependencies.add(task_ids[-1])
            
            task_ids.append(task_id)
        
        return task_ids
    
    def _action_to_task_type(self, action_name: str) -> TaskType:
        """Map GOAP action name to TaskType."""
        action_lower = action_name.lower()
        
        # Check for planning first (before create)
        if 'plan' in action_lower or 'break_down' in action_lower:
            return TaskType.PLANNING
        elif 'analyze' in action_lower or 'review' in action_lower:
            return TaskType.ANALYSIS
        elif 'create' in action_lower or 'draft' in action_lower or 'document' in action_lower:
            return TaskType.CREATION
        elif 'send' in action_lower or 'notify' in action_lower or 'schedule' in action_lower:
            return TaskType.COMMUNICATION
        elif 'test' in action_lower or 'verify' in action_lower:
            return TaskType.VERIFICATION
        elif 'gather' in action_lower or 'collect' in action_lower:
            return TaskType.LEARNING
        else:
            return TaskType.EXECUTION
    
    def _action_to_task_title(self, action_name: str, goal_title: str) -> str:
        """Create a human-readable task title from action name."""
        # Convert snake_case to Title Case
        words = action_name.replace('_', ' ').title()
        return f"{words} for: {goal_title}"
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about GOAP planning usage.
        
        Returns:
            Dictionary with planning statistics
        """
        stats = {
            'goap_available': self.goap_available,
            'goap_enabled': self._should_use_goap(),
            'fallback_enabled': self.feature_flags.fallback_to_legacy if self.feature_flags else True
        }
        
        # Add GOAP planner stats if available
        if self.goap_available and self.goap_planner:
            try:
                from ..memory.metrics import metrics_registry
                
                stats['goap_planning_attempts'] = metrics_registry.get_counter('goap_planning_attempts_total')
                stats['goap_plans_found'] = metrics_registry.get_counter('goap_plans_found_total')
                stats['goap_plans_failed'] = metrics_registry.get_counter('goap_plans_not_found_total')
                
            except (ImportError, AttributeError):
                pass
        
        return stats
