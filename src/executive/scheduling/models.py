"""
Data Models for Constraint-Based Scheduler

Defines the core data structures for task scheduling, resource management,
and temporal constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class TaskStatus(Enum):
    """Status of a scheduled task."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    COGNITIVE = "cognitive"  # Mental attention/focus
    TIME = "time"  # Time slots
    TOOL = "tool"  # External tools/APIs
    MEMORY = "memory"  # Working memory capacity
    ENERGY = "energy"  # User energy/motivation
    CUSTOM = "custom"  # Custom resource types


@dataclass(frozen=True)
class Resource:
    """
    A resource that can be allocated to tasks.
    
    Resources represent anything tasks need to execute:
    - Cognitive capacity (attention)
    - Time slots
    - External tools
    - Memory
    """
    id: str
    name: str
    type: ResourceType
    capacity: float  # Maximum available amount
    unit: str = "units"  # e.g., "hours", "%, "points"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class TimeWindow:
    """
    A time window defining when a task can be scheduled.
    
    Supports:
    - Earliest start time
    - Latest end time (deadline)
    - Preferred time slots
    """
    earliest_start: datetime
    latest_end: datetime
    preferred_slots: List[tuple[datetime, datetime]] = field(default_factory=list)
    
    def duration(self) -> timedelta:
        """Total time window duration."""
        return self.latest_end - self.earliest_start
    
    def contains(self, time: datetime) -> bool:
        """Check if time is within window."""
        return self.earliest_start <= time <= self.latest_end
    
    def overlaps(self, other: 'TimeWindow') -> bool:
        """Check if this window overlaps with another."""
        return not (self.latest_end <= other.earliest_start or 
                   other.latest_end <= self.earliest_start)


@dataclass
class Task:
    """
    A task to be scheduled.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        duration: Expected execution time
        priority: Priority level (higher = more important)
        cognitive_load: Mental effort required (0-1)
        resource_requirements: Resources needed with amounts
        time_window: When task can be executed
        dependencies: Tasks that must complete before this one
        status: Current task status
        metadata: Additional key-value data
    """
    id: str
    name: str
    duration: timedelta
    priority: float = 1.0
    cognitive_load: float = 0.5  # 0 (easy) to 1 (very difficult)
    resource_requirements: Dict[Resource, float] = field(default_factory=dict)
    time_window: Optional[TimeWindow] = None
    dependencies: Set[str] = field(default_factory=set)  # Task IDs
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling results (populated after scheduling)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    assigned_resources: Dict[Resource, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_tasks)
    
    def estimated_end(self) -> Optional[datetime]:
        """Get estimated end time if scheduled."""
        if self.scheduled_start:
            return self.scheduled_start + self.duration
        return None


@dataclass
class SchedulingConstraint:
    """
    A constraint that must be satisfied by the schedule.
    
    Types:
    - precedence: Task A must finish before Task B starts
    - resource_capacity: Don't exceed resource limits
    - deadline: Task must finish by time T
    - cognitive_load: Don't exceed max cognitive load at any time
    - time_window: Task must start/end within window
    """
    type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hard: bool = True  # Hard constraint (must satisfy) vs soft (prefer to satisfy)
    weight: float = 1.0  # For soft constraints


@dataclass
class OptimizationObjective:
    """
    An optimization objective for the scheduler.
    
    Common objectives:
    - minimize_makespan: Minimize total schedule duration
    - minimize_cognitive_peaks: Smooth out cognitive load
    - maximize_priority: Complete high-priority tasks first
    - minimize_tardiness: Avoid missing deadlines
    """
    name: str
    description: str
    weight: float = 1.0  # Relative importance
    maximize: bool = False  # True = maximize, False = minimize


@dataclass
class Schedule:
    """
    A complete schedule for a set of tasks.
    
    Contains:
    - Scheduled tasks with start/end times
    - Resource allocations over time
    - Schedule quality metrics
    """
    tasks: List[Task]
    makespan: timedelta  # Total schedule duration
    start_time: datetime
    end_time: datetime
    
    # Resource usage timeline
    resource_timeline: Dict[Resource, List[tuple[datetime, datetime, float]]] = field(default_factory=dict)
    
    # Cognitive load timeline (time -> load)
    cognitive_timeline: List[tuple[datetime, float]] = field(default_factory=list)
    
    # Quality metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Feasibility
    is_feasible: bool = True
    infeasibility_reasons: List[str] = field(default_factory=list)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_scheduled_tasks(self) -> List[Task]:
        """Get tasks that have been scheduled."""
        return [t for t in self.tasks if t.scheduled_start is not None]
    
    def get_resource_utilization(self, resource: Resource) -> float:
        """Calculate average utilization of a resource (0-1)."""
        if resource not in self.resource_timeline:
            return 0.0
        
        total_used = sum((end - start).total_seconds() * amount 
                        for start, end, amount in self.resource_timeline[resource])
        total_available = self.makespan.total_seconds() * resource.capacity
        
        return total_used / total_available if total_available > 0 else 0.0
    
    def get_peak_cognitive_load(self) -> float:
        """Get maximum cognitive load at any point in time."""
        if not self.cognitive_timeline:
            return 0.0
        return max(load for _, load in self.cognitive_timeline)
    
    def get_average_cognitive_load(self) -> float:
        """Get average cognitive load across schedule."""
        if not self.cognitive_timeline:
            return 0.0
        return sum(load for _, load in self.cognitive_timeline) / len(self.cognitive_timeline)
    
    def calculate_critical_path(self) -> List[str]:
        """
        Calculate critical path - longest path through task dependencies.
        Tasks on critical path have zero slack time.
        
        Returns:
            List of task IDs on the critical path
        """
        scheduled = self.get_scheduled_tasks()
        if not scheduled:
            return []
        
        # Calculate earliest start/finish times (forward pass)
        earliest_start: Dict[str, datetime] = {}
        earliest_finish: Dict[str, datetime] = {}
        
        for task in sorted(scheduled, key=lambda t: t.scheduled_start or self.start_time):
            deps_finish = [earliest_finish[dep] for dep in task.dependencies if dep in earliest_finish]
            earliest_start[task.id] = max(deps_finish) if deps_finish else self.start_time
            earliest_finish[task.id] = earliest_start[task.id] + task.duration
        
        # Calculate latest start/finish times (backward pass)
        latest_finish: Dict[str, datetime] = {}
        latest_start: Dict[str, datetime] = {}
        
        # Start from end tasks
        for task in reversed(sorted(scheduled, key=lambda t: t.scheduled_end or self.end_time)):
            # Find tasks that depend on this one
            dependents = [t.id for t in scheduled if task.id in t.dependencies]
            
            if not dependents:
                latest_finish[task.id] = self.end_time
            else:
                latest_finish[task.id] = min(latest_start[dep] for dep in dependents if dep in latest_start)
            
            latest_start[task.id] = latest_finish[task.id] - task.duration
        
        # Tasks with zero slack are on critical path
        critical_tasks = [
            task.id for task in scheduled
            if task.id in earliest_start and task.id in latest_start
            and (latest_start[task.id] - earliest_start[task.id]).total_seconds() < 1.0
        ]
        
        return critical_tasks
    
    def calculate_slack_time(self, task_id: str) -> Optional[timedelta]:
        """
        Calculate slack/float time for a task.
        Slack = latest_start - earliest_start
        
        Returns:
            Slack time, or None if task not scheduled
        """
        task = self.get_task(task_id)
        if not task or not task.scheduled_start:
            return None
        
        # Calculate earliest possible start considering dependencies
        earliest = self.start_time
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if dep_task and dep_task.scheduled_end:
                earliest = max(earliest, dep_task.scheduled_end)
        
        # Calculate latest possible start without delaying end
        latest = task.scheduled_start
        for other in self.tasks:
            if task.id in other.dependencies and other.scheduled_start:
                latest = min(latest, other.scheduled_start - task.duration)
        
        return latest - earliest
    
    def calculate_buffer_time(self) -> timedelta:
        """
        Calculate total buffer/slack time in schedule.
        Buffer = makespan - critical_path_length
        
        Returns:
            Total buffer time available
        """
        critical_path_ids = self.calculate_critical_path()
        critical_tasks = [self.get_task(tid) for tid in critical_path_ids]
        
        critical_path_duration = sum(
            (task.duration for task in critical_tasks if task),
            timedelta()
        )
        
        return self.makespan - critical_path_duration
    
    def calculate_robustness_score(self) -> float:
        """
        Calculate schedule robustness (0-1).
        Higher score = more resilient to disruptions.
        
        Considers:
        - Total slack/buffer time
        - Resource utilization variance (less variance = more robust)
        - Cognitive load smoothness
        
        Returns:
            Robustness score 0-1 (higher is better)
        """
        if not self.tasks or self.makespan.total_seconds() == 0:
            return 0.0
        
        # Component 1: Normalized buffer time (0-1)
        buffer = self.calculate_buffer_time()
        buffer_ratio = buffer.total_seconds() / self.makespan.total_seconds()
        buffer_score = min(1.0, buffer_ratio * 2)  # 50% buffer = full score
        
        # Component 2: Resource utilization variance (lower = better)
        utilizations = [self.get_resource_utilization(r) for r in self.resource_timeline.keys()]
        if utilizations:
            mean_util = sum(utilizations) / len(utilizations)
            variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
            util_score = 1.0 - min(1.0, variance * 4)  # Low variance = high score
        else:
            util_score = 1.0
        
        # Component 3: Cognitive load smoothness (lower variance = better)
        if self.cognitive_timeline:
            loads = [load for _, load in self.cognitive_timeline]
            mean_load = sum(loads) / len(loads)
            load_variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
            cognitive_score = 1.0 - min(1.0, load_variance * 2)
        else:
            cognitive_score = 1.0
        
        # Weighted average
        return 0.4 * buffer_score + 0.3 * util_score + 0.3 * cognitive_score
    
    def calculate_resource_utilization_variance(self) -> float:
        """
        Calculate variance in resource utilization across all resources.
        Lower variance = more balanced resource usage.
        
        Returns:
            Variance in utilization (0 = perfect balance)
        """
        utilizations = [self.get_resource_utilization(r) for r in self.resource_timeline.keys()]
        if not utilizations:
            return 0.0
        
        mean = sum(utilizations) / len(utilizations)
        variance = sum((u - mean) ** 2 for u in utilizations) / len(utilizations)
        return variance
    
    def calculate_cognitive_load_smoothness(self) -> float:
        """
        Calculate smoothness of cognitive load over time.
        Higher score = smoother (less spiky) cognitive load.
        
        Returns:
            Smoothness score 0-1 (higher is smoother)
        """
        if len(self.cognitive_timeline) < 2:
            return 1.0
        
        # Calculate variance of load changes
        loads = [load for _, load in self.cognitive_timeline]
        changes = [abs(loads[i+1] - loads[i]) for i in range(len(loads)-1)]
        
        if not changes:
            return 1.0
        
        mean_change = sum(changes) / len(changes)
        # Normalize: small changes = smooth = high score
        return 1.0 - min(1.0, mean_change * 2)
    
    def update_quality_metrics(self):
        """Calculate and update all quality metrics."""
        self.metrics["critical_path_length"] = len(self.calculate_critical_path())
        self.metrics["buffer_time_hours"] = self.calculate_buffer_time().total_seconds() / 3600
        self.metrics["robustness_score"] = self.calculate_robustness_score()
        self.metrics["resource_utilization_variance"] = self.calculate_resource_utilization_variance()
        self.metrics["cognitive_load_smoothness"] = self.calculate_cognitive_load_smoothness()
        self.metrics["peak_cognitive_load"] = self.get_peak_cognitive_load()
        self.metrics["average_cognitive_load"] = self.get_average_cognitive_load()
        
        # Calculate per-resource utilization
        for resource in self.resource_timeline.keys():
            util = self.get_resource_utilization(resource)
            self.metrics[f"utilization_{resource.name}"] = util


@dataclass
class SchedulingProblem:
    """
    A complete scheduling problem specification.
    
    Contains:
    - Tasks to schedule
    - Available resources
    - Constraints
    - Optimization objectives
    """
    tasks: List[Task]
    resources: List[Resource]
    constraints: List[SchedulingConstraint] = field(default_factory=list)
    objectives: List[OptimizationObjective] = field(default_factory=list)
    horizon: timedelta = field(default_factory=lambda: timedelta(days=7))  # Planning horizon
    time_resolution: timedelta = field(default_factory=lambda: timedelta(minutes=15))  # Time slot size
    
    def add_precedence_constraint(self, before: Task, after: Task):
        """Add constraint that 'before' must complete before 'after' starts."""
        self.constraints.append(SchedulingConstraint(
            type="precedence",
            description=f"{before.name} must finish before {after.name}",
            parameters={"before": before.id, "after": after.id}
        ))
    
    def add_deadline_constraint(self, task: Task, deadline: datetime):
        """Add constraint that task must complete by deadline."""
        self.constraints.append(SchedulingConstraint(
            type="deadline",
            description=f"{task.name} must complete by {deadline}",
            parameters={"task": task.id, "deadline": deadline}
        ))
    
    def add_cognitive_load_limit(self, max_load: float):
        """Add constraint on maximum cognitive load at any time."""
        self.constraints.append(SchedulingConstraint(
            type="cognitive_load_limit",
            description=f"Cognitive load must not exceed {max_load}",
            parameters={"max_load": max_load}
        ))
