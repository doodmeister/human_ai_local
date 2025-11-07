"""
Constraint-Based Task Scheduling Module

Provides sophisticated task scheduling using constraint programming (CP-SAT).

Key Components:
- CPScheduler: Main scheduler using OR-Tools CP-SAT solver
- Data models: Task, Resource, Schedule, constraints, objectives
- Resource management and timeline tracking

Usage:
    from src.executive.scheduling import CPScheduler, SchedulingProblem, Task, Resource
    
    # Create tasks
    task1 = Task(id="t1", name="Analyze data", duration=timedelta(hours=2))
    task2 = Task(id="t2", name="Write report", duration=timedelta(hours=3))
    
    # Create resources
    analyst = Resource(id="r1", name="Analyst", type=ResourceType.COGNITIVE, capacity=1.0)
    
    # Create problem
    problem = SchedulingProblem(tasks=[task1, task2], resources=[analyst])
    problem.add_precedence_constraint(task1, task2)
    
    # Schedule
    scheduler = CPScheduler()
    schedule = scheduler.schedule(problem)
    
    if schedule.is_feasible:
        for task in schedule.get_scheduled_tasks():
            print(f"{task.name}: {task.scheduled_start} - {task.scheduled_end}")
"""

from .models import (
    Task,
    Resource,
    ResourceType,
    TimeWindow,
    Schedule,
    SchedulingProblem,
    SchedulingConstraint,
    OptimizationObjective,
    TaskStatus,
)

from .cp_scheduler import (
    CPScheduler,
    SchedulerConfig,
)

from .task_planner_adapter import (
    TaskPlannerSchedulerAdapter,
    SchedulerFeatureFlags,
    create_scheduler_adapter,
    get_scheduler_feature_flags,
)

from .dynamic_scheduler import (
    DynamicScheduler,
    ScheduleMonitor,
    ScheduleAnalyzer,
    Disruption,
    DisruptionType,
    ScheduleWarning,
)

from .visualizer import (
    ScheduleVisualizer,
    GanttBar,
    TimelineEvent,
    ResourceUtilizationPoint,
    DependencyEdge,
)

__all__ = [
    # Data models
    "Task",
    "Resource",
    "ResourceType",
    "TimeWindow",
    "Schedule",
    "SchedulingProblem",
    "SchedulingConstraint",
    "OptimizationObjective",
    "TaskStatus",
    # Core scheduler
    "CPScheduler",
    "SchedulerConfig",
    # Dynamic scheduling (Week 14)
    "DynamicScheduler",
    "ScheduleMonitor",
    "ScheduleAnalyzer",
    "Disruption",
    "DisruptionType",
    "ScheduleWarning",
    # Visualization
    "ScheduleVisualizer",
    "GanttBar",
    "TimelineEvent",
    "ResourceUtilizationPoint",
    "DependencyEdge",
    # Legacy integration
    "TaskPlannerSchedulerAdapter",
    "SchedulerFeatureFlags",
    "create_scheduler_adapter",
    "get_scheduler_feature_flags",
]

__version__ = "0.1.0"
