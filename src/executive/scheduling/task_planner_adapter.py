"""
Task Planner Integration Adapter for CP Scheduler

Bridges the new constraint-based scheduler with the existing TaskPlanner.
Provides backward compatibility while enabling advanced scheduling features.

Features:
- Convert TaskPlanner tasks to CP scheduler format
- Feature flags for gradual rollout
- Fallback to legacy planning if CP scheduler unavailable
- Metrics tracking for comparison
"""

from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..task_planner import Task as PlannerTask, TaskStatus as PlannerStatus, TaskPlanner
from .models import (
    Task as SchedulerTask,
    Resource,
    ResourceType,
    SchedulingProblem,
    TimeWindow,
    OptimizationObjective,
)
from .cp_scheduler import CPScheduler, SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class SchedulerFeatureFlags:
    """Feature flags for CP scheduler integration."""
    enabled: bool = False  # Master switch for CP scheduler
    use_resource_constraints: bool = True
    use_cognitive_load_limits: bool = True
    use_deadlines: bool = True
    optimize_makespan: bool = True
    optimize_priority: bool = True
    fallback_on_error: bool = True  # Fall back to legacy if CP fails
    log_performance: bool = True


def get_scheduler_feature_flags() -> SchedulerFeatureFlags:
    """Get current feature flags (can be overridden by config/env)."""
    return SchedulerFeatureFlags(
        enabled=False,  # Start disabled for safety
        fallback_on_error=True
    )


class TaskPlannerSchedulerAdapter:
    """
    Adapter between TaskPlanner and CP Scheduler.
    
    Converts TaskPlanner's task format to CP scheduler format,
    invokes scheduling, and converts results back.
    
    Usage:
        adapter = TaskPlannerSchedulerAdapter(task_planner)
        scheduled_tasks = adapter.schedule_tasks(tasks)
    """
    
    def __init__(
        self,
        task_planner: Optional[TaskPlanner] = None,
        feature_flags: Optional[SchedulerFeatureFlags] = None
    ):
        """
        Initialize adapter.
        
        Args:
            task_planner: Existing TaskPlanner instance (for fallback)
            feature_flags: Feature flag configuration
        """
        self.task_planner = task_planner
        self.feature_flags = feature_flags or get_scheduler_feature_flags()
        
        if self.feature_flags.enabled:
            self.scheduler = CPScheduler(SchedulerConfig(
                max_solve_time_seconds=30.0,
                num_workers=4,
                log_search_progress=self.feature_flags.log_performance
            ))
        else:
            self.scheduler = None
    
    def schedule_tasks(
        self,
        tasks: List[PlannerTask],
        available_resources: Optional[List[Resource]] = None,
        max_cognitive_load: float = 1.0,
        horizon_days: int = 7
    ) -> List[PlannerTask]:
        """
        Schedule tasks using CP scheduler (if enabled) or fall back to legacy.
        
        Args:
            tasks: List of TaskPlanner tasks to schedule
            available_resources: Available resources (default: cognitive only)
            max_cognitive_load: Maximum cognitive load at any time
            horizon_days: Planning horizon in days
            
        Returns:
            List of scheduled tasks with start times
        """
        if not self.feature_flags.enabled or not self.scheduler:
            logger.info("CP scheduler disabled, using legacy planning")
            return self._legacy_schedule(tasks)
        
        try:
            return self._cp_schedule(tasks, available_resources, max_cognitive_load, horizon_days)
        except Exception as e:
            logger.error(f"CP scheduler failed: {e}")
            
            if self.feature_flags.fallback_on_error:
                logger.info("Falling back to legacy planning")
                return self._legacy_schedule(tasks)
            else:
                raise
    
    def _cp_schedule(
        self,
        tasks: List[PlannerTask],
        available_resources: Optional[List[Resource]],
        max_cognitive_load: float,
        horizon_days: int
    ) -> List[PlannerTask]:
        """Schedule using CP-SAT scheduler."""
        start_time = datetime.now()
        
        # Convert tasks to scheduler format
        scheduler_tasks = self._convert_tasks_to_scheduler_format(tasks)
        
        # Create default resources if none provided
        if available_resources is None:
            available_resources = [
                Resource(
                    id="cognitive",
                    name="Cognitive Capacity",
                    type=ResourceType.COGNITIVE,
                    capacity=1.0
                )
            ]
        
        # Create scheduling problem
        problem = SchedulingProblem(
            tasks=scheduler_tasks,
            resources=available_resources,
            horizon=timedelta(days=horizon_days)
        )
        
        # Add constraints
        if self.feature_flags.use_cognitive_load_limits:
            problem.add_cognitive_load_limit(max_cognitive_load)
        
        # Add objectives
        if self.feature_flags.optimize_makespan:
            problem.objectives.append(
                OptimizationObjective(
                    name="minimize_makespan",
                    description="Minimize total schedule duration",
                    weight=1.0
                )
            )
        
        if self.feature_flags.optimize_priority:
            problem.objectives.append(
                OptimizationObjective(
                    name="maximize_priority",
                    description="Prioritize high-priority tasks",
                    weight=0.5
                )
            )
        
        # Solve
        if self.scheduler is None:
            raise RuntimeError("CP scheduler not initialized")
        
        schedule = self.scheduler.schedule(problem)
        
        if not schedule.is_feasible:
            logger.warning(f"CP scheduler found no feasible schedule: {schedule.infeasibility_reasons}")
            if self.feature_flags.fallback_on_error:
                return self._legacy_schedule(tasks)
            else:
                raise ValueError(f"No feasible schedule: {schedule.infeasibility_reasons}")
        
        # Convert results back
        scheduled_tasks = self._convert_schedule_to_planner_format(schedule, tasks)
        
        # Log performance
        if self.feature_flags.log_performance:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"CP scheduler completed in {elapsed:.2f}s: "
                f"{len(scheduled_tasks)} tasks, "
                f"makespan={schedule.metrics.get('makespan_hours', 0):.1f}h"
            )
        
        return scheduled_tasks
    
    def _legacy_schedule(self, tasks: List[PlannerTask]) -> List[PlannerTask]:
        """Fall back to legacy task planning (simple ordering)."""
        # Use simple dependency ordering since TaskPlanner doesn't have order_tasks_by_dependencies
        return self._simple_dependency_order(tasks)
    
    def _convert_tasks_to_scheduler_format(self, tasks: List[PlannerTask]) -> List[SchedulerTask]:
        """Convert TaskPlanner tasks to CP scheduler format."""
        scheduler_tasks = []
        
        for task in tasks:
            # Estimate duration (default 1 hour if not specified)
            duration = timedelta(hours=getattr(task, 'estimated_duration_hours', 1.0))
            
            # Convert priority (normalize to 0-2 range)
            priority = float(getattr(task, 'priority_score', 1.0))
            
            # Get cognitive load (default 0.5)
            cognitive_load = float(getattr(task, 'cognitive_load', 0.5))
            
            # Convert dependencies (task IDs)
            dependencies = set(getattr(task, 'dependencies', []))
            
            # Create time window if task has deadline
            time_window = None
            deadline = getattr(task, 'deadline', None)
            if deadline:
                time_window = TimeWindow(
                    earliest_start=datetime.now(),
                    latest_end=deadline
                )
            
            scheduler_task = SchedulerTask(
                id=task.id,
                name=task.title,
                duration=duration,
                priority=priority,
                cognitive_load=cognitive_load,
                dependencies=dependencies,
                time_window=time_window,
                metadata={"original_task": task}
            )
            
            scheduler_tasks.append(scheduler_task)
        
        return scheduler_tasks
    
    def _convert_schedule_to_planner_format(
        self,
        schedule,
        original_tasks: List[PlannerTask]
    ) -> List[PlannerTask]:
        """Convert CP scheduler results back to TaskPlanner format."""
        # Create task ID to original task mapping
        task_map = {t.id: t for t in original_tasks}
        
        # Create schedule metadata mapping
        schedule_metadata = {}
        
        for scheduled_task in schedule.get_scheduled_tasks():
            if scheduled_task.id in task_map:
                schedule_metadata[scheduled_task.id] = {
                    'scheduled_start': scheduled_task.scheduled_start,
                    'scheduled_end': scheduled_task.scheduled_end
                }
        
        # Return tasks sorted by scheduled start time
        # Tasks keep their original state; scheduling info is in metadata
        scheduled = []
        unscheduled = []
        
        for task in original_tasks:
            if task.id in schedule_metadata:
                # Store scheduling info in task's existing metadata if available
                metadata = getattr(task, 'metadata', None)
                if isinstance(metadata, dict):
                    metadata.update(schedule_metadata[task.id])
                scheduled.append(task)
            else:
                unscheduled.append(task)
        
        # Sort scheduled tasks by their scheduled start time
        scheduled.sort(key=lambda t: schedule_metadata[t.id]['scheduled_start'])
        
        return scheduled + unscheduled
    
    def _simple_dependency_order(self, tasks: List[PlannerTask]) -> List[PlannerTask]:
        """Simple topological sort of tasks by dependencies."""
        ordered = []
        completed = set()
        
        remaining = tasks.copy()
        
        while remaining:
            # Find tasks with no unmet dependencies
            ready = []
            for task in remaining:
                deps = set(getattr(task, 'dependencies', []))
                if deps.issubset(completed):
                    ready.append(task)
            
            if not ready:
                # Break cycles by picking any task
                ready = [remaining[0]]
            
            # Add ready tasks to order
            for task in ready:
                ordered.append(task)
                completed.add(task.id)
                remaining.remove(task)
        
        return ordered


def create_scheduler_adapter(
    task_planner: Optional[TaskPlanner] = None,
    enable_cp_scheduler: bool = False
) -> TaskPlannerSchedulerAdapter:
    """
    Factory function to create scheduler adapter.
    
    Args:
        task_planner: Existing TaskPlanner instance
        enable_cp_scheduler: Enable CP scheduler (default: False for safety)
        
    Returns:
        TaskPlannerSchedulerAdapter instance
    """
    flags = get_scheduler_feature_flags()
    
    if enable_cp_scheduler:
        flags.enabled = True
    
    return TaskPlannerSchedulerAdapter(task_planner, flags)
