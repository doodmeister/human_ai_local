"""
Dynamic Scheduling: Reactive and Proactive Schedule Management

Features:
- Real-time schedule updates (add/remove tasks)
- Reactive scheduling (handle disruptions)
- Proactive scheduling (anticipate issues)
- Schedule monitoring and conflict detection
"""

from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import Task, Resource, Schedule, SchedulingProblem, TaskStatus
from .cp_scheduler import CPScheduler, SchedulerConfig

logger = logging.getLogger(__name__)


class DisruptionType(Enum):
    """Types of schedule disruptions."""
    TASK_FAILED = "task_failed"
    TASK_DELAYED = "task_delayed"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    DEADLINE_CHANGED = "deadline_changed"
    NEW_TASK_ADDED = "new_task_added"
    DEPENDENCY_ADDED = "dependency_added"


@dataclass
class Disruption:
    """A disruption to the current schedule."""
    type: DisruptionType
    timestamp: datetime
    affected_task_ids: List[str] = field(default_factory=list)
    affected_resources: List[Resource] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleWarning:
    """A proactive warning about potential schedule issues."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "resource_contention", "deadline_risk", "cognitive_overload"
    description: str
    affected_task_ids: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    risk_probability: float = 0.0  # 0-1
    impact_score: float = 0.0  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScheduleMonitor:
    """
    Monitors schedule execution and detects disruptions.
    
    Responsibilities:
    - Track task execution status
    - Detect failures and delays
    - Identify resource unavailability
    - Trigger reactive rescheduling
    """
    
    def __init__(self):
        """Initialize the schedule monitor."""
        self.disruptions: List[Disruption] = []
        self.task_status_history: Dict[str, List[Tuple[datetime, TaskStatus]]] = {}
    
    def check_task_status(self, task: Task, current_time: datetime) -> Optional[Disruption]:
        """
        Check if task is on track or experiencing issues.
        
        Args:
            task: Task to check
            current_time: Current time
            
        Returns:
            Disruption if detected, None otherwise
        """
        # Record status
        if task.id not in self.task_status_history:
            self.task_status_history[task.id] = []
        self.task_status_history[task.id].append((current_time, task.status))
        
        # Check for failure
        if task.status == TaskStatus.FAILED:
            disruption = Disruption(
                type=DisruptionType.TASK_FAILED,
                timestamp=current_time,
                affected_task_ids=[task.id],
                description=f"Task {task.name} failed"
            )
            self.disruptions.append(disruption)
            return disruption
        
        # Check for delay (started but not finished and past scheduled end)
        if (task.status == TaskStatus.IN_PROGRESS and 
            task.scheduled_end and current_time > task.scheduled_end):
            disruption = Disruption(
                type=DisruptionType.TASK_DELAYED,
                timestamp=current_time,
                affected_task_ids=[task.id],
                description=f"Task {task.name} delayed beyond scheduled end"
            )
            self.disruptions.append(disruption)
            return disruption
        
        return None
    
    def check_resource_availability(self, resource: Resource, 
                                   current_time: datetime,
                                   available_capacity: float) -> Optional[Disruption]:
        """
        Check if resource is available as expected.
        
        Args:
            resource: Resource to check
            current_time: Current time
            available_capacity: Current available capacity
            
        Returns:
            Disruption if resource unavailable, None otherwise
        """
        if available_capacity < resource.capacity:
            disruption = Disruption(
                type=DisruptionType.RESOURCE_UNAVAILABLE,
                timestamp=current_time,
                affected_resources=[resource],
                description=f"Resource {resource.name} reduced capacity: {available_capacity}/{resource.capacity}",
                metadata={"available_capacity": available_capacity}
            )
            self.disruptions.append(disruption)
            return disruption
        
        return None
    
    def get_affected_tasks(self, disruption: Disruption, schedule: Schedule) -> Set[str]:
        """
        Determine which tasks are affected by a disruption.
        
        Args:
            disruption: The disruption
            schedule: Current schedule
            
        Returns:
            Set of affected task IDs
        """
        affected = set(disruption.affected_task_ids)
        
        if disruption.type == DisruptionType.TASK_FAILED:
            # Find all dependent tasks
            failed_id = disruption.affected_task_ids[0]
            for task in schedule.tasks:
                if failed_id in task.dependencies:
                    affected.add(task.id)
                    # Recursively add dependents
                    affected.update(self._find_dependent_tasks(task.id, schedule))
        
        elif disruption.type == DisruptionType.RESOURCE_UNAVAILABLE:
            # Find all tasks using the affected resource
            for task in schedule.tasks:
                for resource in disruption.affected_resources:
                    if resource in task.resource_requirements:
                        affected.add(task.id)
        
        return affected
    
    def _find_dependent_tasks(self, task_id: str, schedule: Schedule) -> Set[str]:
        """Recursively find all tasks that depend on this task."""
        dependents = set()
        for task in schedule.tasks:
            if task_id in task.dependencies:
                dependents.add(task.id)
                dependents.update(self._find_dependent_tasks(task.id, schedule))
        return dependents
    
    def should_reschedule(self, schedule: Schedule, current_time: datetime) -> bool:
        """
        Determine if schedule needs to be regenerated.
        
        Args:
            schedule: Current schedule
            current_time: Current time
            
        Returns:
            True if rescheduling recommended
        """
        # Check for any disruptions in last hour
        recent_disruptions = [
            d for d in self.disruptions 
            if (current_time - d.timestamp) < timedelta(hours=1)
        ]
        
        # Reschedule if critical disruptions or multiple disruptions
        if recent_disruptions:
            critical = any(d.type in [DisruptionType.TASK_FAILED, DisruptionType.RESOURCE_UNAVAILABLE]
                          for d in recent_disruptions)
            return critical or len(recent_disruptions) >= 3
        
        return False


class ScheduleAnalyzer:
    """
    Proactively analyzes schedules to predict and warn about potential issues.
    
    Responsibilities:
    - Resource contention prediction
    - Deadline risk assessment
    - Cognitive overload warnings
    - Buffer time recommendations
    """
    
    def analyze_schedule(self, schedule: Schedule, current_time: datetime) -> List[ScheduleWarning]:
        """
        Comprehensive schedule analysis.
        
        Args:
            schedule: Schedule to analyze
            current_time: Current time
            
        Returns:
            List of warnings about potential issues
        """
        warnings = []
        
        warnings.extend(self.check_resource_contention(schedule, current_time))
        warnings.extend(self.check_deadline_risks(schedule, current_time))
        warnings.extend(self.check_cognitive_overload(schedule, current_time))
        warnings.extend(self.check_critical_path_risks(schedule))
        
        return warnings
    
    def check_resource_contention(self, schedule: Schedule, 
                                  current_time: datetime) -> List[ScheduleWarning]:
        """Check for resource contention issues."""
        warnings = []
        
        for resource, timeline in schedule.resource_timeline.items():
            utilization = schedule.get_resource_utilization(resource)
            
            # High utilization warning
            if utilization > 0.9:
                warnings.append(ScheduleWarning(
                    severity="high",
                    category="resource_contention",
                    description=f"Resource {resource.name} at {utilization:.1%} utilization",
                    affected_task_ids=[t.id for t in schedule.tasks 
                                      if resource in t.resource_requirements],
                    suggested_actions=[
                        f"Consider reducing {resource.name} requirements",
                        "Add additional resource capacity",
                        "Distribute tasks over longer timeframe"
                    ],
                    risk_probability=0.8,
                    impact_score=0.7
                ))
            
            elif utilization > 0.75:
                warnings.append(ScheduleWarning(
                    severity="medium",
                    category="resource_contention",
                    description=f"Resource {resource.name} approaching capacity at {utilization:.1%}",
                    affected_task_ids=[t.id for t in schedule.tasks 
                                      if resource in t.resource_requirements],
                    suggested_actions=["Monitor resource usage closely"],
                    risk_probability=0.5,
                    impact_score=0.4
                ))
        
        return warnings
    
    def check_deadline_risks(self, schedule: Schedule, 
                            current_time: datetime) -> List[ScheduleWarning]:
        """Check for tasks at risk of missing deadlines."""
        warnings = []
        
        for task in schedule.tasks:
            if not task.scheduled_end:
                continue
            
            # Calculate slack time
            slack = schedule.calculate_slack_time(task.id)
            if slack is None:
                continue
            
            slack_hours = slack.total_seconds() / 3600
            
            # Critical: zero or negative slack
            if slack_hours <= 0:
                warnings.append(ScheduleWarning(
                    severity="critical",
                    category="deadline_risk",
                    description=f"Task {task.name} has no slack time - on critical path",
                    affected_task_ids=[task.id],
                    suggested_actions=[
                        "Increase task priority",
                        "Allocate additional resources",
                        "Consider extending deadline if possible"
                    ],
                    risk_probability=0.9,
                    impact_score=0.9
                ))
            
            # High risk: very little slack
            elif slack_hours < 2:
                warnings.append(ScheduleWarning(
                    severity="high",
                    category="deadline_risk",
                    description=f"Task {task.name} has minimal slack time ({slack_hours:.1f}h)",
                    affected_task_ids=[task.id],
                    suggested_actions=["Monitor progress closely", "Prepare contingency plans"],
                    risk_probability=0.6,
                    impact_score=0.7
                ))
        
        return warnings
    
    def check_cognitive_overload(self, schedule: Schedule, 
                                current_time: datetime) -> List[ScheduleWarning]:
        """Check for cognitive load issues."""
        warnings = []
        
        peak_load = schedule.get_peak_cognitive_load()
        avg_load = schedule.get_average_cognitive_load()
        
        # Very high peak load
        if peak_load > 0.9:
            # Find time periods with high load
            high_load_times = [(time, load) for time, load in schedule.cognitive_timeline 
                              if load > 0.8]
            
            warnings.append(ScheduleWarning(
                severity="high",
                category="cognitive_overload",
                description=f"Peak cognitive load very high ({peak_load:.1%})",
                suggested_actions=[
                    "Distribute complex tasks more evenly",
                    "Add breaks between high-load tasks",
                    "Consider simplifying some tasks"
                ],
                risk_probability=0.7,
                impact_score=0.8,
                metadata={"peak_load": peak_load, "high_load_periods": len(high_load_times)}
            ))
        
        # High variance (spiky load)
        smoothness = schedule.calculate_cognitive_load_smoothness()
        if smoothness < 0.5:
            warnings.append(ScheduleWarning(
                severity="medium",
                category="cognitive_overload",
                description=f"Cognitive load is uneven (smoothness: {smoothness:.1%})",
                suggested_actions=["Smooth out task distribution"],
                risk_probability=0.5,
                impact_score=0.5
            ))
        
        return warnings
    
    def check_critical_path_risks(self, schedule: Schedule) -> List[ScheduleWarning]:
        """Check for risks on critical path."""
        warnings = []
        
        critical_tasks = schedule.calculate_critical_path()
        
        if len(critical_tasks) > len(schedule.tasks) * 0.7:
            warnings.append(ScheduleWarning(
                severity="high",
                category="critical_path_risk",
                description=f"Too many tasks on critical path ({len(critical_tasks)}/{len(schedule.tasks)})",
                affected_task_ids=critical_tasks,
                suggested_actions=[
                    "Increase parallelization where possible",
                    "Add buffer time to schedule",
                    "Consider reducing task scope"
                ],
                risk_probability=0.7,
                impact_score=0.8
            ))
        
        return warnings
    
    def recommend_buffer_time(self, schedule: Schedule) -> timedelta:
        """
        Recommend additional buffer time based on schedule analysis.
        
        Args:
            schedule: Schedule to analyze
            
        Returns:
            Recommended buffer time to add
        """
        robustness = schedule.calculate_robustness_score()
        
        # Low robustness = need more buffer
        if robustness < 0.3:
            # Recommend 30% additional buffer
            return timedelta(seconds=schedule.makespan.total_seconds() * 0.3)
        elif robustness < 0.6:
            # Recommend 15% additional buffer
            return timedelta(seconds=schedule.makespan.total_seconds() * 0.15)
        
        # Adequate robustness
        return timedelta(0)


class DynamicScheduler:
    """
    Dynamic scheduler with reactive and proactive capabilities.
    
    Features:
    - Real-time schedule updates
    - Reactive rescheduling on disruptions
    - Proactive warnings and recommendations
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize dynamic scheduler."""
        self.base_scheduler = CPScheduler(config)
        self.monitor = ScheduleMonitor()
        self.analyzer = ScheduleAnalyzer()
        self.current_schedule: Optional[Schedule] = None
        self.current_problem: Optional[SchedulingProblem] = None
    
    def create_initial_schedule(self, problem: SchedulingProblem) -> Schedule:
        """
        Create initial schedule and begin monitoring.
        
        Args:
            problem: Scheduling problem
            
        Returns:
            Initial schedule
        """
        self.current_problem = problem
        self.current_schedule = self.base_scheduler.schedule(problem)
        
        logger.info(f"Created initial schedule: makespan={self.current_schedule.makespan}, "
                   f"robustness={self.current_schedule.metrics.get('robustness_score', 0):.2f}")
        
        return self.current_schedule
    
    def update_schedule(self, new_tasks: Optional[List[Task]] = None,
                       modified_tasks: Optional[List[Task]] = None,
                       removed_task_ids: Optional[List[str]] = None) -> Schedule:
        """
        Update schedule with new/modified/removed tasks.
        
        Args:
            new_tasks: New tasks to add
            modified_tasks: Tasks with updated parameters
            removed_task_ids: IDs of tasks to remove
            
        Returns:
            Updated schedule
        """
        if not self.current_problem or not self.current_schedule:
            raise ValueError("No current schedule to update")
        
        # Create modified problem
        updated_problem = SchedulingProblem(
            tasks=list(self.current_problem.tasks),
            resources=self.current_problem.resources,
            constraints=self.current_problem.constraints,
            objectives=self.current_problem.objectives,
            horizon=self.current_problem.horizon,
            time_resolution=self.current_problem.time_resolution
        )
        
        # Apply changes
        if removed_task_ids:
            updated_problem.tasks = [t for t in updated_problem.tasks 
                                    if t.id not in removed_task_ids]
        
        if modified_tasks:
            task_map = {t.id: t for t in updated_problem.tasks}
            for modified in modified_tasks:
                task_map[modified.id] = modified
            updated_problem.tasks = list(task_map.values())
        
        if new_tasks:
            updated_problem.tasks.extend(new_tasks)
        
        # Reschedule
        self.current_problem = updated_problem
        self.current_schedule = self.base_scheduler.schedule(updated_problem)
        
        logger.info(f"Updated schedule: {len(new_tasks or [])} added, "
                   f"{len(modified_tasks or [])} modified, {len(removed_task_ids or [])} removed")
        
        return self.current_schedule
    
    def handle_disruption(self, disruption: Disruption) -> Schedule:
        """
        Reactively handle a schedule disruption.
        
        Args:
            disruption: The disruption that occurred
            
        Returns:
            New schedule accommodating the disruption
        """
        if not self.current_schedule or not self.current_problem:
            raise ValueError("No current schedule")
        
        logger.warning(f"Handling disruption: {disruption.type.value} - {disruption.description}")
        
        # Get affected tasks
        affected_ids = self.monitor.get_affected_tasks(disruption, self.current_schedule)
        
        # Handle based on disruption type
        if disruption.type == DisruptionType.TASK_FAILED:
            # Mark failed task and reschedule dependents
            failed_id = disruption.affected_task_ids[0]
            for task in self.current_problem.tasks:
                if task.id == failed_id:
                    task.status = TaskStatus.FAILED
        
        elif disruption.type == DisruptionType.RESOURCE_UNAVAILABLE:
            # Note: Resource capacity is immutable, so we'd need to create new Resource
            # For now, just log the issue - in production would need resource replacement
            logger.warning(f"Resource capacity reduced - consider creating new Resource objects")
        
        # Reschedule
        self.current_schedule = self.base_scheduler.schedule(self.current_problem)
        
        logger.info(f"Rescheduled after disruption: {len(affected_ids)} tasks affected")
        
        return self.current_schedule
    
    def get_proactive_warnings(self, current_time: datetime) -> List[ScheduleWarning]:
        """
        Get proactive warnings about potential issues.
        
        Args:
            current_time: Current time
            
        Returns:
            List of warnings
        """
        if not self.current_schedule:
            return []
        
        return self.analyzer.analyze_schedule(self.current_schedule, current_time)
    
    def get_schedule_health(self) -> Dict[str, Any]:
        """
        Get overall schedule health metrics.
        
        Returns:
            Dictionary of health indicators
        """
        if not self.current_schedule:
            return {"status": "no_schedule"}
        
        warnings = self.analyzer.analyze_schedule(self.current_schedule, datetime.now())
        
        return {
            "status": "healthy" if not warnings else "at_risk",
            "robustness_score": self.current_schedule.metrics.get("robustness_score", 0),
            "critical_warnings": len([w for w in warnings if w.severity == "critical"]),
            "high_warnings": len([w for w in warnings if w.severity == "high"]),
            "total_warnings": len(warnings),
            "buffer_time_hours": self.current_schedule.metrics.get("buffer_time_hours", 0),
            "resource_utilization_variance": self.current_schedule.metrics.get("resource_utilization_variance", 0),
            "cognitive_load_smoothness": self.current_schedule.metrics.get("cognitive_load_smoothness", 0)
        }
