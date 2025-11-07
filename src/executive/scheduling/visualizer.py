"""
Schedule Visualization Data Export

Exports schedule data in formats suitable for various visualizations:
- Gantt charts
- Timeline views
- Resource utilization graphs
- Dependency graphs
- Critical path visualization
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json

from .models import Task, Resource, Schedule, TaskStatus


@dataclass
class GanttBar:
    """A bar for a Gantt chart."""
    task_id: str
    task_name: str
    start: datetime
    end: datetime
    duration_hours: float
    progress: float = 0.0  # 0-1
    color: str = "#3498db"
    dependencies: List[str] = field(default_factory=list)
    is_critical: bool = False
    slack_hours: float = 0.0
    priority: float = 1.0
    status: str = "pending"


@dataclass
class TimelineEvent:
    """An event on a timeline."""
    timestamp: datetime
    event_type: str  # "start", "end", "milestone", "deadline"
    task_id: str
    task_name: str
    description: str
    icon: str = "●"
    color: str = "#3498db"


@dataclass
class ResourceUtilizationPoint:
    """Resource utilization at a point in time."""
    timestamp: datetime
    resource_name: str
    utilization: float  # 0-1
    capacity: float
    used: float
    tasks_using: List[str] = field(default_factory=list)


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    from_task_id: str
    to_task_id: str
    from_task_name: str
    to_task_name: str
    edge_type: str = "dependency"  # "dependency", "resource_conflict", "critical_path"
    strength: float = 1.0


class ScheduleVisualizer:
    """
    Export schedule data for various visualizations.
    
    Formats:
    - Gantt chart data (task bars with dependencies)
    - Timeline events (milestones, starts, ends)
    - Resource utilization over time
    - Dependency graph (nodes and edges)
    - Critical path visualization
    """
    
    def export_gantt_data(self, schedule: Schedule) -> List[GanttBar]:
        """
        Export schedule as Gantt chart data.
        
        Args:
            schedule: Schedule to export
            
        Returns:
            List of Gantt bars for each task
        """
        critical_tasks = set(schedule.calculate_critical_path())
        bars = []
        
        for task in schedule.get_scheduled_tasks():
            if not task.scheduled_start or not task.scheduled_end:
                continue
            
            slack = schedule.calculate_slack_time(task.id)
            slack_hours = slack.total_seconds() / 3600 if slack else 0.0
            
            # Color based on status and criticality
            if task.id in critical_tasks:
                color = "#e74c3c"  # Red for critical path
            elif task.status == TaskStatus.COMPLETED:
                color = "#27ae60"  # Green for completed
            elif task.status == TaskStatus.IN_PROGRESS:
                color = "#f39c12"  # Orange for in progress
            else:
                color = "#3498db"  # Blue for pending
            
            bar = GanttBar(
                task_id=task.id,
                task_name=task.name,
                start=task.scheduled_start,
                end=task.scheduled_end,
                duration_hours=task.duration.total_seconds() / 3600,
                progress=self._calculate_progress(task),
                color=color,
                dependencies=list(task.dependencies),
                is_critical=task.id in critical_tasks,
                slack_hours=slack_hours,
                priority=task.priority,
                status=task.status.value
            )
            bars.append(bar)
        
        # Sort by start time
        bars.sort(key=lambda b: b.start)
        
        return bars
    
    def export_timeline_data(self, schedule: Schedule) -> List[TimelineEvent]:
        """
        Export schedule as timeline events.
        
        Args:
            schedule: Schedule to export
            
        Returns:
            List of timeline events
        """
        events = []
        critical_tasks = set(schedule.calculate_critical_path())
        
        for task in schedule.get_scheduled_tasks():
            if not task.scheduled_start or not task.scheduled_end:
                continue
            
            color = "#e74c3c" if task.id in critical_tasks else "#3498db"
            
            # Start event
            events.append(TimelineEvent(
                timestamp=task.scheduled_start,
                event_type="start",
                task_id=task.id,
                task_name=task.name,
                description=f"Start: {task.name}",
                icon="▶",
                color=color
            ))
            
            # End event
            events.append(TimelineEvent(
                timestamp=task.scheduled_end,
                event_type="end",
                task_id=task.id,
                task_name=task.name,
                description=f"Complete: {task.name}",
                icon="✓",
                color=color
            ))
        
        # Add milestones (critical path tasks)
        for task_id in critical_tasks:
            task = schedule.get_task(task_id)
            if task and task.scheduled_end:
                events.append(TimelineEvent(
                    timestamp=task.scheduled_end,
                    event_type="milestone",
                    task_id=task.id,
                    task_name=task.name,
                    description=f"Milestone: {task.name}",
                    icon="◆",
                    color="#e74c3c"
                ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def export_resource_utilization_data(self, schedule: Schedule,
                                        time_resolution: timedelta = timedelta(hours=1)) -> List[ResourceUtilizationPoint]:
        """
        Export resource utilization over time.
        
        Args:
            schedule: Schedule to export
            time_resolution: Sampling interval
            
        Returns:
            List of utilization points for each resource at each time
        """
        points = []
        
        # Sample at regular intervals
        current_time = schedule.start_time
        while current_time <= schedule.end_time:
            for resource, timeline in schedule.resource_timeline.items():
                # Calculate utilization at this time
                used = 0.0
                using_tasks = []
                
                for start, end, amount in timeline:
                    if start <= current_time < end:
                        used += amount
                        # Find which tasks are using it
                        for task in schedule.tasks:
                            if (task.scheduled_start and task.scheduled_end and
                                task.scheduled_start <= current_time < task.scheduled_end and
                                resource in task.resource_requirements):
                                using_tasks.append(task.id)
                
                utilization = used / resource.capacity if resource.capacity > 0 else 0.0
                
                points.append(ResourceUtilizationPoint(
                    timestamp=current_time,
                    resource_name=resource.name,
                    utilization=utilization,
                    capacity=resource.capacity,
                    used=used,
                    tasks_using=using_tasks
                ))
            
            current_time += time_resolution
        
        return points
    
    def export_dependency_graph(self, schedule: Schedule) -> Tuple[List[Dict], List[DependencyEdge]]:
        """
        Export task dependency graph.
        
        Args:
            schedule: Schedule to export
            
        Returns:
            Tuple of (nodes, edges) for graph visualization
        """
        critical_tasks = set(schedule.calculate_critical_path())
        
        # Nodes (tasks)
        nodes = []
        for task in schedule.tasks:
            slack = schedule.calculate_slack_time(task.id)
            slack_hours = slack.total_seconds() / 3600 if slack else 0.0
            
            nodes.append({
                "id": task.id,
                "label": task.name,
                "is_critical": task.id in critical_tasks,
                "priority": task.priority,
                "duration_hours": task.duration.total_seconds() / 3600,
                "slack_hours": slack_hours,
                "status": task.status.value,
                "cognitive_load": task.cognitive_load
            })
        
        # Edges (dependencies)
        edges = []
        for task in schedule.tasks:
            for dep_id in task.dependencies:
                dep_task = schedule.get_task(dep_id)
                if dep_task:
                    # Determine edge type
                    if task.id in critical_tasks and dep_id in critical_tasks:
                        edge_type = "critical_path"
                        color = "#e74c3c"
                    else:
                        edge_type = "dependency"
                        color = "#95a5a6"
                    
                    edges.append(DependencyEdge(
                        from_task_id=dep_id,
                        to_task_id=task.id,
                        from_task_name=dep_task.name,
                        to_task_name=task.name,
                        edge_type=edge_type,
                        strength=1.0
                    ))
        
        return nodes, edges
    
    def export_critical_path_data(self, schedule: Schedule) -> Dict[str, Any]:
        """
        Export critical path visualization data.
        
        Args:
            schedule: Schedule to export
            
        Returns:
            Critical path data with highlighting
        """
        critical_task_ids = schedule.calculate_critical_path()
        critical_tasks = [schedule.get_task(tid) for tid in critical_task_ids]
        
        # Calculate critical path duration
        critical_duration = sum(
            (task.duration for task in critical_tasks if task),
            timedelta()
        )
        
        # Get task details
        task_details = []
        for task_id in critical_task_ids:
            task = schedule.get_task(task_id)
            if task and task.scheduled_start and task.scheduled_end:
                task_details.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "start": task.scheduled_start.isoformat(),
                    "end": task.scheduled_end.isoformat(),
                    "duration_hours": task.duration.total_seconds() / 3600,
                    "priority": task.priority,
                    "cognitive_load": task.cognitive_load
                })
        
        return {
            "critical_path_length": len(critical_task_ids),
            "critical_path_duration_hours": critical_duration.total_seconds() / 3600,
            "total_makespan_hours": schedule.makespan.total_seconds() / 3600,
            "critical_path_ratio": critical_duration.total_seconds() / schedule.makespan.total_seconds() if schedule.makespan.total_seconds() > 0 else 0,
            "tasks": task_details
        }
    
    def export_cognitive_load_graph(self, schedule: Schedule) -> List[Dict[str, Any]]:
        """
        Export cognitive load over time for graphing.
        
        Args:
            schedule: Schedule to export
            
        Returns:
            List of time points with cognitive load
        """
        points = []
        
        for timestamp, load in schedule.cognitive_timeline:
            points.append({
                "timestamp": timestamp.isoformat(),
                "cognitive_load": load,
                "threshold": 0.8,  # Warning threshold
                "status": "high" if load > 0.8 else "medium" if load > 0.5 else "low"
            })
        
        return points
    
    def export_to_json(self, schedule: Schedule, include_all: bool = True) -> str:
        """
        Export complete schedule data as JSON.
        
        Args:
            schedule: Schedule to export
            include_all: Include all visualization formats
            
        Returns:
            JSON string with all schedule data
        """
        data = {
            "schedule_info": {
                "start_time": schedule.start_time.isoformat(),
                "end_time": schedule.end_time.isoformat(),
                "makespan_hours": schedule.makespan.total_seconds() / 3600,
                "num_tasks": len(schedule.tasks),
                "is_feasible": schedule.is_feasible
            },
            "metrics": schedule.metrics
        }
        
        if include_all:
            # Gantt data
            gantt_bars = self.export_gantt_data(schedule)
            data["gantt_chart"] = [
                {
                    "task_id": bar.task_id,
                    "task_name": bar.task_name,
                    "start": bar.start.isoformat(),
                    "end": bar.end.isoformat(),
                    "duration_hours": bar.duration_hours,
                    "progress": bar.progress,
                    "color": bar.color,
                    "dependencies": bar.dependencies,
                    "is_critical": bar.is_critical,
                    "slack_hours": bar.slack_hours,
                    "priority": bar.priority,
                    "status": bar.status
                }
                for bar in gantt_bars
            ]
            
            # Timeline data
            timeline_events = self.export_timeline_data(schedule)
            data["timeline"] = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "task_id": event.task_id,
                    "task_name": event.task_name,
                    "description": event.description,
                    "icon": event.icon,
                    "color": event.color
                }
                for event in timeline_events
            ]
            
            # Resource utilization
            util_points = self.export_resource_utilization_data(schedule)
            data["resource_utilization"] = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "resource_name": point.resource_name,
                    "utilization": point.utilization,
                    "capacity": point.capacity,
                    "used": point.used,
                    "tasks_using": point.tasks_using
                }
                for point in util_points
            ]
            
            # Dependency graph
            nodes, edges = self.export_dependency_graph(schedule)
            data["dependency_graph"] = {
                "nodes": nodes,
                "edges": [
                    {
                        "from": edge.from_task_id,
                        "to": edge.to_task_id,
                        "from_name": edge.from_task_name,
                        "to_name": edge.to_task_name,
                        "type": edge.edge_type,
                        "strength": edge.strength
                    }
                    for edge in edges
                ]
            }
            
            # Critical path
            data["critical_path"] = self.export_critical_path_data(schedule)
            
            # Cognitive load
            data["cognitive_load"] = self.export_cognitive_load_graph(schedule)
        
        return json.dumps(data, indent=2, default=str)
    
    def _calculate_progress(self, task: Task) -> float:
        """Calculate task progress (0-1) based on status."""
        if task.status == TaskStatus.COMPLETED:
            return 1.0
        elif task.status == TaskStatus.IN_PROGRESS:
            return 0.5  # Could be enhanced with actual progress tracking
        else:
            return 0.0
