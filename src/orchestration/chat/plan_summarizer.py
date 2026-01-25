"""
Plan and Schedule Summarizer - Task 7

Converts technical GOAP plans and schedules into human-readable natural language
for chat responses.

Architecture:
    Technical Plan/Schedule → PlanSummarizer → Natural Language
    
Usage:
    summarizer = PlanSummarizer()
    
    # Humanize GOAP plan
    text = summarizer.humanize_plan(plan)
    # Output: "I'll need to: 1) Gather data (2 min), 2) Analyze (5 min), 3) Report (8 min)"
    
    # Humanize schedule
    text = summarizer.humanize_schedule(schedule)
    # Output: "Starting at 2:00 PM: Data gathering → Analysis → Reporting. Done by 2:15 PM."
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from src.executive.planning.goap_planner import Plan
from src.executive.scheduling.models import Schedule

logger = logging.getLogger(__name__)


# Action name humanization mappings
ACTION_HUMANIZATION = {
    # Data operations
    "gather_data": "gather data",
    "analyze_data": "analyze the data",
    "validate_data": "validate data quality",
    "transform_data": "transform the data",
    
    # Document operations  
    "create_document": "create a document",
    "review_document": "review the document",
    "publish_document": "publish the document",
    "archive_document": "archive the document",
    
    # Communication
    "send_notification": "send notifications",
    "request_approval": "request approval",
    "schedule_meeting": "schedule a meeting",
    
    # Resource management
    "allocate_resources": "allocate resources",
    "release_resources": "release resources",
    
    # Generic fallback
    "execute_task": "execute the task",
    "verify_result": "verify the results",
}


@dataclass
class PlanSummary:
    """
    Human-readable summary of a plan.
    
    Contains both brief and detailed descriptions for different contexts.
    """
    brief: str  # One-line summary
    detailed: str  # Multi-line with steps
    step_count: int
    estimated_duration: Optional[timedelta] = None
    key_actions: List[str] = None
    
    def __post_init__(self):
        if self.key_actions is None:
            self.key_actions = []


class PlanSummarizer:
    """
    Converts technical plans and schedules into natural language.
    
    Provides human-friendly descriptions of execution plans for chat responses.
    """
    
    def __init__(self, action_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize summarizer.
        
        Args:
            action_mappings: Custom action name humanization (extends defaults)
        """
        self.action_mappings = ACTION_HUMANIZATION.copy()
        if action_mappings:
            self.action_mappings.update(action_mappings)
    
    def humanize_action(self, action_name: str) -> str:
        """
        Convert technical action name to human-friendly phrase.
        
        Args:
            action_name: Technical action name (e.g., "gather_data")
            
        Returns:
            Human-friendly phrase (e.g., "gather data")
        """
        # Try direct mapping
        if action_name in self.action_mappings:
            return self.action_mappings[action_name]
        
        # Fallback: convert snake_case to title case
        return action_name.replace('_', ' ').lower()
    
    def humanize_plan(self, plan: Plan, include_cost: bool = False) -> PlanSummary:
        """
        Convert GOAP plan to natural language summary.
        
        Args:
            plan: GOAP plan to humanize
            include_cost: Whether to mention costs (default False for chat)
            
        Returns:
            PlanSummary with brief and detailed descriptions
        """
        if plan.is_empty():
            return PlanSummary(
                brief="No actions needed",
                detailed="The goal can be achieved with no additional actions.",
                step_count=0
            )
        
        # Build brief summary
        if len(plan.steps) == 1:
            brief = f"I'll {self.humanize_action(plan.steps[0].action.name)}"
        elif len(plan.steps) == 2:
            brief = f"I'll {self.humanize_action(plan.steps[0].action.name)} and {self.humanize_action(plan.steps[1].action.name)}"
        else:
            first = self.humanize_action(plan.steps[0].action.name)
            last = self.humanize_action(plan.steps[-1].action.name)
            brief = f"I'll {first}, then {len(plan.steps) - 2} more steps, ending with {last}"
        
        # Build detailed summary
        lines = ["Here's the plan:"]
        for i, step in enumerate(plan.steps, 1):
            action_text = self.humanize_action(step.action.name)
            
            # Estimate time from cost (rough heuristic: cost = minutes)
            time_estimate = ""
            if include_cost and step.cost > 0:
                minutes = int(step.cost)
                if minutes < 60:
                    time_estimate = f" ({minutes} min)"
                else:
                    hours = minutes / 60
                    time_estimate = f" ({hours:.1f} hrs)"
            
            lines.append(f"  {i}. {action_text.capitalize()}{time_estimate}")
        
        # Add total estimate
        if include_cost and plan.total_cost > 0:
            total_minutes = int(plan.total_cost)
            if total_minutes < 60:
                lines.append(f"\nEstimated total: {total_minutes} minutes")
            else:
                hours = total_minutes / 60
                lines.append(f"\nEstimated total: {hours:.1f} hours")
        
        detailed = "\n".join(lines)
        
        # Extract key actions for metadata
        all_actions = [self.humanize_action(s.action.name) for s in plan.steps]
        
        return PlanSummary(
            brief=brief,
            detailed=detailed,
            step_count=len(plan.steps),
            estimated_duration=timedelta(minutes=plan.total_cost) if plan.total_cost > 0 else None,
            key_actions=all_actions
        )
    
    def humanize_schedule(self, schedule: Schedule, show_times: bool = True) -> str:
        """
        Convert schedule to natural language timeline.
        
        Args:
            schedule: Schedule with task timings
            show_times: Whether to show specific times (default True)
            
        Returns:
            Human-readable schedule description
        """
        if not schedule.tasks:
            return "No tasks scheduled yet."
        
        lines = []
        
        # Group tasks by start time
        tasks_by_time = {}
        for task in schedule.tasks:
            if task.scheduled_start:
                start_key = task.scheduled_start.strftime("%H:%M")
                if start_key not in tasks_by_time:
                    tasks_by_time[start_key] = []
                tasks_by_time[start_key].append(task)
        
        # Build timeline
        if show_times and tasks_by_time:
            # Sort by time
            sorted_times = sorted(tasks_by_time.keys())
            
            lines.append("**Schedule:**")
            for time_key in sorted_times:
                tasks = tasks_by_time[time_key]
                if len(tasks) == 1:
                    task = tasks[0]
                    duration_min = int(task.duration.total_seconds() / 60)
                    if duration_min > 0:
                        lines.append(f"• {time_key}: {task.id} ({duration_min} min)")
                    else:
                        lines.append(f"• {time_key}: {task.id}")
                else:
                    lines.append(f"• {time_key}: {len(tasks)} tasks in parallel")
                    for task in tasks:
                        lines.append(f"  - {task.id}")
            
            # Add completion time
            if schedule.tasks:
                last_task = max(schedule.tasks, 
                              key=lambda t: t.scheduled_end if t.scheduled_end else datetime.min)
                if last_task.scheduled_end:
                    end_time = last_task.scheduled_end.strftime("%H:%M")
                    lines.append(f"\n✓ Expected completion: {end_time}")
        
        else:
            # Simple list without times
            lines.append("**Task Sequence:**")
            for i, task in enumerate(schedule.tasks, 1):
                lines.append(f"{i}. {task.id}")
        
        # Add makespan
        if schedule.makespan:
            total_min = int(schedule.makespan.total_seconds() / 60)
            if total_min < 60:
                lines.append(f"\nTotal duration: {total_min} minutes")
            else:
                hours = total_min / 60
                lines.append(f"\nTotal duration: {hours:.1f} hours")
        
        return "\n".join(lines)
    
    def format_execution_summary(
        self,
        plan: Optional[Plan] = None,
        schedule: Optional[Schedule] = None,
        include_details: bool = True
    ) -> str:
        """
        Format combined plan and schedule for chat response.
        
        Args:
            plan: Optional GOAP plan
            schedule: Optional schedule
            include_details: Whether to include detailed breakdown
            
        Returns:
            Formatted summary for chat
        """
        lines = []
        
        # Plan summary
        if plan:
            plan_summary = self.humanize_plan(plan, include_cost=True)
            lines.append(plan_summary.brief)
            
            if include_details and len(plan.steps) > 1:
                lines.append("")
                lines.append(plan_summary.detailed)
        
        # Schedule summary
        if schedule:
            lines.append("")
            schedule_text = self.humanize_schedule(schedule, show_times=True)
            lines.append(schedule_text)
        
        return "\n".join(lines)
    
    def format_progress_update(
        self,
        completed_steps: List[str],
        current_step: Optional[str],
        remaining_steps: List[str],
        percent_complete: float
    ) -> str:
        """
        Format progress update for ongoing execution.
        
        Args:
            completed_steps: List of completed step names
            current_step: Current step being executed
            remaining_steps: List of remaining steps
            percent_complete: Progress percentage (0-100)
            
        Returns:
            Formatted progress update
        """
        lines = []
        
        # Overall progress
        lines.append(f"**Progress: {percent_complete:.0f}% complete**")
        lines.append("")
        
        # Recent completions (last 3)
        if completed_steps:
            lines.append("✓ **Completed:**")
            for step in completed_steps[-3:]:
                humanized = self.humanize_action(step)
                lines.append(f"  • {humanized}")
            lines.append("")
        
        # Current work
        if current_step:
            humanized = self.humanize_action(current_step)
            lines.append(f"▶️ **Current:** {humanized}")
            lines.append("")
        
        # What's next (next 3)
        if remaining_steps:
            lines.append("⏭️ **Coming up:**")
            for step in remaining_steps[:3]:
                humanized = self.humanize_action(step)
                lines.append(f"  • {humanized}")
        
        return "\n".join(lines)


# Factory function
def create_summarizer(action_mappings: Optional[Dict[str, str]] = None) -> PlanSummarizer:
    """
    Create PlanSummarizer instance.
    
    Args:
        action_mappings: Optional custom action humanization mappings
        
    Returns:
        PlanSummarizer instance
    """
    return PlanSummarizer(action_mappings)
