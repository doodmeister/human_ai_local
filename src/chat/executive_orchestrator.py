"""
Executive Pipeline Orchestrator - Task 6

Async orchestration layer for chat-integrated goal execution.
Provides background execution of goals with progress tracking and status queries.

Architecture:
    ChatService → ExecutiveOrchestrator → ExecutiveSystem → GOAP/Scheduler
    
Usage:
    orchestrator = ExecutiveOrchestrator()
    
    # Start background execution
    await orchestrator.execute_goal_async(goal_id)
    
    # Check status
    status = orchestrator.get_execution_status(goal_id)
    
    # Query for chat responses
    summary = orchestrator.format_status_for_chat(goal_id)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

from src.executive.integration import ExecutiveSystem, ExecutionContext, ExecutionStatus
from src.executive.planning.world_state import WorldState
from src.executive.goal_manager import Goal, GoalStatus
from src.chat.plan_summarizer import create_summarizer

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Detailed execution phase for progress tracking."""
    QUEUED = "queued"
    DECIDING = "deciding"
    PLANNING = "planning"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionProgress:
    """
    Progress information for a goal execution.
    
    Provides detailed status for chat responses and UI display.
    """
    goal_id: str
    goal_title: str
    phase: ExecutionPhase
    
    # Progress metrics
    progress_percent: float = 0.0  # 0-100
    actions_completed: int = 0
    total_actions: int = 0
    current_action: Optional[str] = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    # Status
    success: bool = False
    error_message: Optional[str] = None
    
    # Details for chat
    recent_steps: List[str] = field(default_factory=list)  # Last 3 steps
    next_steps: List[str] = field(default_factory=list)  # Next 3 steps
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutiveOrchestrator:
    """
    Async orchestrator for chat-integrated goal execution.
    
    Manages background goal execution tasks and provides status queries
    for chat interface integration.
    """
    
    def __init__(self, executive_system: Optional[ExecutiveSystem] = None):
        """
        Initialize orchestrator.
        
        Args:
            executive_system: ExecutiveSystem instance (creates new if None)
        """
        self.executive_system = executive_system or ExecutiveSystem()
        self.summarizer = create_summarizer()  # For humanizing plans/schedules
        
        # Active execution tracking
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_progress: Dict[str, ExecutionProgress] = {}
        
        # Completed execution history (keep last 100)
        self.completed_executions: List[str] = []
        self.max_history = 100
        
        logger.info("ExecutiveOrchestrator initialized")
    
    async def execute_goal_async(
        self,
        goal_id: str,
        initial_state: Optional[WorldState] = None,
        auto_start: bool = True
    ) -> ExecutionProgress:
        """
        Start asynchronous goal execution in background.
        
        Args:
            goal_id: ID of goal to execute
            initial_state: Optional initial world state
            auto_start: If False, queue but don't start (for batching)
            
        Returns:
            ExecutionProgress with initial status
            
        Raises:
            ValueError: If goal not found or already executing
        """
        # Validate goal exists
        goal = self.executive_system.goal_manager.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")
        
        # Check if already executing
        if goal_id in self.active_executions:
            logger.warning(f"Goal {goal_id} already executing")
            return self.execution_progress[goal_id]
        
        # Create progress tracker
        progress = ExecutionProgress(
            goal_id=goal_id,
            goal_title=goal.title,
            phase=ExecutionPhase.QUEUED
        )
        self.execution_progress[goal_id] = progress
        
        # Start background task
        if auto_start:
            task = asyncio.create_task(self._execute_goal_task(goal_id, initial_state))
            self.active_executions[goal_id] = task
            logger.info(f"Started async execution for goal {goal_id}")
        
        return progress
    
    async def _execute_goal_task(
        self,
        goal_id: str,
        initial_state: Optional[WorldState]
    ):
        """
        Background task for goal execution.
        
        Args:
            goal_id: Goal to execute
            initial_state: Initial world state
        """
        progress = self.execution_progress[goal_id]
        
        try:
            # Phase 1: Decision
            progress.phase = ExecutionPhase.DECIDING
            progress.recent_steps.append("Making decision...")
            await asyncio.sleep(0)  # Yield control
            
            # Phase 2: Planning
            progress.phase = ExecutionPhase.PLANNING
            progress.recent_steps.append("Creating plan...")
            await asyncio.sleep(0)
            
            # Phase 3: Scheduling
            progress.phase = ExecutionPhase.SCHEDULING
            progress.recent_steps.append("Building schedule...")
            await asyncio.sleep(0)
            
            # Execute synchronously (ExecutiveSystem.execute_goal is sync)
            context = await asyncio.to_thread(
                self.executive_system.execute_goal,
                goal_id,
                initial_state
            )
            
            # Store plan and schedule in metadata for status formatting
            if context.plan:
                progress.metadata["plan"] = context.plan
            if context.schedule:
                progress.metadata["schedule"] = context.schedule
            
            # Phase 4: Execution simulation (mock for now)
            progress.phase = ExecutionPhase.EXECUTING
            progress.total_actions = context.total_actions
            
            # Simulate step execution
            for i in range(context.total_actions):
                if context.plan and i < len(context.plan.steps):
                    step = context.plan.steps[i]
                    action_name = step.action.name
                    progress.current_action = self.summarizer.humanize_action(action_name)
                    progress.actions_completed = i + 1
                    progress.progress_percent = ((i + 1) / context.total_actions) * 100
                    
                    # Update recent steps (keep last 3)
                    humanized = self.summarizer.humanize_action(action_name)
                    progress.recent_steps.append(f"Completed: {humanized}")
                    if len(progress.recent_steps) > 3:
                        progress.recent_steps = progress.recent_steps[-3:]
                    
                    # Update next steps
                    progress.next_steps = [
                        self.summarizer.humanize_action(context.plan.steps[j].action.name)
                        for j in range(i + 1, min(i + 4, len(context.plan.steps)))
                    ]
                    
                    # Simulate work
                    await asyncio.sleep(0.5)  # Mock execution time
            
            # Completion
            progress.phase = ExecutionPhase.COMPLETED
            progress.progress_percent = 100.0
            progress.actual_completion = datetime.now()
            progress.success = True
            
            # Update goal status
            goal = self.executive_system.goal_manager.get_goal(goal_id)
            if goal:
                goal.status = GoalStatus.COMPLETED
            
            logger.info(f"Goal {goal_id} execution completed successfully")
            
        except Exception as e:
            logger.error(f"Goal {goal_id} execution failed: {e}", exc_info=True)
            progress.phase = ExecutionPhase.FAILED
            progress.error_message = str(e)
            progress.success = False
            
            # Update goal status
            goal = self.executive_system.goal_manager.get_goal(goal_id)
            if goal:
                goal.status = GoalStatus.FAILED
        
        finally:
            # Move to history
            self.active_executions.pop(goal_id, None)
            self.completed_executions.append(goal_id)
            if len(self.completed_executions) > self.max_history:
                oldest = self.completed_executions.pop(0)
                self.execution_progress.pop(oldest, None)
    
    def get_execution_status(self, goal_id: str) -> Optional[ExecutionProgress]:
        """
        Get current execution status for a goal.
        
        Args:
            goal_id: Goal ID to query
            
        Returns:
            ExecutionProgress if found, None otherwise
        """
        return self.execution_progress.get(goal_id)
    
    def list_active_executions(self) -> List[ExecutionProgress]:
        """
        List all currently executing goals.
        
        Returns:
            List of active execution progress
        """
        return [
            self.execution_progress[goal_id]
            for goal_id in self.active_executions.keys()
            if goal_id in self.execution_progress
        ]
    
    def format_status_for_chat(self, goal_id: str) -> str:
        """
        Format execution status as natural language for chat response.
        
        Args:
            goal_id: Goal to format status for
            
        Returns:
            Human-readable status message
        """
        progress = self.get_execution_status(goal_id)
        if not progress:
            return f"No execution found for goal {goal_id}"
        
        # Build status message
        lines = [f"**{progress.goal_title}**"]
        
        # Phase and progress
        if progress.phase == ExecutionPhase.COMPLETED:
            lines.append("✅ **Status**: Completed")
            if progress.actual_completion:
                lines.append(f"📅 Finished: {progress.actual_completion.strftime('%I:%M %p')}")
        
        elif progress.phase == ExecutionPhase.FAILED:
            lines.append("❌ **Status**: Failed")
            if progress.error_message:
                lines.append(f"⚠️ Error: {progress.error_message}")
        
        elif progress.phase == ExecutionPhase.PLANNING:
            lines.append(f"🔄 **Status**: Creating plan...")
            
            # If we have a plan in metadata, show summary
            if "plan" in progress.metadata:
                plan_summary = self.summarizer.humanize_plan(progress.metadata["plan"])
                lines.append("")
                lines.append(plan_summary.brief)
        
        elif progress.phase in [ExecutionPhase.EXECUTING, ExecutionPhase.SCHEDULING]:
            lines.append(f"🔄 **Status**: {progress.phase.value.title()}")
            lines.append(f"📊 Progress: {progress.progress_percent:.0f}% ({progress.actions_completed}/{progress.total_actions} steps)")
            
            if progress.current_action:
                lines.append(f"▶️ Current: {progress.current_action}")
            
            # Recent steps
            if progress.recent_steps:
                lines.append("\n**Recent Steps:**")
                for step in progress.recent_steps[-3:]:
                    lines.append(f"  • {step}")
            
            # Next steps
            if progress.next_steps:
                lines.append("\n**Coming Up:**")
                for step in progress.next_steps[:3]:
                    lines.append(f"  • {step}")
        
        else:
            lines.append(f"⏸️ **Status**: {progress.phase.value.title()}")
        
        # Timing
        elapsed = datetime.now() - progress.start_time
        if elapsed.total_seconds() < 60:
            lines.append(f"⏱️ Running: {int(elapsed.total_seconds())}s")
        else:
            lines.append(f"⏱️ Running: {int(elapsed.total_seconds() / 60)}m")
        
        return "\n".join(lines)
    
    async def cancel_execution(self, goal_id: str) -> bool:
        """
        Cancel an active goal execution.
        
        Args:
            goal_id: Goal to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        task = self.active_executions.get(goal_id)
        if not task:
            return False
        
        # Cancel task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Remove from active executions
        self.active_executions.pop(goal_id, None)
        
        # Update status
        if goal_id in self.execution_progress:
            progress = self.execution_progress[goal_id]
            progress.phase = ExecutionPhase.CANCELLED
            progress.actual_completion = datetime.now()
        
        # Update goal
        goal = self.executive_system.goal_manager.get_goal(goal_id)
        if goal:
            goal.status = GoalStatus.CANCELLED
        
        logger.info(f"Cancelled execution for goal {goal_id}")
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.
        
        Returns:
            Dict with system stats
        """
        active = len(self.active_executions)
        completed = len([
            p for p in self.execution_progress.values()
            if p.phase == ExecutionPhase.COMPLETED
        ])
        failed = len([
            p for p in self.execution_progress.values()
            if p.phase == ExecutionPhase.FAILED
        ])
        
        return {
            "active_executions": active,
            "completed_executions": completed,
            "failed_executions": failed,
            "total_tracked": len(self.execution_progress),
            "success_rate": completed / (completed + failed) if (completed + failed) > 0 else 0.0
        }


# Factory function
def create_orchestrator(executive_system: Optional[ExecutiveSystem] = None) -> ExecutiveOrchestrator:
    """
    Create ExecutiveOrchestrator instance.
    
    Args:
        executive_system: Optional ExecutiveSystem instance
        
    Returns:
        ExecutiveOrchestrator instance
    """
    return ExecutiveOrchestrator(executive_system)
