"""Executive System Integration (Phase 2: Planning/Scheduling Advisor API)

This module provides an integrated *advisor* pipeline:
    Goal → Decision → Plan → Schedule

Phase 2 constraint:
    - No module except the executive core may commit actions.

Accordingly, `ExecutiveSystem` does not actuate tools or execute actions.
It may be used to produce plans/schedules, status, and learning analytics.
Any side effects (e.g., auto-generating reminders) are opt-in.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

# Component imports
from src.executive.goal_manager import GoalManager, Goal, GoalPriority
from src.executive.goals import HTNGoalManagerAdapter
from src.executive.decision_engine import DecisionEngine, DecisionResult
from src.executive.planning.goap_planner import GOAPPlanner, Plan, WorldState, get_metrics_registry
from src.executive.planning.action_library import create_default_action_library
from src.executive.scheduling import DynamicScheduler, Schedule
from src.executive.stages import (
    ExecutiveDecisionStage,
    ExecutivePlanningStage,
    ExecutiveReminderStage,
    ExecutiveReportingStage,
    ExecutiveSchedulingStage,
)

# Type checking imports (Week 16)
if TYPE_CHECKING:
    from src.executive.learning.outcome_tracker import OutcomeRecord

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution workflow."""
    IDLE = "idle"
    PLANNING = "planning"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionContext:
    """
    Context for a complete execution workflow from goal to completion.
    
    Tracks the full pipeline: Goal → Decision → Plan → Schedule → Execution
    """
    # Source goal
    goal_id: str
    goal_title: str
    
    # Pipeline stages
    decision_result: Optional[DecisionResult] = None
    plan: Optional[Plan] = None
    schedule: Optional[Schedule] = None
    
    # Status tracking
    status: ExecutionStatus = ExecutionStatus.IDLE
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Progress tracking
    actions_completed: int = 0
    total_actions: int = 0
    current_action: Optional[str] = None
    
    # Metrics
    decision_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    scheduling_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Feedback
    success: bool = False
    failure_reason: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Outcome tracking (Week 16)
    actual_completion_time: Optional[datetime] = None
    actual_success: Optional[bool] = None  # Did goal actually succeed?
    outcome_score: float = 0.0  # Quality score 0-1
    deviations: List[str] = field(default_factory=list)  # Issues encountered
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)  # Predicted vs actual
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for executive system integration."""
    # Component toggles
    enable_enhanced_decisions: bool = True
    enable_goap_planning: bool = True
    enable_dynamic_scheduling: bool = True
    
    # Decision settings
    decision_strategy: str = "weighted_scoring"  # or 'ahp', 'pareto'
    decision_timeout_ms: float = 5000.0
    
    # Planning settings
    planning_max_iterations: int = 1000
    planning_timeout_ms: float = 10000.0
    use_planning_constraints: bool = True
    
    # Scheduling settings
    scheduling_horizon_hours: float = 24.0
    scheduling_timeout_s: int = 30
    enable_proactive_warnings: bool = True
    
    # Execution settings
    auto_replan_on_failure: bool = True
    max_replan_attempts: int = 3

    # Side effects (advisor-only defaults)
    create_reminders_from_schedule: bool = False

    # HTN integration
    enable_htn_goals: bool = True
    
    # Performance settings
    enable_telemetry: bool = True
    profile_performance: bool = True


class ExecutiveSystem:
    """
    Integrated planning/scheduling advisor.

    Produces decision recommendations, GOAP plans, and schedules for a goal.
    Does not execute actions or actuate tools.
    
    Usage:
        system = ExecutiveSystem()
        
        # Create goal
        goal_id = system.goal_manager.create_goal("Complete project", priority=GoalPriority.HIGH)
        
        # Plan + schedule goal (integrated pipeline)
        context = system.plan_goal(goal_id)
        
        # Monitor progress
        status = system.get_execution_status(goal_id)
        health = system.get_system_health()
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        Initialize executive system with all components.
        
        Args:
            config: Integration configuration
        """
        self.config = config or IntegrationConfig()
        self.metrics = get_metrics_registry()
        
        # Initialize components
        logger.info("Initializing ExecutiveSystem components...")
        
        self.goal_manager = GoalManager(max_active_goals=20)
        self.htn_adapter = HTNGoalManagerAdapter(self.goal_manager) if self.config.enable_htn_goals else None
        self.decision_engine = DecisionEngine()
        
        # GOAP planner with default action library
        self.action_library = create_default_action_library()
        self.goap_planner = GOAPPlanner(
            action_library=self.action_library,
            constraints=[] if not self.config.use_planning_constraints else None
        )
        
        # Dynamic scheduler
        self.scheduler = DynamicScheduler()
        self.decision_stage = ExecutiveDecisionStage(
            config=self.config,
            decision_engine=self.decision_engine,
            metrics=self.metrics,
        )
        self.planning_stage = ExecutivePlanningStage(
            config=self.config,
            goap_planner=self.goap_planner,
            metrics=self.metrics,
        )
        self.scheduling_stage = ExecutiveSchedulingStage(
            config=self.config,
            scheduler=self.scheduler,
            metrics=self.metrics,
        )
        self.reporting_stage = ExecutiveReportingStage(
            get_goal_manager=lambda: self.goal_manager,
            get_execution_contexts=lambda: self.execution_contexts,
            get_active_schedules=lambda: self.active_schedules,
            get_outcome_tracker=lambda: self.outcome_tracker,
        )
        self.reminder_stage = ExecutiveReminderStage()
        
        # Outcome tracking (Week 16)
        from src.executive.learning.outcome_tracker import OutcomeTracker
        self.outcome_tracker = OutcomeTracker()
        
        # Execution tracking
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.active_schedules: Dict[str, Schedule] = {}
        
        logger.info("ExecutiveSystem initialized successfully")
        self.metrics.inc("executive_system_init_total")

    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        target_date: Optional[datetime] = None,
        success_criteria: Optional[List[str]] = None,
        resources_needed: Optional[List[str]] = None,
        *,
        use_htn: Optional[bool] = None,
        compound: bool = False,
        preconditions: Optional[Dict[str, Any]] = None,
        postconditions: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Create a goal, optionally via HTN decomposition."""
        if use_htn is None:
            use_htn = self.htn_adapter is not None

        if use_htn and self.htn_adapter is not None:
            htn_priority = self._htn_priority_from_legacy(priority)
            if compound:
                result = self.htn_adapter.create_compound_goal(
                    description=description or title,
                    priority=htn_priority,
                    preconditions=preconditions,
                    postconditions=postconditions,
                    deadline=target_date,
                    dependencies=dependencies,
                    current_state={},
                )
                if result.goals:
                    return result.goals[0].id
                return ""

            return self.htn_adapter.create_primitive_goal(
                description=description or title,
                priority=htn_priority,
                preconditions=preconditions,
                postconditions=postconditions,
                deadline=target_date,
                dependencies=dependencies,
                parent_id=parent_id,
            )

        return self.goal_manager.create_goal(
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            target_date=target_date,
            success_criteria=success_criteria,
            resources_needed=resources_needed,
        )

    def _htn_priority_from_legacy(self, priority: GoalPriority) -> int:
        mapping = {
            GoalPriority.LOW: 2,
            GoalPriority.MEDIUM: 5,
            GoalPriority.HIGH: 7,
            GoalPriority.URGENT: 8,
            GoalPriority.CRITICAL: 10,
        }
        return mapping.get(priority, 5)
    
    def execute_goal(self, goal_id: str, initial_state: Optional[WorldState] = None) -> ExecutionContext:
        """Backward-compatible alias for :meth:`plan_goal`.

        This method name is legacy; it does not execute actions.
        """
        return self.plan_goal(goal_id, initial_state=initial_state)

    def plan_goal(
        self,
        goal_id: str,
        initial_state: Optional[WorldState] = None,
        *,
        create_reminders: Optional[bool] = None,
    ) -> ExecutionContext:
        """
        Plan and schedule a goal: Decision → Plan → Schedule.

        Advisor-only: produces artifacts but does not commit actions.
        
        Args:
            goal_id: ID of goal to execute
            initial_state: Optional initial world state (defaults to empty)
            
        Returns:
            ExecutionContext with pipeline results
            
        Raises:
            ValueError: If goal not found or invalid
        """
        start_time = datetime.now()
        
        # Get goal
        goal = self.goal_manager.get_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")
        
        logger.info(f"Planning goal pipeline: {goal.title} (priority={goal.priority.value})")
        
        # Create execution context
        context = ExecutionContext(
            goal_id=goal_id,
            goal_title=goal.title,
            status=ExecutionStatus.PLANNING
        )
        self.execution_contexts[goal_id] = context
        
        try:
            # Stage 1: Decision Making
            logger.info("Stage 1: Decision making...")
            decision_start = datetime.now()
            
            decision_result = self._make_goal_decision(goal)
            context.decision_result = decision_result
            context.decision_time_ms = (datetime.now() - decision_start).total_seconds() * 1000
            
            # Stage 2: Planning
            logger.info("Stage 2: GOAP planning...")
            planning_start = datetime.now()
            
            plan = self._create_goal_plan(goal, decision_result, initial_state)
            if not plan:
                context.status = ExecutionStatus.FAILED
                context.failure_reason = "Planning failed - no valid plan found"
                return context
            
            context.plan = plan
            context.total_actions = len(plan.steps)
            context.planning_time_ms = (datetime.now() - planning_start).total_seconds() * 1000
            
            # Stage 3: Scheduling
            logger.info("Stage 3: Dynamic scheduling...")
            context.status = ExecutionStatus.SCHEDULING
            scheduling_start = datetime.now()
            
            schedule = self._create_schedule_from_plan(plan, goal)
            if not schedule or not schedule.is_feasible:
                context.status = ExecutionStatus.FAILED
                context.failure_reason = "Scheduling failed - infeasible schedule"
                return context
            
            context.schedule = schedule
            context.scheduling_time_ms = (datetime.now() - scheduling_start).total_seconds() * 1000
            self.active_schedules[goal_id] = schedule
            
            # Stage 4 (optional side effect): Auto-generate reminders from scheduled tasks
            do_reminders = self.config.create_reminders_from_schedule if create_reminders is None else bool(create_reminders)
            if do_reminders:
                logger.info("Stage 4: Creating reminders from plan (opt-in)...")
                reminder_count = self._create_reminders_from_schedule(goal_id, schedule, plan)
                if reminder_count > 0:
                    logger.info(f"  - Created {reminder_count} reminders for plan steps")
            
            # Mark as ready (advisor-only: does not execute actions)
            context.status = ExecutionStatus.EXECUTING
            context.total_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            context.metadata["advisor_only"] = True
            context.metadata["ready_to_execute"] = True
            
            logger.info(f"Goal {goal.title} planned + scheduled (ready to execute by executive core):")
            logger.info(f"  - Decision time: {context.decision_time_ms:.1f}ms")
            logger.info(f"  - Planning time: {context.planning_time_ms:.1f}ms")
            logger.info(f"  - Scheduling time: {context.scheduling_time_ms:.1f}ms")
            logger.info(f"  - Actions: {len(plan.steps)}")
            logger.info(f"  - Schedule makespan: {schedule.makespan}")
            
            # Update metrics
            self.metrics.inc("executive_goal_executions_total")
            self.metrics.observe("executive_pipeline_latency_ms", context.total_time_ms)
            
            return context
            
        except Exception as e:
            logger.error(f"Goal execution failed: {e}", exc_info=True)
            context.status = ExecutionStatus.FAILED
            context.failure_reason = str(e)
            self.metrics.inc("executive_goal_failures_total")
            return context
    
    def _make_goal_decision(self, goal: Goal) -> DecisionResult:
        """
        Make decision on how to approach goal.
        
        Currently uses a simple template-based approach.
        Future: Use enhanced decision strategies based on goal context.
        
        Args:
            goal: Goal to make decision for
            
        Returns:
            Decision result
        """
        return self.decision_stage.make_goal_decision(goal)
    
    def _create_goal_plan(self, goal: Goal, decision: DecisionResult,
                         initial_state: Optional[WorldState] = None) -> Optional[Plan]:
        """
        Create GOAP plan to achieve goal.
        
        Args:
            goal: Goal to plan for
            decision: Decision result influencing planning
            initial_state: Starting world state
            
        Returns:
            Plan or None if planning failed
        """
        return self.planning_stage.create_goal_plan(goal, decision, initial_state)
    
    def _goal_to_world_state(self, goal: Goal) -> WorldState:
        """
        Convert goal to GOAP world state representation.
        
        Args:
            goal: Goal to convert
            
        Returns:
            WorldState representing goal achievement
        """
        return self.planning_stage.goal_to_world_state(goal)
    
    def _create_schedule_from_plan(self, plan: Plan, goal: Goal) -> Optional[Schedule]:
        """
        Convert GOAP plan to scheduling problem and create schedule.
        
        Args:
            plan: GOAP plan to schedule
            goal: Associated goal
            
        Returns:
            Schedule or None if scheduling failed
        """
        return self.scheduling_stage.create_schedule_from_plan(plan, goal)
    
    def _goal_priority_to_float(self, priority: GoalPriority) -> float:
        """Convert goal priority to float score."""
        return self.scheduling_stage.goal_priority_to_float(priority)
    
    def _create_reminders_from_schedule(
        self, 
        goal_id: str, 
        schedule: Schedule, 
        plan: Plan
    ) -> int:
        """
        Auto-generate prospective memory reminders from scheduled tasks.
        
        Bridges the executive planning system with prospective memory,
        ensuring plan steps become actionable reminders.
        
        Args:
            goal_id: Associated goal ID
            schedule: The computed schedule
            plan: The GOAP plan (for action descriptions)
            
        Returns:
            Number of reminders created
        """
        return self.reminder_stage.create_reminders_from_schedule(goal_id, schedule, plan)

    def get_execution_status(self, goal_id: str) -> Optional[ExecutionContext]:
        """Get execution context for a goal."""
        return self.execution_contexts.get(goal_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            Dictionary with health indicators
        """
        return self.reporting_stage.get_system_health(
            executing_status=ExecutionStatus.EXECUTING,
            failed_status=ExecutionStatus.FAILED,
        )
    
    def clear_execution_history(self, goal_id: Optional[str] = None):
        """
        Clear execution history.
        
        Args:
            goal_id: If provided, clear only this goal's history.
                    Otherwise clear all history.
        """
        self.reporting_stage.clear_execution_history(goal_id)
    
    # Outcome tracking methods (Week 16)
    
    def complete_goal_execution(
        self,
        goal_id: str,
        success: bool,
        outcome_score: Optional[float] = None,
        deviations: Optional[List[str]] = None,
    ) -> Optional['OutcomeRecord']:
        """
        Mark a goal execution as complete and record outcome.
        
        Args:
            goal_id: Goal to complete
            success: Whether goal was successfully achieved
            outcome_score: Quality score 0-1 (optional, auto-calculated)
            deviations: List of issues encountered (optional)
        
        Returns:
            OutcomeRecord if context exists, None otherwise
        """
        context = self.execution_contexts.get(goal_id)
        if not context:
            logger.warning(f"Cannot complete goal {goal_id}: no execution context")
            return None
        
        # Update context
        context.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        context.end_time = datetime.now()
        context.actual_completion_time = datetime.now()
        context.actual_success = success
        
        if deviations:
            context.deviations.extend(deviations)
        
        # Calculate total time
        context.total_time_ms = (context.end_time - context.start_time).total_seconds() * 1000
        
        # Record outcome
        outcome = self.outcome_tracker.record_outcome(
            context,
            outcome_score=outcome_score,
            deviations=deviations,
        )
        
        logger.info(
            f"Completed goal '{context.goal_title}': "
            f"success={success}, score={outcome.outcome_score:.2f}"
        )
        
        return outcome
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get learning and outcome metrics.
        
        Returns:
            Dict with decision/planning/scheduling accuracy metrics
        """
        return self.reporting_stage.get_learning_metrics()
