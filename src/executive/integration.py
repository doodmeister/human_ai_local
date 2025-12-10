"""
Executive System Integration - Week 15

Unified orchestration layer connecting all executive components:
- GoalManager: Hierarchical goal tracking
- DecisionEngine: Multi-criteria decision making (AHP, Pareto)
- GOAPPlanner: Goal-oriented action planning
- DynamicScheduler: Constraint-based scheduling with adaptation

Architecture:
    Goal → Decision → Plan → Schedule → Execution → Feedback

Integration flow:
1. Goals define what to achieve (GoalManager)
2. Decisions determine how to approach goals (DecisionEngine)
3. Plans break down decisions into action sequences (GOAPPlanner)
4. Schedules organize actions with resources and constraints (DynamicScheduler)
5. Execution feedback updates progress and triggers replanning
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import defaultdict
import logging

# Component imports
from src.executive.goal_manager import GoalManager, Goal, GoalStatus, GoalPriority
from src.executive.decision_engine import DecisionEngine, DecisionOption, DecisionResult
from src.executive.planning.goap_planner import GOAPPlanner, Plan, WorldState, get_metrics_registry
from src.executive.planning.action_library import ActionLibrary, create_default_action_library
from src.executive.scheduling import (
    DynamicScheduler, SchedulingProblem, Schedule,
    Task, Resource, ResourceType, OptimizationObjective
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
    
    # Performance settings
    enable_telemetry: bool = True
    profile_performance: bool = True


class ExecutiveSystem:
    """
    Unified executive control system integrating goals, decisions, planning, and scheduling.
    
    Provides end-to-end orchestration from high-level goals to executable schedules,
    with monitoring, adaptation, and learning capabilities.
    
    Usage:
        system = ExecutiveSystem()
        
        # Create goal
        goal_id = system.goal_manager.create_goal("Complete project", priority=GoalPriority.HIGH)
        
        # Execute goal (integrated pipeline)
        context = system.execute_goal(goal_id)
        
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
        self.decision_engine = DecisionEngine()
        
        # GOAP planner with default action library
        self.action_library = create_default_action_library()
        self.goap_planner = GOAPPlanner(
            action_library=self.action_library,
            constraints=[] if not self.config.use_planning_constraints else None
        )
        
        # Dynamic scheduler
        self.scheduler = DynamicScheduler()
        
        # Outcome tracking (Week 16)
        from src.executive.learning.outcome_tracker import OutcomeTracker
        self.outcome_tracker = OutcomeTracker()
        
        # Execution tracking
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.active_schedules: Dict[str, Schedule] = {}
        
        logger.info("ExecutiveSystem initialized successfully")
        self.metrics.inc("executive_system_init_total")
    
    def execute_goal(self, goal_id: str, initial_state: Optional[WorldState] = None) -> ExecutionContext:
        """
        Execute complete pipeline for a goal: Decision → Plan → Schedule.
        
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
        
        logger.info(f"Executing goal: {goal.title} (priority={goal.priority.value})")
        
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
            
            # Stage 4: Auto-generate reminders from scheduled tasks
            logger.info("Stage 4: Creating reminders from plan...")
            reminder_count = self._create_reminders_from_schedule(goal_id, schedule, plan)
            if reminder_count > 0:
                logger.info(f"  - Created {reminder_count} reminders for plan steps")
            
            # Mark as ready for execution
            context.status = ExecutionStatus.EXECUTING
            context.total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Goal {goal.title} ready for execution:")
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
        # Create decision options for approaching the goal
        options = [
            DecisionOption(
                name="direct_approach",
                description="Direct approach - tackle goal immediately",
                data={"approach": "direct", "risk": 0.3}
            ),
            DecisionOption(
                name="incremental_approach",
                description="Incremental approach - break into smaller steps",
                data={"approach": "incremental", "risk": 0.2}
            ),
            DecisionOption(
                name="parallel_approach",
                description="Parallel approach - work on multiple aspects simultaneously",
                data={"approach": "parallel", "risk": 0.4}
            )
        ]
        
        # Make decision using selected strategy
        result = self.decision_engine.make_decision(
            options=options,
            criteria=self.decision_engine.criterion_templates.get('task_selection', []),
            strategy=self.config.decision_strategy,
            context={"goal_id": goal.id, "goal_priority": goal.priority.value}
        )
        
        self.metrics.inc("executive_decisions_made_total")
        return result
    
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
        # Convert goal to GOAP goal state
        goal_state = self._goal_to_world_state(goal)
        
        # Use provided initial state or create default
        if initial_state is None:
            initial_state = WorldState()
        
        # Plan with GOAP
        plan = self.goap_planner.plan(
            initial_state=initial_state,
            goal_state=goal_state,
            max_iterations=self.config.planning_max_iterations,
            plan_context={"goal_id": goal.id, "decision": decision}
        )
        
        if plan:
            self.metrics.inc("executive_plans_created_total")
            self.metrics.observe("executive_plan_length", len(plan.steps))
        else:
            self.metrics.inc("executive_planning_failures_total")
        
        return plan
    
    def _goal_to_world_state(self, goal: Goal) -> WorldState:
        """
        Convert goal to GOAP world state representation.
        
        Args:
            goal: Goal to convert
            
        Returns:
            WorldState representing goal achievement
        """
        # Create goal state based on success criteria
        goal_conditions = {}
        
        if goal.success_criteria:
            for i, criterion in enumerate(goal.success_criteria):
                # Parse criterion into state variable
                # Format: "variable=value" or just "variable" (implies True)
                if "=" in criterion:
                    var, val = criterion.split("=", 1)
                    var = var.strip()
                    val = val.strip()
                    
                    # Try to parse value as bool/int/float
                    if val.lower() == "true":
                        goal_conditions[var] = True
                    elif val.lower() == "false":
                        goal_conditions[var] = False
                    elif val.isdigit():
                        goal_conditions[var] = int(val)
                    else:
                        try:
                            goal_conditions[var] = float(val)
                        except ValueError:
                            goal_conditions[var] = val
                else:
                    goal_conditions[criterion.strip()] = True
        else:
            # Default: goal completed
            goal_conditions[f"goal_{goal.id}_completed"] = True
        
        # WorldState takes a dict, not kwargs
        return WorldState(goal_conditions)
    
    def _create_schedule_from_plan(self, plan: Plan, goal: Goal) -> Optional[Schedule]:
        """
        Convert GOAP plan to scheduling problem and create schedule.
        
        Args:
            plan: GOAP plan to schedule
            goal: Associated goal
            
        Returns:
            Schedule or None if scheduling failed
        """
        # Convert plan actions to scheduling tasks
        tasks = []
        for i, step in enumerate(plan.steps):
            action = step.action
            task = Task(
                id=f"action_{i}_{action.name}",
                name=action.name,
                duration=timedelta(hours=1),  # Default 1 hour, could be in action metadata
                priority=self._goal_priority_to_float(goal.priority),
                cognitive_load=0.5,  # Default, could be in action metadata
                dependencies=set([f"action_{i-1}_{plan.steps[i-1].action.name}"] if i > 0 else [])
            )
            tasks.append(task)
        
        # Create default resources
        resources = [
            Resource(
                id="cognitive",
                name="Cognitive Capacity",
                type=ResourceType.COGNITIVE,
                capacity=1.0
            )
        ]
        
        # Create scheduling problem
        horizon_hours = self.config.scheduling_horizon_hours
        problem = SchedulingProblem(
            tasks=tasks,
            resources=resources,
            objectives=[
                OptimizationObjective(
                    name="minimize_makespan",
                    description="Minimize total schedule duration",
                    weight=1.0
                )
            ],
            horizon=timedelta(hours=horizon_hours)
        )
        
        # Schedule
        try:
            schedule = self.scheduler.create_initial_schedule(problem)
            self.metrics.inc("executive_schedules_created_total")
            return schedule
        except Exception as e:
            logger.error(f"Scheduling failed: {e}")
            self.metrics.inc("executive_scheduling_failures_total")
            return None
    
    def _goal_priority_to_float(self, priority: GoalPriority) -> float:
        """Convert goal priority to float score."""
        mapping = {
            GoalPriority.LOW: 0.3,
            GoalPriority.MEDIUM: 0.6,
            GoalPriority.HIGH: 0.9,
            GoalPriority.CRITICAL: 1.0
        }
        return mapping.get(priority, 0.5)
    
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
        try:
            # Lazy import to avoid circular dependencies
            from src.memory.prospective.prospective_memory import (
                create_prospective_memory,
                get_prospective_memory,
            )
            
            # Get or create prospective memory system
            try:
                prospective = get_prospective_memory()
            except Exception:
                prospective = create_prospective_memory(use_vector=False)
            
            if prospective is None:
                logger.warning("Prospective memory not available - skipping reminder creation")
                return 0
            
            # Create action name to description mapping from plan
            action_descriptions = {}
            for i, step in enumerate(plan.steps):
                action_name = getattr(step, 'name', str(step))
                action_descriptions[action_name] = {
                    'description': getattr(step, 'description', action_name),
                    'sequence': i + 1,
                    'total': len(plan.steps)
                }
            
            reminders_created = 0
            base_time = datetime.now()
            
            # Create reminder for each scheduled task
            for task in schedule.tasks:
                # Calculate due time from scheduled start
                if task.scheduled_start is not None:
                    # scheduled_start is typically in minutes from base time
                    due_time = base_time + timedelta(minutes=task.scheduled_start)
                else:
                    # Fallback: spread tasks evenly
                    due_time = base_time + timedelta(hours=reminders_created + 1)
                
                # Get action description if available
                action_info = action_descriptions.get(task.id, {})
                seq = action_info.get('sequence', reminders_created + 1)
                total = action_info.get('total', len(schedule.tasks))
                
                # Create reminder content
                task_name = getattr(task, 'name', task.id)
                content = f"[Step {seq}/{total}] {task_name}"
                
                # Add reminder
                reminder = prospective.add_reminder(
                    content=content,
                    due_time=due_time,
                    tags=["auto-generated", "plan-step", f"goal:{goal_id}"],
                    metadata={
                        "goal_id": goal_id,
                        "task_id": task.id,
                        "sequence": seq,
                        "total_steps": total,
                        "auto_generated": True,
                        "source": "executive_system"
                    }
                )
                
                if reminder:
                    reminders_created += 1
                    logger.debug(f"Created reminder: {content} due at {due_time}")
            
            logger.info(f"Created {reminders_created} reminders for goal {goal_id}")
            return reminders_created
            
        except Exception as e:
            logger.warning(f"Failed to create reminders from schedule: {e}")
            return 0

    def get_execution_status(self, goal_id: str) -> Optional[ExecutionContext]:
        """Get execution context for a goal."""
        return self.execution_contexts.get(goal_id)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            Dictionary with health indicators
        """
        active_goals = len([g for g in self.goal_manager.goals.values() 
                          if g.status == GoalStatus.ACTIVE])
        
        executing_contexts = len([c for c in self.execution_contexts.values()
                                 if c.status == ExecutionStatus.EXECUTING])
        
        failed_contexts = len([c for c in self.execution_contexts.values()
                              if c.status == ExecutionStatus.FAILED])
        
        return {
            "status": "healthy" if failed_contexts == 0 else "degraded",
            "active_goals": active_goals,
            "executing_workflows": executing_contexts,
            "failed_workflows": failed_contexts,
            "total_contexts": len(self.execution_contexts),
            "active_schedules": len(self.active_schedules),
            "components": {
                "goal_manager": "operational",
                "decision_engine": "operational",
                "goap_planner": "operational",
                "scheduler": "operational"
            }
        }
    
    def clear_execution_history(self, goal_id: Optional[str] = None):
        """
        Clear execution history.
        
        Args:
            goal_id: If provided, clear only this goal's history.
                    Otherwise clear all history.
        """
        if goal_id:
            self.execution_contexts.pop(goal_id, None)
            self.active_schedules.pop(goal_id, None)
        else:
            self.execution_contexts.clear()
            self.active_schedules.clear()
    
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
        from src.executive.learning.outcome_tracker import OutcomeRecord
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
        return {
            "decision_accuracy": self.outcome_tracker.analyze_decision_accuracy(),
            "planning_accuracy": self.outcome_tracker.analyze_planning_accuracy(),
            "scheduling_accuracy": self.outcome_tracker.analyze_scheduling_accuracy(),
            "improvement_trends": self.outcome_tracker.get_improvement_trends(),
        }
