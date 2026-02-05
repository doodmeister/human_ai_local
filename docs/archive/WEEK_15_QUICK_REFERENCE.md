# Week 15: Executive System Integration - Quick Reference

## 🚀 Quick Start

```python
from src.executive.integration import ExecutiveSystem
from src.executive.goal_manager import GoalPriority

# 1. Create system
system = ExecutiveSystem()

# 2. Create goal
goal_id = system.goal_manager.create_goal(
    title="Analyze data",
    priority=GoalPriority.HIGH,
    success_criteria=["data_analyzed=True"]
)

# 3. Execute
context = system.execute_goal(goal_id)

# 4. Check results
print(f"Status: {context.status}")
print(f"Plan: {len(context.plan.steps)} steps")
print(f"Time: {context.total_time_ms:.1f}ms")
```

---

## 📦 Import Reference

```python
# Main orchestrator
from src.executive.integration import (
    ExecutiveSystem,
    ExecutionContext,
    ExecutionStatus,
    IntegrationConfig
)

# Component imports
from src.executive.goal_manager import GoalManager, GoalPriority
from src.executive.decision_engine import DecisionEngine, DecisionOption
from src.executive.planning.goap_planner import GOAPPlanner
from src.executive.planning.world_state import WorldState
from src.executive.planning.action_library import create_default_action_library
from src.executive.scheduling.dynamic_scheduler import DynamicScheduler
```

---

## 🎯 Common Patterns

### Pattern 1: Simple Goal Execution

```python
system = ExecutiveSystem()

goal_id = system.goal_manager.create_goal(
    title="Generate report",
    success_criteria=["report_complete=True"]
)

context = system.execute_goal(goal_id)

if context.status == ExecutionStatus.EXECUTING:
    print("Success! Plan created and scheduled")
```

### Pattern 2: With Initial State

```python
from src.executive.planning.world_state import WorldState

initial_state = WorldState({
    "data_available": True,
    "system_ready": True
})

context = system.execute_goal(goal_id, initial_state=initial_state)
```

### Pattern 3: Custom Configuration

```python
from src.executive.integration import IntegrationConfig

config = IntegrationConfig(
    decision_strategy="ahp",           # Use AHP instead of weighted scoring
    planning_max_iterations=500,       # Limit planning iterations
    scheduling_timeout_s=60,           # Allow more time for scheduling
    enable_proactive_warnings=True     # Get early warnings
)

system = ExecutiveSystem(config)
```

### Pattern 4: Monitor Execution

```python
# Get current status
status = system.get_execution_status(goal_id)
print(f"Progress: {status.actions_completed}/{status.total_actions}")

# Get system health
health = system.get_system_health()
print(f"Active goals: {health['active_goals']}")
print(f"Status: {health['status']}")
```

### Pattern 5: Multiple Goals

```python
goal_ids = []

# Create multiple goals
for i in range(3):
    goal_id = system.goal_manager.create_goal(
        title=f"Task {i+1}",
        priority=GoalPriority.MEDIUM,
        success_criteria=[f"task_{i}_done=True"]
    )
    goal_ids.append(goal_id)

# Execute all
for goal_id in goal_ids:
    context = system.execute_goal(goal_id)
    print(f"Goal {goal_id}: {context.status}")
```

---

## 🔧 Configuration Options

### IntegrationConfig Fields

```python
IntegrationConfig(
    # Component toggles
    enable_enhanced_decisions=True,
    enable_goap_planning=True,
    enable_dynamic_scheduling=True,
    
    # Decision settings
    decision_strategy="weighted_scoring",  # or 'ahp', 'pareto'
    decision_timeout_ms=5000.0,
    
    # Planning settings
    planning_max_iterations=1000,
    planning_timeout_ms=10000.0,
    use_planning_constraints=True,
    
    # Scheduling settings
    scheduling_horizon_hours=24.0,
    scheduling_timeout_s=30,
    enable_proactive_warnings=True,
    
    # Execution settings
    auto_replan_on_failure=True,
    max_replan_attempts=3,
    
    # Performance settings
    enable_telemetry=True,
    profile_performance=True
)
```

---

## 📊 Success Criteria Format

### Basic Format

```python
# Boolean (parsed to True/False)
success_criteria=["data_analyzed=True"]
success_criteria=["report_complete=False"]

# Integer
success_criteria=["items_processed=100"]

# Float
success_criteria=["confidence_score=0.95"]

# String
success_criteria=["status=completed"]

# Implicit True (no value)
success_criteria=["task_done"]  # → task_done=True
```

### Multiple Criteria

```python
success_criteria=[
    "data_analyzed=True",
    "report_generated=True",
    "stakeholders_notified=True"
]
```

### Complex Example

```python
success_criteria=[
    "phase_1_complete=True",
    "items_processed=100",
    "quality_score=0.85",
    "status=approved"
]
```

---

## 🎭 ExecutionContext Fields

```python
context = system.execute_goal(goal_id)

# Basic info
context.goal_id          # Goal identifier
context.goal_title       # Goal name
context.status           # ExecutionStatus enum

# Pipeline results
context.decision_result  # DecisionResult object
context.plan            # Plan object (with steps)
context.schedule        # Schedule object

# Progress
context.actions_completed  # Completed actions count
context.total_actions      # Total actions in plan
context.current_action     # Current action name

# Timing (milliseconds)
context.decision_time_ms   # Decision stage
context.planning_time_ms   # Planning stage
context.scheduling_time_ms # Scheduling stage
context.total_time_ms      # Total pipeline time

# Outcome
context.success           # Success boolean
context.failure_reason    # Failure explanation (if any)
context.lessons_learned   # List of lessons
```

---

## 🏥 Health Monitoring

```python
health = system.get_system_health()

health['status']              # 'healthy', 'degraded', 'unhealthy'
health['active_goals']        # Count of active goals
health['executing_workflows'] # Count of executing workflows
health['failed_workflows']    # Count of failed workflows

health['component_health']    # Dict of component statuses
# {
#     'goal_manager': 'ok',
#     'decision_engine': 'ok',
#     'goap_planner': 'ok',
#     'scheduler': 'ok'
# }
```

---

## 📈 Metrics Access

```python
from src.executive.planning.goap_planner import get_metrics_registry

metrics = get_metrics_registry()

# Available counters
# - executive_system_init_total
# - executive_goal_executions_total
# - executive_decisions_made_total
# - executive_plans_created_total
# - executive_schedules_created_total
# - executive_pipeline_failures_total

# Get counter value
count = metrics.get_counter('executive_goal_executions_total')
print(f"Total executions: {count}")
```

---

## 🐛 Error Handling

```python
try:
    context = system.execute_goal(goal_id)
    
    if context.status == ExecutionStatus.FAILED:
        print(f"Execution failed: {context.failure_reason}")
        
        # Check which stage failed
        if context.decision_result is None:
            print("Failed at decision stage")
        elif context.plan is None:
            print("Failed at planning stage")
        elif context.schedule is None:
            print("Failed at scheduling stage")
            
except ValueError as e:
    print(f"Goal not found: {e}")
```

---

## 🧪 Testing

### Run All Integration Tests

```bash
pytest tests/test_integration_week15.py -v
```

### Run Specific Test Class

```bash
# Initialization tests
pytest tests/test_integration_week15.py::TestExecutiveSystemInitialization -v

# Pipeline tests
pytest tests/test_integration_week15.py::TestPipelineStages -v

# Health monitoring tests
pytest tests/test_integration_week15.py::TestSystemHealth -v
```

### Run Single Test

```bash
pytest tests/test_integration_week15.py::TestGoalExecution::test_simple_goal_execution -v
```

---

## 🔍 Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src.executive.integration')
logger.setLevel(logging.DEBUG)
```

### Inspect Pipeline Stages

```python
context = system.execute_goal(goal_id)

# Check decision
if context.decision_result:
    print(f"Chosen: {context.decision_result.recommended_option.name}")
    print(f"Confidence: {context.decision_result.confidence}")

# Check plan
if context.plan:
    for i, step in enumerate(context.plan.steps):
        print(f"Step {i+1}: {step.action.name}")
        print(f"  Cost: {step.cost}")
        print(f"  Before: {step.state_before.state}")
        print(f"  After: {step.state_after.state}")

# Check schedule
if context.schedule:
    print(f"Makespan: {context.schedule.makespan}")
    print(f"Feasible: {context.schedule.is_feasible}")
    for task in context.schedule.tasks:
        print(f"Task: {task.id}")
        print(f"  Start: {task.scheduled_start}")
        print(f"  End: {task.scheduled_end}")
```

### Test Components Individually

```python
# Test decision engine
goal = system.goal_manager.get_goal(goal_id)
decision = system._make_goal_decision(goal)
print(f"Decision: {decision.recommended_option.name}")

# Test planning
goal_state = system._goal_to_world_state(goal)
plan = system._create_goal_plan(goal, decision, initial_state=None)
print(f"Plan steps: {len(plan.steps) if plan else 0}")

# Test scheduling
if plan:
    schedule = system._create_schedule_from_plan(plan, goal)
    print(f"Schedule feasible: {schedule.is_feasible if schedule else False}")
```

---

## 📚 Related Documentation

- [Week 15 Completion Summary](WEEK_15_COMPLETION_SUMMARY.md) - Full documentation
- [Week 14 Completion Summary](WEEK_14_COMPLETION_SUMMARY.md) - Dynamic scheduling
- [Week 12 Completion Summary](WEEK_12_COMPLETION_SUMMARY.md) - CP-SAT scheduling
- [Phase 2 Final Complete](PHASE_2_FINAL_COMPLETE.md) - GOAP planning
- [Executive Refactoring Plan](executive_refactoring_plan.md) - Overall roadmap

---

## 🎓 Learning Path

1. **Start Simple**: Run the quick start example
2. **Add State**: Try providing initial WorldState
3. **Customize Config**: Experiment with different strategies
4. **Monitor Health**: Use health monitoring
5. **Debug Pipeline**: Inspect each stage individually
6. **Multiple Goals**: Execute concurrent goals
7. **Read Tests**: Review test_integration_week15.py for patterns

---

## ⚠️ Known Issues

1. **Timing Tests**: Some tests fail on `planning_time_ms > 0` (execution too fast)
2. **No Execution**: Pipeline plans but doesn't execute actions yet
3. **Synchronous Only**: No async/concurrent execution yet
4. **Limited Actions**: Only 10 predefined actions (extensible)

---

## 🔜 Coming in Week 16

- Outcome tracking for completed goals
- ML learning from execution history
- Feature extraction from pipeline data
- A/B testing framework for strategies
- Model persistence and versioning

---

**Questions? Issues?** See [WEEK_15_COMPLETION_SUMMARY.md](WEEK_15_COMPLETION_SUMMARY.md) for detailed information.
