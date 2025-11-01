# GOAP Planning System - Quick Reference

## Core Components

### WorldState
```python
from src.executive.planning import WorldState

# Create state
state = WorldState({"key1": value1, "key2": value2})

# Access (immutable)
value = state.get("key1")              # Get single value
all_state = state.state                # Get all state (read-only)

# Modify (returns new state)
new_state = state.set("key3", value3)  # Add/update single key
new_state = state.update({"k1": v1, "k2": v2})  # Update multiple

# Compare
satisfies = state.satisfies(goal)      # Check if state meets goal
delta = state.delta(other_state)       # Get differences
distance = state.distance_to(goal)     # Count unsatisfied goals
```

### Action
```python
from src.executive.planning import Action, WorldState

action = Action(
    name="action_name",
    preconditions=WorldState({"req1": True}),  # Required state
    effects=WorldState({"result": True}),      # Resulting state
    cost=10.0,                                 # Execution cost
    cognitive_load=5.0,                        # Mental effort (optional)
    resources=["time", "compute"],             # Resources needed (optional)
    duration=300                               # Seconds (optional)
)

# Check if action can be applied
can_apply = action.is_applicable(current_state)

# Apply action
new_state = action.apply(current_state)

# Get cost between states
cost = action.get_cost(from_state, to_state)
```

### ActionLibrary
```python
from src.executive.planning import ActionLibrary, create_default_action_library

# Create library
library = ActionLibrary()

# Add actions
library.add_action(action1)
library.add_action(action2)

# Get applicable actions
applicable = library.get_applicable_actions(current_state)

# Use default library (10 predefined actions)
library = create_default_action_library()
```

### GOAPPlanner
```python
from src.executive.planning import GOAPPlanner

planner = GOAPPlanner(
    action_library=library,
    heuristic='goal_distance'  # or 'weighted_goal_distance', 'relaxed_plan'
)

plan = planner.plan(
    initial_state=initial,
    goal_state=goal,
    max_iterations=1000  # Optional, default 1000
)

if plan:
    # Plan found
    plan.steps                    # List[PlanStep]
    plan.total_cost               # float
    plan.nodes_expanded           # int
    plan.planning_time_ms         # float
    plan.initial_state            # WorldState
    plan.goal_state               # WorldState
else:
    # No plan found
    pass
```

### Plan & PlanStep
```python
# Plan structure
for step in plan.steps:
    step.action                   # Action
    step.state_before             # WorldState
    step.state_after              # WorldState
    step.cost                     # float
    step.step_number              # int
```

## Heuristics

```python
from src.executive.planning.heuristics import (
    get_heuristic,
    CompositeHeuristic
)

# Built-in heuristics
h = get_heuristic('goal_distance')           # Simple count
h = get_heuristic('weighted_goal_distance')  # Priority-based
h = get_heuristic('relaxed_plan')            # Ignores preconditions
h = get_heuristic('zero')                    # Dijkstra's algorithm
h = get_heuristic('max')                     # Max of all heuristics

# Composite heuristic
h = CompositeHeuristic(
    heuristics=[
        get_heuristic('goal_distance'),
        get_heuristic('weighted_goal_distance')
    ],
    mode='max'  # or 'average', 'weighted'
)
```

### Weighted Goals
```python
# Prefix goals with priority markers for weighted_goal_distance heuristic
goal = WorldState({
    "critical_security_check": True,   # Weight: 3.0
    "important_validation": True,      # Weight: 2.0
    "optional_logging": True           # Weight: 1.0
})
```

## Default Actions

The default action library includes 10 predefined actions:

| Action | Preconditions | Effects | Cost |
|--------|---------------|---------|------|
| gather_data | - | has_data: True | 10.0 |
| analyze_data | has_data: True | has_analysis: True | 10.0 |
| create_document | has_analysis: True | has_document: True | 10.0 |
| draft_outline | has_analysis: True | has_outline: True | 5.0 |
| send_notification | has_document: True | notification_sent: True | 3.0 |
| schedule_meeting | has_outline: True | meeting_scheduled: True | 5.0 |
| review_work | has_document: True | reviewed: True | 8.0 |
| run_tests | has_document: True | tests_passed: True | 7.0 |
| create_plan | has_analysis: True | has_plan: True | 8.0 |
| break_down_goal | has_analysis: True | subtasks_defined: True | 12.0 |

## Common Patterns

### Basic Planning
```python
from src.executive.planning import (
    WorldState, GOAPPlanner, create_default_action_library
)

initial = WorldState({"has_data": False})
goal = WorldState({"has_document": True})

planner = GOAPPlanner(
    action_library=create_default_action_library(),
    heuristic='goal_distance'
)

plan = planner.plan(initial, goal)
```

### Plan Execution
```python
current_state = plan.initial_state

for step in plan.steps:
    # Verify preconditions
    if not current_state.satisfies(step.action.preconditions):
        print(f"Failed: {step.action.name}")
        break
    
    # Execute action
    current_state = step.action.apply(current_state)
    print(f"Executed: {step.action.name}")

# Check goal
if current_state.satisfies(plan.goal_state):
    print("Success!")
```

### Custom Actions
```python
from src.executive.planning import Action, WorldState

custom_action = Action(
    name="my_action",
    preconditions=WorldState({"ready": True}),
    effects=WorldState({"done": True}),
    cost=5.0
)

library.add_action(custom_action)
```

### Fallback Strategy
```python
# Try best heuristic first, fall back if needed
heuristics = ['relaxed_plan', 'weighted_goal_distance', 'goal_distance']

for h_name in heuristics:
    planner = GOAPPlanner(library, heuristic=h_name)
    plan = planner.plan(initial, goal)
    if plan:
        break
```

### Conditional Planning
```python
# Choose heuristic based on conditions
if time_pressure:
    heuristic = 'goal_distance'  # Fast
    max_iter = 500
else:
    heuristic = 'relaxed_plan'   # Better quality
    max_iter = 2000

planner = GOAPPlanner(library, heuristic=heuristic)
plan = planner.plan(initial, goal, max_iterations=max_iter)
```

## Performance Guidelines

### Heuristic Selection
- **goal_distance**: Fast, simple problems
- **weighted_goal_distance**: Fast, prioritized goals
- **relaxed_plan**: Slower, complex dependencies
- **zero**: Slowest, guaranteed optimal
- **max**: Conservative, multiple heuristics

### Iteration Limits
- Simple (2-3 steps): 100-500 iterations
- Medium (4-6 steps): 500-1000 iterations
- Complex (7+ steps): 1000-5000 iterations

### Performance Targets
- Simple plans: <1ms
- Medium plans: <10ms
- Complex plans: <100ms

## Telemetry Metrics

```python
from src.chat.metrics import get_metrics_registry

registry = get_metrics_registry()

# Counters
registry.inc('goap_planning_attempts_total')
registry.inc('goap_plans_found_total')
registry.inc('goap_plans_not_found_total')

# Histograms
registry.observe('goap_plan_length', len(plan.steps))
registry.observe('goap_plan_cost', plan.total_cost)
registry.observe('goap_nodes_expanded', plan.nodes_expanded)
registry.observe_hist('goap_planning_latency_ms', plan.planning_time_ms)
```

### Available Metrics
- `goap_planning_attempts_total`: Total planning attempts
- `goap_plans_found_total`: Successful plans
- `goap_plans_not_found_total`: Failed attempts
- `goap_plan_length`: Steps in plan (histogram)
- `goap_plan_cost`: Total cost (histogram)
- `goap_nodes_expanded`: Search nodes (histogram)
- `goap_planning_latency_ms`: Time in ms (histogram)
- `goap_failed_iterations`: Max iterations hit
- `goap_failed_nodes_expanded`: Nodes when failed
- `goap_failed_planning_latency_ms`: Time when failed

## Debugging

### State Inspection
```python
# Print state
print(f"State: {state.state}")

# Check differences
delta = state1.delta(state2)
print(f"Differences: {delta}")

# Check goal satisfaction
if not state.satisfies(goal):
    unsatisfied = goal.distance_to(state)
    print(f"Unsatisfied goals: {unsatisfied}")
```

### Plan Validation
```python
# Check plan validity
state = plan.initial_state
for i, step in enumerate(plan.steps):
    if not state.satisfies(step.action.preconditions):
        print(f"Invalid at step {i+1}: {step.action.name}")
        break
    state = step.action.apply(state)

if state.satisfies(plan.goal_state):
    print("Valid plan")
```

### Search Statistics
```python
if plan:
    print(f"Planning time: {plan.planning_time_ms:.2f}ms")
    print(f"Nodes expanded: {plan.nodes_expanded}")
    print(f"Plan length: {len(plan.steps)}")
    print(f"Total cost: {plan.total_cost:.2f}")
```

## Error Handling

```python
try:
    plan = planner.plan(initial, goal, max_iterations=1000)
    
    if plan is None:
        # No solution found
        print("No plan exists to achieve goal")
        # Try relaxing goal or adding more actions
    else:
        # Execute plan
        execute_plan(plan)
        
except Exception as e:
    print(f"Planning error: {e}")
    # Fallback to alternative planning method
```

## Best Practices

1. **Immutability**: Always use `.set()` or `.update()` to modify WorldState
2. **Admissible Heuristics**: Never overestimate remaining cost
3. **Action Costs**: Use realistic costs for meaningful optimization
4. **Preconditions**: Make preconditions as specific as needed
5. **Effects**: Define all state changes in effects
6. **Iteration Limits**: Set appropriate limits to prevent runaway search
7. **Telemetry**: Track metrics for performance analysis
8. **Testing**: Validate plans before execution
9. **Fallbacks**: Have backup strategies when planning fails
10. **Documentation**: Document custom actions and their purpose

## Common Issues

### Issue: No plan found
**Solution**: 
- Check if goal is reachable with available actions
- Increase `max_iterations`
- Try simpler heuristic
- Add intermediate actions

### Issue: Slow planning
**Solution**:
- Use simpler heuristic (`goal_distance`)
- Reduce action library size
- Lower `max_iterations`
- Simplify goal

### Issue: Suboptimal plan
**Solution**:
- Use better heuristic (`relaxed_plan`)
- Check action costs are realistic
- Ensure heuristic is admissible
- Increase search iterations

### Issue: Invalid plan
**Solution**:
- Verify action preconditions/effects are correct
- Check WorldState immutability
- Validate initial state has required keys

## Further Reading

- **Tests**: `tests/test_executive_goap_planner.py` (40 comprehensive tests)
- **Examples**: `docs/goap_usage_examples.md` (detailed usage patterns)
- **API**: `src/executive/planning/` (source code with docstrings)
- **Theory**: Orkin 2006, "Three States and a Plan: The A.I. of F.E.A.R."
