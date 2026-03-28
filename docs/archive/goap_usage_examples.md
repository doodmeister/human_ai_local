# GOAP Planning System - Usage Examples

This document provides practical examples of using the Goal-Oriented Action Planning (GOAP) system in the Human-AI Cognition Framework.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Custom Actions](#custom-actions)
- [Heuristic Selection](#heuristic-selection)
- [Plan Execution](#plan-execution)
- [Performance Tuning](#performance-tuning)
- [Integration Patterns](#integration-patterns)

---

## Basic Usage

### Simple Planning Workflow

```python
from src.executive.planning import (
    WorldState,
    GOAPPlanner,
    create_default_action_library
)

# Define the problem: what state are we in, what state do we want?
initial_state = WorldState({
    "has_data": False,
    "has_analysis": False,
    "has_document": False
})

goal_state = WorldState({
    "has_document": True  # We want to end up with a document
})

# Create planner with default actions
action_library = create_default_action_library()
planner = GOAPPlanner(
    action_library=action_library,
    heuristic='goal_distance'
)

# Find optimal plan
plan = planner.plan(
    initial_state=initial_state,
    goal_state=goal_state,
    max_iterations=1000
)

# Check results
if plan:
    print(f"✓ Found plan with {len(plan.steps)} steps")
    print(f"  Total cost: {plan.total_cost:.2f}")
    print(f"  Planning time: {plan.planning_time_ms:.2f}ms")
    print(f"  Nodes expanded: {plan.nodes_expanded}")
    print("\nAction sequence:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step.action.name}")
        print(f"     Cost: {step.cost:.2f}")
        print(f"     Cognitive Load: {step.action.cognitive_load:.1f}")
else:
    print("✗ No plan found to achieve goal")
```

**Expected output:**
```
✓ Found plan with 3 steps
  Total cost: 30.00
  Planning time: 0.87ms
  Nodes expanded: 7

Action sequence:
  1. gather_data
     Cost: 10.00
     Cognitive Load: 3.0
  2. analyze_data
     Cost: 10.00
     Cognitive Load: 5.0
  3. create_document
     Cost: 10.00
     Cognitive Load: 4.0
```

---

## Custom Actions

### Creating Domain-Specific Actions

```python
from src.executive.planning import WorldState, Action, ActionLibrary

# Define custom actions for a specific domain (e.g., software development)
def create_software_dev_actions():
    actions = []
    
    # Action: Write code
    actions.append(Action(
        name="write_code",
        preconditions=WorldState({"requirements_clear": True}),
        effects=WorldState({"code_written": True}),
        cost=15.0,
        cognitive_load=6.0,
        resources=["editor", "time"],
        duration=3600  # 1 hour in seconds
    ))
    
    # Action: Write tests
    actions.append(Action(
        name="write_tests",
        preconditions=WorldState({"code_written": True}),
        effects=WorldState({"tests_written": True}),
        cost=10.0,
        cognitive_load=5.0,
        resources=["editor", "time"],
        duration=1800
    ))
    
    # Action: Run tests
    actions.append(Action(
        name="run_tests",
        preconditions=WorldState({
            "code_written": True,
            "tests_written": True
        }),
        effects=WorldState({"tests_passed": True}),
        cost=5.0,
        cognitive_load=2.0,
        resources=["compute"],
        duration=300
    ))
    
    # Action: Code review
    actions.append(Action(
        name="code_review",
        preconditions=WorldState({
            "code_written": True,
            "tests_passed": True
        }),
        effects=WorldState({"reviewed": True}),
        cost=8.0,
        cognitive_load=4.0,
        resources=["reviewer", "time"],
        duration=1200
    ))
    
    # Action: Deploy
    actions.append(Action(
        name="deploy",
        preconditions=WorldState({
            "reviewed": True,
            "tests_passed": True
        }),
        effects=WorldState({"deployed": True}),
        cost=3.0,
        cognitive_load=2.0,
        resources=["infrastructure"],
        duration=600
    ))
    
    # Build library
    library = ActionLibrary()
    for action in actions:
        library.add_action(action)
    
    return library

# Use custom actions
action_library = create_software_dev_actions()
planner = GOAPPlanner(action_library, heuristic='weighted_goal_distance')

initial = WorldState({"requirements_clear": True})
goal = WorldState({"deployed": True})

plan = planner.plan(initial, goal)
if plan:
    print("Deployment plan:")
    for step in plan.steps:
        duration_min = step.action.duration / 60
        print(f"  • {step.action.name} ({duration_min:.0f} min)")
```

---

## Heuristic Selection

### Choosing the Right Heuristic

Different heuristics provide different trade-offs between planning speed and optimality.

#### 1. Goal Distance (Fast, Admissible)
```python
# Best for: Simple problems, when speed matters
planner = GOAPPlanner(action_library, heuristic='goal_distance')
```
- **Speed**: Fastest
- **Optimality**: Guaranteed optimal
- **Use when**: Goals are uniform priority, simple state spaces

#### 2. Weighted Goal Distance (Fast, Admissible with correct weights)
```python
# Best for: Problems with priority goals
planner = GOAPPlanner(action_library, heuristic='weighted_goal_distance')
```
- **Speed**: Fast
- **Optimality**: Optimal if weights are admissible
- **Use when**: Some goals are more important (prefix with `critical_`, `important_`, `optional_`)

Example with priorities:
```python
goal = WorldState({
    "critical_data_secured": True,      # Weight: 3.0
    "important_report_done": True,      # Weight: 2.0
    "optional_email_sent": True         # Weight: 1.0
})
```

#### 3. Relaxed Plan (Slower, More Informed)
```python
# Best for: Complex problems with many dependencies
planner = GOAPPlanner(action_library, heuristic='relaxed_plan')
```
- **Speed**: Slower (runs mini-planner per node)
- **Optimality**: Very good, not guaranteed
- **Use when**: Complex precondition chains, willing to trade speed for better guidance

#### 4. Composite Heuristic (Flexible)
```python
from src.executive.planning.heuristics import CompositeHeuristic, get_heuristic

# Combine multiple heuristics
composite = CompositeHeuristic(
    heuristics=[
        get_heuristic('goal_distance'),
        get_heuristic('weighted_goal_distance')
    ],
    mode='max'  # or 'average', 'weighted'
)

planner = GOAPPlanner(action_library, heuristic=composite)
```
- **Mode options**:
  - `max`: Most conservative (guaranteed admissible if all are admissible)
  - `average`: Balanced estimate
  - `weighted`: Custom importance per heuristic

---

## Plan Execution

### Executing a Plan Step-by-Step

```python
def execute_plan(plan, current_state):
    """Execute plan steps and verify state transitions."""
    print(f"Executing {len(plan.steps)}-step plan...\n")
    
    state = current_state
    
    for i, step in enumerate(plan.steps, 1):
        print(f"Step {i}/{len(plan.steps)}: {step.action.name}")
        
        # Verify preconditions
        if not state.satisfies(step.action.preconditions):
            print(f"  ✗ Preconditions not met!")
            print(f"    Required: {step.action.preconditions.state}")
            print(f"    Current: {state.state}")
            return False
        
        # Apply action
        state = step.action.apply(state)
        
        print(f"  ✓ Applied (cost: {step.cost:.2f})")
        print(f"    New state: {state.state}")
        
        # Check if we've achieved intermediate goals
        if state.satisfies(plan.goal_state):
            print(f"\n✓ Goal achieved early at step {i}!")
            return True
    
    # Verify final state
    if state.satisfies(plan.goal_state):
        print("\n✓ Plan executed successfully, goal achieved!")
        return True
    else:
        print("\n✗ Plan completed but goal not achieved")
        print(f"  Expected: {plan.goal_state.state}")
        print(f"  Got: {state.state}")
        return False

# Usage
if plan:
    success = execute_plan(plan, initial_state)
```

---

## Performance Tuning

### Controlling Search Complexity

```python
# 1. Limit iterations to prevent runaway search
plan = planner.plan(
    initial_state=initial,
    goal_state=goal,
    max_iterations=500  # Stop after 500 nodes
)

# 2. Use simpler heuristic for speed
fast_planner = GOAPPlanner(
    action_library=action_library,
    heuristic='goal_distance'  # Fastest heuristic
)

# 3. Reduce action library size
# Only include actions relevant to current goal
relevant_actions = action_library.get_applicable_actions(initial_state)
small_library = ActionLibrary()
for action in relevant_actions:
    small_library.add_action(action)

optimized_planner = GOAPPlanner(small_library, heuristic='goal_distance')
```

### Monitoring Performance

```python
# Check telemetry after planning
from src.chat.metrics import get_metrics_registry

registry = get_metrics_registry()

# Get planning statistics
attempts = registry.get_counter('goap_planning_attempts_total')
successes = registry.get_counter('goap_plans_found_total')
failures = registry.get_counter('goap_plans_not_found_total')

print(f"Planning success rate: {successes}/{attempts} ({100*successes/attempts:.1f}%)")

# Get performance metrics
avg_nodes = registry.get_histogram('goap_nodes_expanded').mean
avg_time = registry.get_histogram('goap_planning_latency_ms').mean

print(f"Average nodes expanded: {avg_nodes:.0f}")
print(f"Average planning time: {avg_time:.2f}ms")
```

---

## Integration Patterns

### Pattern 1: Fallback Planning

```python
def plan_with_fallback(initial, goal, action_library):
    """Try GOAP, fall back to simpler heuristics if needed."""
    
    # Try with best heuristic first
    planner = GOAPPlanner(action_library, heuristic='relaxed_plan')
    plan = planner.plan(initial, goal, max_iterations=1000)
    
    if plan:
        return plan
    
    # Fallback to faster heuristic
    print("Relaxed plan failed, trying goal_distance...")
    planner = GOAPPlanner(action_library, heuristic='goal_distance')
    plan = planner.plan(initial, goal, max_iterations=2000)
    
    if plan:
        return plan
    
    # Final fallback: Dijkstra (no heuristic)
    print("Goal distance failed, trying exhaustive search...")
    planner = GOAPPlanner(action_library, heuristic='zero')
    plan = planner.plan(initial, goal, max_iterations=5000)
    
    return plan
```

### Pattern 2: Adaptive Replanning

```python
def execute_with_replanning(initial, goal, action_library):
    """Execute plan, replan on failures."""
    
    planner = GOAPPlanner(action_library, heuristic='weighted_goal_distance')
    current_state = initial
    attempts = 0
    max_attempts = 3
    
    while not current_state.satisfies(goal) and attempts < max_attempts:
        attempts += 1
        
        # Plan from current state
        plan = planner.plan(current_state, goal)
        if not plan:
            print(f"✗ Planning failed (attempt {attempts})")
            break
        
        print(f"\nAttempt {attempts}: Executing {len(plan.steps)}-step plan")
        
        # Execute steps until failure or completion
        for step in plan.steps:
            if not current_state.satisfies(step.action.preconditions):
                print(f"✗ Preconditions failed for {step.action.name}")
                print("  Replanning from current state...")
                break  # Replan from here
            
            current_state = step.action.apply(current_state)
            print(f"  ✓ {step.action.name}")
            
            if current_state.satisfies(goal):
                print("✓ Goal achieved!")
                return True
    
    return current_state.satisfies(goal)
```

### Pattern 3: Multi-Agent Coordination

```python
def coordinate_agents(agents, shared_goal, action_libraries):
    """Plan for multiple agents with shared resources."""
    
    plans = {}
    
    for agent_id, action_library in action_libraries.items():
        initial = agents[agent_id].state
        
        # Create agent-specific goal (subset of shared goal)
        agent_goal = extract_agent_goals(shared_goal, agent_id)
        
        planner = GOAPPlanner(action_library, heuristic='goal_distance')
        plan = planner.plan(initial, agent_goal)
        
        if plan:
            plans[agent_id] = plan
        else:
            print(f"✗ No plan for agent {agent_id}")
    
    # Merge plans, resolve conflicts
    merged_plan = merge_agent_plans(plans, shared_goal)
    return merged_plan
```

---

## Common Patterns

### Pattern: Goal Cascading
```python
# Break down complex goals into stages
def cascade_goals(ultimate_goal, action_library):
    stages = [
        WorldState({"data_prepared": True}),
        WorldState({"analysis_complete": True}),
        WorldState({"report_written": True}),
        ultimate_goal
    ]
    
    current_state = WorldState({})
    all_steps = []
    
    for stage_goal in stages:
        planner = GOAPPlanner(action_library, heuristic='goal_distance')
        plan = planner.plan(current_state, stage_goal)
        
        if not plan:
            return None
        
        all_steps.extend(plan.steps)
        current_state = plan.steps[-1].state_after
    
    return all_steps
```

### Pattern: Conditional Planning
```python
# Plan based on runtime conditions
def conditional_plan(state, goal, action_library):
    # Check current conditions
    if state.get("time_pressure"):
        # Use fast heuristic under time pressure
        heuristic = 'goal_distance'
        max_iter = 500
    else:
        # Use better heuristic when time permits
        heuristic = 'relaxed_plan'
        max_iter = 2000
    
    planner = GOAPPlanner(action_library, heuristic=heuristic)
    return planner.plan(state, goal, max_iterations=max_iter)
```

---

## Debugging Tips

### 1. State Inspection
```python
# Print state differences
def debug_state_transition(before, after, action):
    delta = before.delta(after)
    print(f"Action: {action.name}")
    print(f"  Changed: {delta}")
    print(f"  Before: {before.state}")
    print(f"  After: {after.state}")
```

### 2. Plan Validation
```python
# Verify plan is valid before execution
def validate_plan(plan):
    state = plan.initial_state
    
    for i, step in enumerate(plan.steps):
        # Check preconditions
        if not state.satisfies(step.action.preconditions):
            print(f"✗ Step {i+1} preconditions not met")
            return False
        
        # Apply action
        state = step.action.apply(state)
    
    # Check final state achieves goal
    if not state.satisfies(plan.goal_state):
        print("✗ Final state doesn't achieve goal")
        return False
    
    print("✓ Plan is valid")
    return True
```

### 3. Search Visualization
```python
# Track search progress
def plan_with_logging(planner, initial, goal):
    import time
    start = time.time()
    
    plan = planner.plan(initial, goal)
    elapsed = time.time() - start
    
    if plan:
        print(f"✓ Planning succeeded")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Nodes: {plan.nodes_expanded}")
        print(f"  Steps: {len(plan.steps)}")
        print(f"  Cost: {plan.total_cost:.2f}")
    else:
        print(f"✗ Planning failed after {elapsed*1000:.2f}ms")
    
    return plan
```

---

## Next Steps

- See `tests/test_executive_goap_planner.py` for comprehensive test examples
- See `src/executive/planning/README.md` for API reference (coming soon)
- See `docs/executive_refactoring_plan.md` for Phase 2 completion roadmap
