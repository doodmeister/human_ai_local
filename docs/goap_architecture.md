# GOAP Planning Architecture

## Overview

The Goal-Oriented Action Planning (GOAP) system implements an A* search-based planner for finding optimal action sequences to achieve goals. This document describes the architecture, design decisions, and technical details.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     GOAP Planning System                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  WorldState  │      │    Action    │     │  Heuristics  │
│              │      │   Library    │     │              │
│ • Immutable  │      │              │     │ • Goal Dist  │
│   State      │◄─────│ • Actions    │     │ • Weighted   │
│ • Key-Value  │      │ • Preconds   │     │ • Relaxed    │
│ • Satisfies  │      │ • Effects    │     │ • Composite  │
│ • Delta      │      │ • Costs      │     │              │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  GOAPPlanner     │
                    │                  │
                    │ • A* Search      │
                    │ • Open/Closed    │
                    │ • Plan Output    │
                    │ • Telemetry      │
                    └──────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │     Plan     │
                      │              │
                      │ • Steps      │
                      │ • Cost       │
                      │ • Metrics    │
                      └──────────────┘
```

## Core Components

### 1. WorldState (world_state.py)

**Purpose**: Represent the state of the world at any point in time.

**Design Decisions**:
- **Immutable**: Frozen dataclass ensures states can be used as dictionary keys in A* closed set
- **Key-Value**: Generic dict representation supports arbitrary domains
- **Frozenset Hashing**: Converts dict to frozenset for efficient hashing and set membership

**Key Methods**:
```python
class WorldState:
    _state: Dict[str, Any]              # Internal state storage
    _hashable_state: frozenset          # For hashing
    
    @property
    def state -> Dict[str, Any]:        # Public read-only access
    
    def get(key, default=None)          # Get value
    def set(key, value) -> WorldState   # Create new state with change
    def update(updates) -> WorldState   # Create new state with changes
    def satisfies(goal) -> bool         # Check if goal is met
    def delta(other) -> Dict            # Get differences
    def distance_to(goal) -> int        # Count unsatisfied goals
```

**Implementation Notes**:
- Uses `__post_init__` to create hashable frozenset from dict
- `__hash__` and `__eq__` based on frozenset for set operations
- All modifications return new instances (functional style)

### 2. Action & ActionLibrary (action_library.py)

**Purpose**: Define actions with preconditions, effects, and costs.

**Design Decisions**:
- **Preconditions as WorldState**: Declarative precondition specification
- **Effects as WorldState**: Declarative effect specification  
- **Cost Function**: Takes from_state and to_state for context-dependent costs
- **Rich Metadata**: cognitive_load, resources, duration for future use

**Key Classes**:
```python
@dataclass
class Action:
    name: str
    preconditions: WorldState           # Required state
    effects: WorldState                 # Resulting changes
    cost: float                         # Base cost
    cognitive_load: float = 0.0         # Mental effort
    resources: List[str] = field(default_factory=list)
    duration: int = 0                   # Seconds
    
    def is_applicable(state) -> bool    # Check preconditions
    def apply(state) -> WorldState      # Apply effects
    def get_cost(from_state, to_state) -> float

class ActionLibrary:
    _actions: Dict[str, Action]
    
    def add_action(action)
    def remove_action(name)
    def get_action(name) -> Optional[Action]
    def get_applicable_actions(state) -> List[Action]
```

**Default Actions**:
10 predefined actions covering common planning scenarios:
- Data handling: gather_data, analyze_data
- Document creation: create_document, draft_outline
- Communication: send_notification, schedule_meeting
- Quality: review_work, run_tests
- Planning: create_plan, break_down_goal

### 3. GOAPPlanner (goap_planner.py)

**Purpose**: Find optimal action sequence using A* search.

**Algorithm**: A* graph search over state space
- **Nodes**: States in the world
- **Edges**: Actions that transition between states
- **Cost**: Action costs (g-score)
- **Heuristic**: Estimated cost to goal (h-score)
- **f-score**: g-score + h-score (priority)

**Design Decisions**:
- **Priority Queue**: heapq for efficient min-heap operations
- **Closed Set**: Set of visited states (requires hashable WorldState)
- **Path Reconstruction**: Backtrack from goal to start using parent pointers
- **Telemetry**: Optional metrics tracking with graceful fallback

**Key Classes**:
```python
@dataclass(order=True)
class _SearchNode:
    f_score: float                      # Priority for heap
    g_score: float = field(compare=False)  # Actual cost
    h_score: float = field(compare=False)  # Heuristic estimate
    state: WorldState = field(compare=False)
    parent: Optional['_SearchNode'] = field(compare=False)
    action: Optional[Action] = field(compare=False)

@dataclass
class PlanStep:
    action: Action
    state_before: WorldState
    state_after: WorldState
    cost: float
    step_number: int

@dataclass  
class Plan:
    steps: List[PlanStep]
    initial_state: WorldState
    goal_state: WorldState
    total_cost: float
    nodes_expanded: int
    planning_time_ms: float

class GOAPPlanner:
    def __init__(action_library, heuristic):
        self._action_library = action_library
        self._heuristic = _resolve_heuristic(heuristic)
    
    def plan(initial_state, goal_state, max_iterations=1000) -> Optional[Plan]:
        # A* search implementation
        # Returns Plan or None
```

**A* Search Algorithm**:
1. Initialize open set with initial state (f=h, g=0)
2. Initialize closed set (empty)
3. While open set not empty and iterations < max:
   - Pop node with lowest f-score
   - If node satisfies goal, reconstruct and return plan
   - Add node to closed set
   - For each applicable action:
     - Apply action to get neighbor state
     - If neighbor in closed set, skip
     - Calculate g-score (current + action cost)
     - Calculate h-score (heuristic estimate)
     - Add neighbor to open set with f = g + h
4. If open set empty or max iterations, return None

**Optimizations**:
- Early goal checking before expansion
- Closed set prevents revisiting states
- Priority queue ensures optimal expansion order
- Configurable iteration limit prevents infinite loops

### 4. Heuristics (heuristics.py)

**Purpose**: Guide A* search toward goal efficiently.

**Design Decisions**:
- **Admissibility**: Must never overestimate remaining cost (ensures optimality)
- **Protocol**: Common interface for all heuristics
- **Composability**: Combine multiple heuristics for better estimates
- **Registry**: Named lookup for easy configuration

**Key Heuristics**:

#### goal_distance_heuristic
- **Strategy**: Count unsatisfied goals
- **Admissible**: Yes (assumes each goal costs at least 1 action)
- **Speed**: Very fast (O(n) where n = goal keys)
- **Use case**: Simple problems, uniform goal importance

#### weighted_goal_distance_heuristic  
- **Strategy**: Weight goals by prefix (critical=3, important=2, optional=1)
- **Admissible**: Yes if weights ≤ minimum action cost to satisfy
- **Speed**: Fast (O(n))
- **Use case**: Prioritized goals, safety-critical systems

#### relaxed_plan_heuristic
- **Strategy**: Run mini-planner ignoring preconditions
- **Admissible**: Yes (ignoring constraints can't increase cost)
- **Speed**: Slower (O(b^d) mini-search per call)
- **Use case**: Complex precondition chains, better guidance needed

#### zero_heuristic
- **Strategy**: Always return 0 (degenerates to Dijkstra)
- **Admissible**: Yes (maximally conservative)
- **Speed**: Slowest (explores more nodes)
- **Use case**: Guarantee optimal when heuristic quality unknown

#### max_heuristic
- **Strategy**: Maximum of all other heuristics
- **Admissible**: Yes if all components admissible
- **Speed**: Depends on components
- **Use case**: Conservative combination, robustness

#### CompositeHeuristic
- **Strategy**: Combine heuristics with different modes
- **Modes**:
  - `max`: Most conservative, admissible if all components admissible
  - `average`: Balanced estimate, may not be admissible
  - `weighted`: Custom weights, admissibility depends on weights
- **Use case**: Flexible combination of domain knowledge

**Heuristic Protocol**:
```python
class Heuristic(Protocol):
    def __call__(current: WorldState, goal: WorldState, action_library: ActionLibrary) -> float:
        ...
```

## Performance Characteristics

### Time Complexity

**A* Search**: O(b^d)
- b = branching factor (average applicable actions per state)
- d = depth of solution (number of steps)
- Heuristic quality dramatically affects practical performance

**Heuristics**:
- goal_distance: O(n) where n = goal keys
- weighted_goal_distance: O(n)
- relaxed_plan: O(b^d) mini-search per node
- zero: O(1)
- max: O(k) where k = number of heuristics

### Space Complexity

**Open Set**: O(b^d) in worst case
- Priority queue stores frontier nodes
- Typically much smaller in practice with good heuristic

**Closed Set**: O(s) where s = unique states visited
- Set of explored states
- Can be large for complex state spaces
- Mitigated by early goal detection

**Plan**: O(d) where d = plan length
- Linear in solution depth

### Practical Performance

Based on test results:
- **Simple plans (2-3 steps)**: <1ms, <10 nodes expanded
- **Medium plans (4-6 steps)**: <10ms, <50 nodes expanded  
- **Complex plans (7-10 steps)**: <100ms, <500 nodes expanded

**Factors affecting performance**:
- State space size (# of state keys)
- Action library size (# of actions)
- Branching factor (avg applicable actions)
- Solution depth (# of steps)
- Heuristic quality (better = fewer nodes)

## Design Patterns

### 1. Immutability Pattern

**Motivation**: A* requires hashable states for closed set membership testing.

**Implementation**: 
- Frozen dataclass prevents modification
- All state changes return new instances
- Frozenset enables hashing and set operations

**Benefits**:
- Thread-safe state sharing
- No accidental mutations
- Efficient set operations
- Clear functional semantics

### 2. Strategy Pattern (Heuristics)

**Motivation**: Different problems benefit from different heuristics.

**Implementation**:
- Protocol defines heuristic interface
- Multiple implementations (goal_distance, weighted, etc.)
- Runtime selection via registry
- Composability via CompositeHeuristic

**Benefits**:
- Easy to add new heuristics
- Configurable planning behavior
- A/B testing of heuristics
- Domain-specific optimization

### 3. Builder Pattern (ActionLibrary)

**Motivation**: Actions are complex with many optional parameters.

**Implementation**:
- Action dataclass with sensible defaults
- ActionLibrary aggregates actions
- create_default_action_library() factory

**Benefits**:
- Readable action definitions
- Gradual complexity (start simple, add details)
- Reusable action libraries
- Domain-specific libraries

### 4. Graceful Degradation (Telemetry)

**Motivation**: Metrics valuable but not critical to planning.

**Implementation**:
```python
def get_metrics_registry():
    try:
        from src.chat.metrics import metrics_registry
        return metrics_registry
    except ImportError:
        return DummyRegistry()  # No-op methods
```

**Benefits**:
- Loose coupling to metrics system
- Works without metrics infrastructure
- Testing doesn't require metrics setup
- Production gets full telemetry

### 5. Result Object Pattern (Plan)

**Motivation**: Planning returns rich result with metadata.

**Implementation**:
- Plan dataclass with steps and metrics
- PlanStep with state transitions
- None indicates failure (no plan found)

**Benefits**:
- Rich information for analysis
- Separate success/failure cases
- Metrics included with result
- Immutable result value

## Integration Points

### With Metrics System (src/chat/metrics.py)

```python
# Planning metrics tracked automatically
registry.inc('goap_planning_attempts_total')
registry.inc('goap_plans_found_total')
registry.observe('goap_plan_length', len(plan.steps))
registry.observe_hist('goap_planning_latency_ms', elapsed)
```

**Metrics Available**:
- Counters: attempts, successes, failures
- Histograms: plan length, cost, nodes expanded, latency
- Failure analysis: iterations hit, nodes at failure, failure latency

### With Legacy TaskPlanner (Future)

```python
# Adapter pattern (Phase 2 completion)
class GOAPTaskPlannerAdapter:
    def __init__(self, goap_planner, feature_flags):
        self._goap = goap_planner
        self._flags = feature_flags
    
    def plan_for_goal(self, goal):
        if self._flags.use_goap:
            # Convert goal → WorldState
            # Plan with GOAP
            # Convert Plan → Task list
            return goap_plan_to_tasks(plan)
        else:
            # Fallback to template-based
            return legacy_plan_for_goal(goal)
```

### With Chat System (Context Enrichment)

```python
# Potential future use
def enrich_context_with_planning(context):
    # Extract intent from context
    # Formulate as planning problem
    # Generate plan
    # Add plan to context as structured reasoning
    context.items.append({
        'type': 'executive_plan',
        'plan': plan,
        'confidence': plan_confidence(plan)
    })
```

## Testing Strategy

### Unit Tests (40 tests)

**Coverage**:
- WorldState: immutability, satisfies, delta, distance, hashing
- Action: applicability, apply, cost calculation
- ActionLibrary: filtering, adding, default actions
- Heuristics: all types, admissibility, composite
- GOAPPlanner: simple/optimal paths, no solution, different heuristics
- Plan Execution: sequence validity, goal achievement
- Edge Cases: empty library, max iterations, complex goals
- Performance: latency targets for small/medium plans

**Test Philosophy**:
- Fast unit tests (<100ms total)
- No external dependencies (no ChromaDB, no models)
- Clear assertions (optimal plan selection, admissibility)
- Property testing (immutability, admissibility)

### Integration Tests (Planned)

**Scenarios**:
- End-to-end: user intent → planning → task execution
- Legacy integration: GOAP + template-based fallback
- Replanning: action failure → replan from current state
- Multi-agent: coordinated planning with shared resources
- Performance: large action libraries, deep plans

## Future Enhancements

### Phase 2 Completion (Tasks 6-8, 10)

**Task 6: Constraint Integration**
- Resource constraints (memory, time, energy)
- Temporal constraints (deadlines, dependencies)
- Constraint propagation during search
- Constraint satisfaction checking

**Task 7: Replanning Engine**
- Detect action failure during execution
- Replan from current state
- Plan repair vs. full replan
- Reactive planning for dynamic environments

**Task 8: Legacy Integration**
- GOAPTaskPlannerAdapter
- Feature flags for gradual rollout
- Fallback to template-based on errors
- A/B testing infrastructure

**Task 10: Integration Tests**
- End-to-end scenarios
- Performance benchmarks
- Stress testing (large state spaces)
- Failure mode validation

### Phase 3+ Enhancements

**HTN Integration**:
- Hierarchical task networks for goal decomposition
- GOAP for low-level action planning
- HTN provides high-level structure

**Learning**:
- Learn action costs from execution data
- Adapt heuristics based on domain
- Predict plan success probability

**Distributed Planning**:
- Multi-agent coordination
- Shared state synchronization
- Conflict resolution

**Optimization**:
- Partial-order planning (parallelize actions)
- Incremental planning (reuse previous plans)
- Anytime planning (return best-so-far)

## References

- Orkin, J. (2006). "Three States and a Plan: The A.I. of F.E.A.R." Game Developers Conference.
- Russell, S. & Norvig, P. (2021). "Artificial Intelligence: A Modern Approach" (4th ed.), Chapter 11: Planning.
- Nau, D. et al. (2004). "SHOP2: An HTN Planning System." Journal of Artificial Intelligence Research.

## Appendix: Key Algorithms

### A* Search (Pseudocode)

```
function A_STAR(initial, goal, actions, heuristic):
    open_set = priority_queue([initial])
    closed_set = set()
    g_scores = {initial: 0}
    parents = {}
    
    while open_set is not empty:
        current = open_set.pop_min()  # Lowest f-score
        
        if current.satisfies(goal):
            return reconstruct_plan(current, parents)
        
        closed_set.add(current)
        
        for action in actions.get_applicable(current):
            neighbor = action.apply(current)
            
            if neighbor in closed_set:
                continue
            
            g = g_scores[current] + action.get_cost(current, neighbor)
            
            if neighbor not in g_scores or g < g_scores[neighbor]:
                g_scores[neighbor] = g
                h = heuristic(neighbor, goal)
                f = g + h
                parents[neighbor] = (current, action)
                open_set.push(neighbor, f)
    
    return None  # No plan found
```

### Relaxed Plan Heuristic (Pseudocode)

```
function RELAXED_PLAN(current, goal, actions):
    state = current.copy()
    plan = []
    
    while not state.satisfies(goal):
        # Find action that makes most progress (ignoring preconditions)
        best_action = None
        best_progress = 0
        
        for action in actions:
            # Apply without checking preconditions
            new_state = state.update(action.effects)
            progress = goal.distance_to(state) - goal.distance_to(new_state)
            
            if progress > best_progress:
                best_action = action
                best_progress = progress
        
        if best_action is None:
            return float('inf')  # Goal not achievable
        
        state = state.update(best_action.effects)
        plan.append(best_action)
    
    return sum(a.cost for a in plan)  # Estimate
```
