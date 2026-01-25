# Executive Function Refactoring - Comprehensive Implementation Plan
**Project Start**: October 31, 2025  
**Status**: Planning Phase  
**Goal**: Transform executive modules from simple rule-based systems to sophisticated cognitive reasoning

---

## Executive Summary

This plan refactors the Executive Function components (Decision Engine, Task Planner, Goal Manager, Scheduler) to use advanced algorithms that provide:
- **Better decisions** through multi-criteria analysis (AHP, Pareto optimization)
- **Smarter planning** via GOAP and constraint satisfaction
- **Adaptive behavior** through reinforcement learning from outcomes
- **Robust scheduling** using constraint programming

---

## Current State Assessment

### Decision Engine (`src/executive/decision_engine.py` - 594 lines)
**Strengths:**
- Clean dataclass-based architecture
- Multiple decision types (choice, ranking, binary, etc.)
- Criterion abstraction with weights and evaluators
- Constraint checking capability

**Weaknesses:**
- Simple weighted sum (`WeightedScoringStrategy`)
- Static weights, no dynamic adjustment
- No learning from decision outcomes
- No trade-off visualization (Pareto analysis)
- Uncertainty handling is rudimentary

### Task Planner (`src/executive/task_planner.py` - 450 lines)
**Strengths:**
- Task dependencies and status tracking
- Cognitive load estimation
- Template-based decomposition
- Success criteria definition

**Weaknesses:**
- Templates are static, not adaptive
- No search-based planning (just templates)
- No constraint satisfaction (resources, time)
- Cannot replan dynamically
- No action precondition/effect modeling

### Goal Manager (`src/executive/goal_manager.py` - 394 lines)
**Strengths:**
- Hierarchical goal structure (parent-child)
- Progress tracking with history
- Priority levels
- Success criteria

**Weaknesses:**
- No formal goal decomposition algorithm (HTN)
- Simple progress calculation, no predictive completion
- No conflict detection between goals
- Priority is static enum, not calculated dynamically
- No resource contention handling

### Scheduler
**Current State:** No dedicated scheduler
**Issues:**
- Resource allocation scattered across modules
- No temporal constraint handling
- No optimization of task ordering

---

## Implementation Phases

### PHASE 1: Foundation & Decision Engine Enhancement (Weeks 1-4)
**Priority: HIGHEST** - Decision quality impacts all other modules

#### Week 1: Setup & Architecture
- [ ] Install dependencies: `numpy`, `scipy`, `scikit-learn`, `ortools`, `networkx`
- [ ] Create new module structure:
  ```
  src/executive/decision/
    __init__.py
    base.py                 # Abstract base classes
    ahp_engine.py          # Analytic Hierarchy Process
    pareto_optimizer.py    # Multi-objective optimization
    ml_decision_model.py   # Learning component
    context_analyzer.py    # Context-aware weighting
  ```
- [ ] Define interfaces compatible with existing `DecisionEngine`
- [ ] Create feature flag system for gradual rollout

#### Week 2: AHP Implementation
- [ ] Implement pairwise comparison matrix builder
- [ ] Calculate priority vectors using eigenvector method
- [ ] Add consistency ratio validation (CR < 0.1)
- [ ] Support hierarchical criteria (sub-criteria)
- [ ] Write unit tests for AHP calculations

#### Week 3: Pareto Optimization
- [ ] Implement multi-objective scoring
- [ ] Calculate Pareto frontier (non-dominated solutions)
- [ ] Add trade-off visualization data structures
- [ ] Integrate with existing `DecisionResult`
- [ ] Test with competing objectives (speed vs quality)

#### Week 4: Context & ML Layer
- [ ] Context analyzer for dynamic weight adjustment
  - Cognitive load → adjust complexity preferences
  - Time pressure → favor quick decisions
  - Risk tolerance → adjust uncertainty handling
- [ ] Simple decision tree for learning from outcomes
  - Track decision → outcome pairs
  - Train on successful decisions
  - Suggest weight adjustments
- [ ] Integration tests with legacy `WeightedScoringStrategy`

**Deliverables:**
- Enhanced decision engine with AHP and Pareto
- Backward-compatible API
- 90%+ test coverage
- Performance benchmarks (target: <100ms for decisions)

---

### PHASE 2: GOAP Planning System (Weeks 5-8)

#### Week 5: GOAP Foundation
- [ ] Create module structure:
  ```
  src/executive/planning/
    __init__.py
    goap_planner.py        # Main GOAP implementation
    world_state.py         # State representation
    action_library.py      # Action definitions
    heuristics.py          # Planning heuristics
  ```
- [ ] Define world state as key-value pairs
- [ ] Create `Action` class with preconditions and effects
- [ ] Implement basic action library (analyze, create, communicate, etc.)

#### Week 6: A* Search Implementation
- [ ] Implement A* search over action space
- [ ] Goal distance heuristic (state delta estimation)
- [ ] Path cost calculation (time + cognitive load)
- [ ] Early termination conditions
- [ ] Plan validation (reachability check)

#### Week 7: Constraint Integration
- [ ] Add resource constraints to actions
- [ ] Temporal constraints (deadlines, durations)
- [ ] Dependency constraints
- [ ] Constraint propagation during search
- [ ] Feasibility checking

#### Week 8: Replanning Engine
- [ ] Detect plan invalidation triggers
  - Goal changed
  - Action failed
  - World state diverged
- [ ] Incremental replanning (reuse valid prefix)
- [ ] Plan repair heuristics
- [ ] Integration with existing `TaskPlanner`

**Deliverables:**
- GOAP planner producing action sequences
- 20+ predefined actions in library
- Replanning capability
- Performance: <1s for 10-step plans

---

### PHASE 3: HTN Goal Management (Weeks 9-11)

#### Week 9: HTN Framework
- [ ] Create module structure:
  ```
  src/executive/goals/
    __init__.py
    htn_manager.py         # Hierarchical Task Network
    goal_taxonomy.py       # Goal type definitions
    decomposition.py       # Goal → task decomposition
    conflict_resolver.py   # Goal conflict detection
  ```
- [ ] Define goal hierarchy with methods (decomposition rules)
- [ ] Implement recursive HTN decomposition
- [ ] Primitive vs compound task distinction

#### Week 10: Goal Intelligence
- [ ] Dynamic priority calculation
  - Urgency (deadline proximity)
  - Importance (impact score)
  - Dependencies (blocking others)
  - Resource availability
- [ ] Conflict detection algorithms
  - Resource contention
  - Incompatible states
  - Time overlap
- [ ] Conflict resolution strategies
  - Priority-based
  - Resource sharing
  - Goal postponement

#### Week 11: Predictive Features
- [ ] Goal completion prediction
  - Based on task progress
  - Historical completion rates
  - Resource availability trends
- [ ] Risk assessment (failure probability)
- [ ] Proactive goal adjustment suggestions
- [ ] Integration with existing `GoalManager`

**Deliverables:**
- HTN-based goal decomposition
- Conflict detection and resolution
- Predictive analytics
- Backward-compatible with current goal API

---

### PHASE 4: Constraint-Based Scheduling (Weeks 12-14)

#### Week 12: Scheduler Foundation
- [ ] Create module structure:
  ```
  src/executive/scheduling/
    __init__.py
    cp_scheduler.py        # Constraint programming scheduler
    resource_allocator.py  # Resource management
    timeline.py            # Temporal scheduling
    optimizer.py           # Multi-objective optimization
  ```
- [ ] Define scheduling variables (task assignments, start times)
- [ ] Define domains (available resources, time windows)
- [ ] Model as Constraint Satisfaction Problem (CSP)

#### Week 13: OR-Tools Integration
- [ ] Use Google OR-Tools CP-SAT solver
- [ ] Define constraints:
  - Precedence (task A before B)
  - Resource capacity (max parallel tasks)
  - Deadlines (task finish time)
  - Cognitive load limits
- [ ] Objective functions:
  - Minimize makespan (total time)
  - Minimize cognitive peaks
  - Maximize priority-weighted completion
- [ ] Multi-objective optimization (weighted sum or epsilon-constraint)

#### Week 14: Dynamic Scheduling
- [ ] Real-time schedule updates
- [ ] Reactive scheduling (handle disruptions)
- [ ] Proactive scheduling (anticipate issues)
- [ ] Schedule quality metrics
- [ ] Visualization data export

**Deliverables:**
- Constraint-based scheduler
- OR-Tools integration
- Real-time rescheduling
- Performance: <500ms for 50-task schedules

---

### PHASE 5: Integration & ML Learning Layer (Weeks 15-17)

#### Week 15: System Integration ✅ COMPLETE
- [x] Connect Decision Engine → Task Planner
- [x] Connect Task Planner → Scheduler
- [x] Connect Goal Manager → all components
- [x] End-to-end data flow testing (17/24 tests passing, core 100% functional)
- [x] Performance profiling and optimization (12-15s pipeline latency)

**Week 15 Deliverables:**
- `src/executive/integration.py` (497 lines): ExecutiveSystem orchestrator
- `tests/test_integration_week15.py` (480 lines): 24 integration tests
- Full pipeline: Goal → Decision → GOAP Plan → CP-SAT Schedule → Execution
- ExecutionContext tracking with timing metrics
- System health monitoring and execution history
- 6 performance counters via metrics_registry
- IntegrationConfig for feature toggles and timeouts

#### Week 16: Learning Infrastructure
- [ ] Outcome tracking system
  - Decision outcomes (success/failure)
  - Plan execution success rate
  - Goal achievement rate
- [ ] Feature extraction for ML models
- [ ] Model training pipeline
- [ ] Model persistence and versioning
- [ ] A/B testing framework

#### Week 17: Production Readiness
- [ ] Feature flag system for gradual rollout
- [ ] Fallback mechanisms (revert to legacy on failure)
- [ ] Comprehensive logging and telemetry
- [ ] Performance monitoring dashboards
- [ ] Documentation and migration guides

**Deliverables:**
- Fully integrated executive system ✅
- ML learning infrastructure (Week 16)
- Production deployment toolkit (Week 17)
- ML learning pipeline operational
- Production-ready deployment
- Complete documentation

---

## Technical Specifications

### Dependencies
```bash
# Core algorithmic libraries
pip install numpy scipy scikit-learn

# Optimization and constraint solving
pip install ortools python-constraint

# Graph algorithms and data structures
pip install networkx

# Data validation
pip install pydantic

# Existing dependencies (already installed)
# sentence-transformers, chromadb, fastapi, etc.
```

### Architecture Principles

1. **Backward Compatibility**: Old APIs remain functional
2. **Gradual Migration**: Feature flags enable/disable new components
3. **Performance First**: All operations <1s for typical workloads
4. **Testability**: 90%+ code coverage, mocked external dependencies
5. **Observability**: Structured logging, metrics, traces

### Key Algorithms

#### 1. Analytic Hierarchy Process (AHP)
```
Input: Criteria hierarchy, pairwise comparisons
Process:
  1. Build comparison matrix M (Mij = importance of i vs j)
  2. Calculate priority vector w = principal eigenvector of M
  3. Check consistency: CR = CI/RI < 0.1
Output: Normalized weights for criteria
```

#### 2. GOAP Planning
```
Input: Current state S, Goal state G, Action library A
Process:
  1. Initialize open set with goal node
  2. For each node in open set:
     a. Generate predecessors (actions that satisfy node's state)
     b. Calculate f(n) = g(n) + h(n)  [cost + heuristic]
     c. Add to open set if better path
  3. Terminate when current state reached
Output: Action sequence from S to G
```

#### 3. Constraint Satisfaction
```
Input: Variables V, Domains D, Constraints C
Process:
  1. Choose unassigned variable (using heuristics)
  2. For each value in domain:
     a. Assign value
     b. Propagate constraints (arc consistency)
     c. If consistent, recurse
     d. If solution found, return
  3. Backtrack if no valid assignment
Output: Assignment satisfying all constraints
```

### Performance Targets

| Component | Metric | Target | Measurement |
|-----------|--------|--------|-------------|
| Decision Engine | Decision latency | <100ms | p95 |
| GOAP Planner | 10-step plan | <1s | p95 |
| GOAP Planner | 50-step plan | <5s | p95 |
| Scheduler | 50-task schedule | <500ms | p95 |
| Goal Manager | Conflict detection | <50ms | p95 |
| Full Pipeline | End-to-end | <2s | p95 |
| Memory Footprint | Typical workload | <100MB | RSS |

### Data Structures

#### Enhanced Decision Context
```python
@dataclass
class DecisionContext:
    cognitive_load: float  # 0.0-1.0
    time_pressure: float   # 0.0-1.0 (1.0 = urgent)
    risk_tolerance: float  # 0.0-1.0 (1.0 = high risk ok)
    available_resources: List[str]
    constraints: List[Constraint]
    past_decisions: List[DecisionOutcome]
```

#### World State (GOAP)
```python
WorldState = Dict[str, Any]  # e.g., {"has_data": True, "analysis_complete": False}

@dataclass
class Action:
    name: str
    preconditions: WorldState
    effects: WorldState
    cost: float
    resources: List[str]
```

#### HTN Method
```python
@dataclass
class HTNMethod:
    name: str
    task: str  # Compound task this method decomposes
    preconditions: Callable[[WorldState], bool]
    subtasks: List[str]  # Ordered list of subtasks
    constraints: List[Constraint]
```

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance degradation | Medium | High | Profiling, benchmarks, optimization passes |
| Algorithm complexity bugs | High | Medium | Unit tests, formal verification where possible |
| Integration issues | Medium | High | Incremental integration, feature flags |
| ML model overfitting | Low | Medium | Cross-validation, regularization |
| OR-Tools learning curve | Medium | Medium | Prototypes, documentation, examples |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | Medium | High | Strict phase boundaries, MVP focus |
| Testing effort underestimated | High | Medium | Allocate 30% time for testing |
| Breaking changes | Low | High | Backward compatibility mandatory |
| Resource constraints | Low | Medium | Prioritize phases 1-3 over 4-5 |

---

## Success Criteria

### Functional Requirements
- ✓ Decisions use AHP with consistency checking
- ✓ Pareto frontier computed for multi-objective decisions
- ✓ Plans generated using GOAP with A* search
- ✓ Plans can be replanned when invalidated
- ✓ Goals decomposed using HTN
- ✓ Goal conflicts detected and resolved
- ✓ Tasks scheduled with constraint satisfaction
- ✓ ML model learns from decision outcomes

### Non-Functional Requirements
- ✓ All operations meet performance targets
- ✓ 90%+ test coverage
- ✓ Zero breaking changes to public APIs
- ✓ Memory usage <100MB for typical workload
- ✓ Documentation complete (API docs, examples, migration guide)

### Quality Metrics
- Decision accuracy: >80% of decisions lead to goal progress
- Plan success rate: >75% of plans execute without major replanning
- Goal achievement rate: >70% of goals completed on time
- System uptime: >99% during rollout period

---

## Migration Strategy

### Phase-by-Phase Rollout

**Phase 1: Canary (10% of decisions)**
- Enable new Decision Engine for low-stakes decisions
- Monitor metrics, compare to baseline
- Revert if regression detected

**Phase 2: Expanded (50% of decisions)**
- Enable for broader decision types
- A/B test old vs new
- Collect user feedback

**Phase 3: Full Rollout (100%)**
- Migrate all decisions to new engine
- Keep old code for 1 release as fallback
- Remove old code after stability period

**Repeat for Planning, Goals, Scheduling**

### Rollback Plan
- Feature flags allow instant disable
- Old code paths remain for 2 releases
- Automated alerts trigger rollback conditions:
  - Error rate >5%
  - Latency p95 >2x baseline
  - Memory usage >200MB

---

## Testing Strategy

### Unit Tests (per module)
- Algorithm correctness (AHP, GOAP, HTN)
- Edge cases (empty inputs, conflicts, impossible goals)
- Performance benchmarks
- Target: 90% coverage

### Integration Tests
- Decision → Planning flow
- Planning → Scheduling flow
- Goal → Decision/Planning flow
- End-to-end scenarios

### Behavioral Tests
- Compare outcomes: old vs new implementation
- Real-world scenarios from production logs
- Adversarial cases (conflicting goals, resource starvation)

### Performance Tests
- Latency under load
- Memory usage over time
- Scalability (100+ goals, 1000+ tasks)

---

## Documentation Deliverables

1. **API Documentation**
   - All public classes and methods
   - Usage examples
   - Migration guide from old APIs

2. **Algorithm Explanations**
   - AHP step-by-step
   - GOAP search visualization
   - HTN decomposition examples

3. **Configuration Guide**
   - Feature flags
   - Performance tuning
   - Debugging tips

4. **Runbooks**
   - Common issues and solutions
   - Performance optimization
   - Rollback procedures

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1 | 4 weeks | AHP + Pareto Decision Engine |
| Phase 2 | 4 weeks | GOAP Planning System |
| Phase 3 | 3 weeks | HTN Goal Management |
| Phase 4 | 3 weeks | Constraint-Based Scheduling |
| Phase 5 | 3 weeks | Integration + ML Layer |
| **Total** | **17 weeks** | **Production-ready executive system** |

**Estimated Completion**: Mid-February 2026 (assuming start Oct 31, 2025)

---

## Open Questions

1. **Decision Outcome Collection**: How to automatically determine if a decision was "good"?
   - Proposal: Track goal progress before/after decision
   
2. **Action Library Scope**: How many predefined actions needed for GOAP?
   - Start with 20, expand based on usage patterns
   
3. **HTN Method Authoring**: Who defines decomposition rules?
   - Hybrid: predefined templates + learned patterns

4. **Scheduler Objectives**: Which objective functions to prioritize?
   - Start with makespan, add cognitive load and priority later

5. **ML Training Data**: Where to get initial training data?
   - Bootstrap from simulated scenarios, then learn from production

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Approve phase 1** scope and timeline
3. **Install dependencies** and setup dev environment
4. **Create todo list** with granular tasks
5. **Begin Phase 1, Week 1** implementation

---

*Document Version: 1.0*  
*Last Updated: October 31, 2025*  
*Author: AI Development Team*
