# Week 16 Phase 4: A/B Testing - Quick Reference

Fast reference for A/B testing framework commands and patterns.

---

## Quick Start (30 seconds)

```python
from src.executive.learning import create_experiment_manager, AssignmentMethod

# 1. Create
manager = create_experiment_manager()
exp = manager.create_experiment("Test", ["strategy_a", "strategy_b"])

# 2. Start
manager.start_experiment(exp.experiment_id)

# 3. Run
for i in range(100):
    a = manager.assign_strategy(exp.experiment_id, f"d{i}", f"g{i}")
    result = execute_with_strategy(a.assigned_strategy)
    manager.record_outcome(a.assignment_id, result.success, result.score)

# 4. Analyze
analysis = manager.analyze_experiment(exp.experiment_id)
print(analysis['recommended_strategy'])

# 5. Complete
manager.complete_experiment(exp.experiment_id)
```

---

## Experiment Lifecycle

### Create
```python
exp = manager.create_experiment(
    name="My Experiment",
    strategies=["s1", "s2", "s3"],
    assignment_method=AssignmentMethod.THOMPSON_SAMPLING,
    epsilon=0.1,              # For epsilon-greedy
    min_sample_size=30        # Before recommendations
)
```

### Control
```python
manager.start_experiment(exp.experiment_id)      # DRAFT → ACTIVE
manager.pause_experiment(exp.experiment_id)      # ACTIVE → PAUSED  
manager.complete_experiment(exp.experiment_id)   # ACTIVE → COMPLETED
```

### Query
```python
exp = manager.get_experiment(exp_id)
exps = manager.list_experiments(status=ExperimentStatus.ACTIVE)
summary = manager.get_experiment_summary(exp_id)
```

---

## Assignment Methods

### Random (Baseline)
```python
assignment_method=AssignmentMethod.RANDOM
```
- Equal probability for all strategies
- No exploitation

### Epsilon-Greedy (Simple)
```python
assignment_method=AssignmentMethod.EPSILON_GREEDY
epsilon=0.1  # 10% explore, 90% exploit
```
- ε% random, (1-ε)% best strategy
- Fast convergence

### Thompson Sampling (Optimal)
```python
assignment_method=AssignmentMethod.THOMPSON_SAMPLING
```
- Bayesian approach
- Adaptive exploration
- No hyperparameters

---

## Running Experiments

### Assign Strategy
```python
assignment = manager.assign_strategy(
    experiment_id=exp.experiment_id,
    decision_id="decision_123",
    goal_id="goal_456",
    context={"user": "alice"}  # Optional
)

strategy = assignment.assigned_strategy
```

### Record Outcome
```python
manager.record_outcome(
    assignment_id=assignment.assignment_id,
    success=True,
    outcome_score=0.85,
    execution_time_seconds=10.5,
    decision_confidence=0.75,
    metadata={"details": "..."}
)
```

---

## Analysis

### Get Performances
```python
perfs = manager.get_strategy_performances(exp_id)

for strategy, perf in perfs.items():
    print(f"{strategy}:")
    print(f"  N: {perf.total_assignments}")
    print(f"  Success: {perf.success_rate:.1%}")
    print(f"  CI: [{perf.success_rate_ci_lower:.1%}, "
          f"{perf.success_rate_ci_upper:.1%}]")
    print(f"  Score: {perf.avg_outcome_score:.2f}")
```

### Get Recommendation
```python
analysis = manager.analyze_experiment(exp_id)

print(f"Winner: {analysis['recommended_strategy']}")
print(f"Confidence: {analysis['confidence']}")  # low/medium/high
print(f"Reason: {analysis['reason']}")
```

### Get Summary
```python
summary = manager.get_experiment_summary(exp_id)

print(f"Total: {summary['total_assignments']}")
print(f"Outcomes: {summary['total_outcomes']}")
print(f"Completion: {summary['completion_rate']:.1%}")
print(f"Duration: {summary['duration_hours']:.1f}h")
```

---

## Statistical Tests

### Chi-Square (Categorical)
```python
from src.executive.learning import chi_square_test

result = chi_square_test(
    strategy_a_successes=70, strategy_a_failures=30,
    strategy_b_successes=50, strategy_b_failures=50
)

print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Winner: {result.winner}")
```

### t-Test (Continuous)
```python
from src.executive.learning import t_test

result = t_test(
    strategy_a_scores=[0.8, 0.9, 0.7, 0.85],
    strategy_b_scores=[0.6, 0.65, 0.7, 0.55]
)
```

### Mann-Whitney (Non-Parametric)
```python
from src.executive.learning import mann_whitney_test

result = mann_whitney_test(
    strategy_a_scores=[...],
    strategy_b_scores=[...]
)
```

### Effect Size
```python
from src.executive.learning import cohens_d, interpret_effect_size

d = cohens_d(group_a, group_b)
interp = interpret_effect_size(d)  # "negligible"/"small"/"medium"/"large"
```

### Confidence Intervals
```python
from src.executive.learning import (
    calculate_confidence_interval,
    calculate_proportion_confidence_interval
)

# For proportions (success rate)
lower, upper = calculate_proportion_confidence_interval(
    successes=70, total=100, confidence_level=0.95
)

# For continuous metrics
lower, upper = calculate_confidence_interval(
    data=[0.7, 0.8, 0.9], confidence_level=0.95
)
```

---

## DecisionEngine Integration

### Setup
```python
from src.executive import DecisionEngine
from src.executive.learning import create_experiment_manager

manager = create_experiment_manager()
exp = manager.create_experiment("Test", ["s1", "s2"])
manager.start_experiment(exp.experiment_id)

engine = DecisionEngine(experiment_manager=manager)
```

### Make Decision
```python
result = engine.make_decision(
    options=options,
    criteria=criteria,
    experiment_id=exp.experiment_id,
    context={'goal_id': 'g123'}
)

# Get assignment info
assignment = result.metadata['experiment_assignment']
strategy = assignment['assigned_strategy']
assignment_id = assignment['assignment_id']
```

### Record Outcome
```python
engine.record_experiment_outcome(
    assignment_id=assignment_id,
    success=True,
    outcome_score=0.85,
    execution_time_seconds=10.0
)
```

---

## Common Patterns

### Complete Workflow
```python
# 1. Setup
manager = create_experiment_manager()
exp = manager.create_experiment("Test", ["s1", "s2"], 
    assignment_method=AssignmentMethod.THOMPSON_SAMPLING)
manager.start_experiment(exp.experiment_id)

# 2. Run loop
for i in range(100):
    a = manager.assign_strategy(exp.experiment_id, f"d{i}", f"g{i}")
    success, score = execute(a.assigned_strategy)
    manager.record_outcome(a.assignment_id, success, score)

# 3. Analyze
manager.complete_experiment(exp.experiment_id)
analysis = manager.analyze_experiment(exp.experiment_id)
winner = analysis['recommended_strategy']
```

### Early Stopping
```python
for i in range(1000):
    # ... assign and record ...
    
    if i % 50 == 0 and i >= 100:
        analysis = manager.analyze_experiment(exp.experiment_id)
        if analysis['confidence'] == 'high':
            print(f"Clear winner at {i} samples")
            break
```

### Multiple Experiments
```python
experiments = {}
for strategy_set in [["s1", "s2"], ["s1", "s3"], ["s2", "s3"]]:
    exp = manager.create_experiment(
        name=f"Test {strategy_set}",
        strategies=strategy_set
    )
    manager.start_experiment(exp.experiment_id)
    experiments[tuple(strategy_set)] = exp.experiment_id
```

### Monitoring Progress
```python
def check_progress(exp_id):
    summary = manager.get_experiment_summary(exp_id)
    print(f"Assignments: {summary['total_assignments']}")
    print(f"Completion: {summary['completion_rate']:.1%}")
    
    for s, p in summary['performance'].items():
        print(f"  {s}: {p['success_rate']:.1%} success, "
              f"n={p['total_assignments']}")
```

---

## Configuration

### Storage Location
```python
from pathlib import Path

manager = create_experiment_manager(
    storage_dir=Path("experiments/prod")
)
```

### Experiment Options
```python
exp = manager.create_experiment(
    name="My Test",
    strategies=["s1", "s2"],
    assignment_method=AssignmentMethod.EPSILON_GREEDY,
    description="Testing strategies",
    epsilon=0.2,              # Exploration rate
    confidence_level=0.95,    # For statistical tests
    min_sample_size=50,       # Before recommendations
    metadata={"owner": "alice"}
)
```

---

## Enums

### ExperimentStatus
```python
from src.executive.learning import ExperimentStatus

ExperimentStatus.DRAFT       # Created, not started
ExperimentStatus.ACTIVE      # Running
ExperimentStatus.PAUSED      # Temporarily stopped
ExperimentStatus.COMPLETED   # Finished
ExperimentStatus.CANCELLED   # Stopped early
```

### AssignmentMethod
```python
from src.executive.learning import AssignmentMethod

AssignmentMethod.RANDOM              # Uniform random
AssignmentMethod.EPSILON_GREEDY      # Explore ε%, exploit 1-ε%
AssignmentMethod.THOMPSON_SAMPLING   # Bayesian Beta distributions
```

### SignificanceTest
```python
from src.executive.learning import SignificanceTest

SignificanceTest.CHI_SQUARE    # For categorical (success/failure)
SignificanceTest.T_TEST        # For continuous (parametric)
SignificanceTest.MANN_WHITNEY  # For continuous (non-parametric)
```

---

## Interpretation Guide

### p-values
- **p < 0.001**: Extremely significant (***) 
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)
- **p ≥ 0.05**: Not significant (ns)

### Effect Sizes (Cohen's d)
- **|d| < 0.2**: Negligible - ignore
- **0.2 ≤ |d| < 0.5**: Small - minor difference
- **0.5 ≤ |d| < 0.8**: Medium - moderate difference  
- **|d| ≥ 0.8**: Large - substantial difference

### Confidence Levels
- **High**: p < 0.05 AND |d| ≥ 0.5 AND n ≥ 30
- **Medium**: p < 0.05 AND |d| ≥ 0.2 AND n ≥ 30
- **Low**: Not significant OR |d| < 0.2 OR n < 30

### Sample Size Guidelines
- **2 strategies**: 50-100 each = 100-200 total
- **3-5 strategies**: 100-200 each = 300-1000 total
- **5+ strategies**: 200+ each = 1000+ total

---

## Troubleshooting

### No recommendation
```python
# Check sample sizes
summary = manager.get_experiment_summary(exp_id)
for s, p in summary['performance'].items():
    print(f"{s}: {p['total_assignments']} samples")
# Need 30+ per strategy
```

### Low confidence
```python
# Check effect size
analysis = manager.analyze_experiment(exp_id)
if 'chi_square_test' in analysis:
    d = analysis['chi_square_test']['effect_size']
    print(f"Effect size: {d:.2f} ({interpret_effect_size(d)})")
# May need more samples or strategies truly equivalent
```

### Assignment error
```python
# Check experiment status
exp = manager.get_experiment(exp_id)
print(f"Status: {exp.status}")
# Must be ACTIVE to assign
```

---

## File Locations

```
data/experiments/
├── experiment_{id}.json       # Experiment metadata
├── assignments_{id}.json      # All assignments
└── outcomes_{id}.json         # All outcomes
```

---

## Import Cheat Sheet

```python
# Core
from src.executive.learning import (
    create_experiment_manager,
    ExperimentManager,
    StrategyExperiment,
    ExperimentAssignment,
    StrategyOutcome,
    ExperimentStatus,
    AssignmentMethod,
)

# Analysis
from src.executive.learning import (
    StrategyPerformance,
    ComparisonResult,
    SignificanceTest,
    chi_square_test,
    t_test,
    mann_whitney_test,
    cohens_d,
    interpret_effect_size,
    calculate_confidence_interval,
    calculate_proportion_confidence_interval,
    recommend_strategy,
)

# Integration
from src.executive import DecisionEngine
```

---

## Performance Notes

- **Assignment latency**: <5ms (random), <10ms (Thompson)
- **Record outcome**: <5ms
- **Analysis**: <50ms for 100 samples per strategy
- **Persistence**: <20ms save/load
- **Memory**: <10MB for 1000 assignments

---

## One-Liners

```python
# Create and start
exp = manager.create_experiment("Test", ["s1", "s2"])
manager.start_experiment(exp.experiment_id)

# Assign and record
a = manager.assign_strategy(exp.experiment_id, "d1", "g1")
manager.record_outcome(a.assignment_id, True, 0.85)

# Analyze
winner = manager.analyze_experiment(exp.experiment_id)['recommended_strategy']

# Complete
manager.complete_experiment(exp.experiment_id)
```

---

## See Also

- **Completion Summary**: `WEEK_16_PHASE_4_COMPLETION_SUMMARY.md`
- **Comprehensive Guide**: `WEEK_16_PHASE_4_AB_TESTING.md`
- **Tests**: `tests/test_experiment_ab_testing.py`
- **Source**: `src/executive/learning/experiment_manager.py`
