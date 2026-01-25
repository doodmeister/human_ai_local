# Week 16 Phase 4: A/B Testing Framework - Comprehensive Guide

Complete guide to using the A/B testing framework for empirical strategy comparison.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Experiment Lifecycle](#experiment-lifecycle)
5. [Assignment Methods](#assignment-methods)
6. [Statistical Analysis](#statistical-analysis)
7. [DecisionEngine Integration](#decisionengine-integration)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### What is A/B Testing?

A/B testing (or split testing) is a method of comparing multiple variants to determine which performs best. In our context, we compare decision strategies to empirically determine which produces better outcomes.

### Why Use A/B Testing?

- **Data-Driven**: Make decisions based on empirical evidence, not assumptions
- **Objective**: Removes bias from strategy selection
- **Scientific**: Uses statistical methods to ensure validity
- **Adaptive**: Supports exploration/exploitation trade-offs
- **Automated**: Provides automatic strategy recommendations

### Key Concepts

**Experiment**: A controlled test comparing 2+ strategies  
**Assignment**: Which strategy a specific decision uses  
**Outcome**: The result of executing a decision  
**Statistical Significance**: Confidence that differences are real, not random  
**Effect Size**: Magnitude of performance difference  

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│               ExperimentManager                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  Experiment Orchestration                   │   │
│  │  - Create/Start/Pause/Complete              │   │
│  │  - Assign strategies                        │   │
│  │  - Record outcomes                          │   │
│  │  - Aggregate metrics                        │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        │
                        ├─── Assignment Methods
                        │    ├─ Random
                        │    ├─ Epsilon-Greedy
                        │    └─ Thompson Sampling
                        │
                        ├─── Storage (JSON)
                        │    ├─ experiment_{id}.json
                        │    ├─ assignments_{id}.json
                        │    └─ outcomes_{id}.json
                        │
                        └─── Analysis
                             ├─ StrategyPerformance
                             ├─ Statistical Tests
                             └─ Recommendations

┌─────────────────────────────────────────────────────┐
│              ExperimentAnalyzer                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  Statistical Analysis                       │   │
│  │  - Chi-square test                          │   │
│  │  - t-test, Mann-Whitney U                   │   │
│  │  - Cohen's d effect size                    │   │
│  │  - Confidence intervals                     │   │
│  │  - Automated recommendations                │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               DecisionEngine                         │
│  ┌─────────────────────────────────────────────┐   │
│  │  Integration Layer                          │   │
│  │  - make_decision(experiment_id=...)        │   │
│  │  - Auto strategy assignment                 │   │
│  │  - record_experiment_outcome()             │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Create Experiment
   ↓
2. Start Experiment
   ↓
3. For each decision:
   ├─ Assign Strategy (random/epsilon-greedy/thompson)
   ├─ Execute Decision with Assigned Strategy
   └─ Record Outcome (success, score, time)
   ↓
4. Analyze Results
   ├─ Aggregate Metrics per Strategy
   ├─ Statistical Tests (chi-square, t-test, etc.)
   └─ Generate Recommendation
   ↓
5. Complete Experiment
```

---

## Getting Started

### Installation

Dependencies already in `requirements.txt`:
```bash
pip install scipy>=1.10.0  # Statistical tests
pip install numpy>=1.24.0  # Numerical operations
```

### Basic Example

```python
from src.executive.learning import create_experiment_manager, AssignmentMethod

# 1. Create manager
manager = create_experiment_manager()

# 2. Create experiment
experiment = manager.create_experiment(
    name="Strategy Comparison",
    strategies=["weighted_scoring", "ahp"],
    assignment_method=AssignmentMethod.RANDOM
)

# 3. Start experiment
manager.start_experiment(experiment.experiment_id)

# 4. Run decisions
for i in range(50):
    # Assign strategy
    assignment = manager.assign_strategy(
        experiment_id=experiment.experiment_id,
        decision_id=f"decision_{i}",
        goal_id=f"goal_{i}"
    )
    
    # Execute with assigned strategy
    success = execute_decision(assignment.assigned_strategy)
    
    # Record outcome
    manager.record_outcome(
        assignment_id=assignment.assignment_id,
        success=success,
        outcome_score=0.8 if success else 0.3
    )

# 5. Analyze
analysis = manager.analyze_experiment(experiment.experiment_id)
print(f"Winner: {analysis['recommended_strategy']}")

# 6. Complete
manager.complete_experiment(experiment.experiment_id)
```

---

## Experiment Lifecycle

### 1. Create Experiment

```python
experiment = manager.create_experiment(
    name="My Experiment",
    strategies=["strategy_a", "strategy_b", "strategy_c"],
    assignment_method=AssignmentMethod.THOMPSON_SAMPLING,
    description="Testing which strategy works best",
    epsilon=0.1,              # For epsilon-greedy
    confidence_level=0.95,    # For statistical tests
    min_sample_size=30,       # Minimum before recommendations
    metadata={"owner": "user@example.com"}
)
```

**Status**: DRAFT

### 2. Start Experiment

```python
manager.start_experiment(experiment.experiment_id)
```

**Status**: DRAFT → ACTIVE  
**Effect**: Experiment can now assign strategies

### 3. Run Experiment

```python
# Assign strategies
assignment = manager.assign_strategy(
    experiment_id=experiment.experiment_id,
    decision_id="unique_decision_id",
    goal_id="goal_id",
    context={"user_id": "123"}  # Optional metadata
)

# Record outcomes
manager.record_outcome(
    assignment_id=assignment.assignment_id,
    success=True,
    outcome_score=0.85,
    execution_time_seconds=12.5,
    decision_confidence=0.75,
    metadata={"details": "..."}
)
```

### 4. Pause Experiment (Optional)

```python
manager.pause_experiment(experiment.experiment_id)
```

**Status**: ACTIVE → PAUSED  
**Effect**: Temporarily stop assigning strategies

### 5. Resume Experiment

```python
manager.start_experiment(experiment.experiment_id)  # Works for paused too
```

**Status**: PAUSED → ACTIVE

### 6. Complete Experiment

```python
manager.complete_experiment(experiment.experiment_id)
```

**Status**: ACTIVE/PAUSED → COMPLETED  
**Effect**: Experiment finalized, can analyze results

### State Diagram

```
    [DRAFT]
       │
       │ start_experiment()
       ↓
    [ACTIVE] ←─────┐
       │            │ start_experiment()
       │ pause()    │
       ↓            │
    [PAUSED] ───────┘
       │
       │ complete()
       ↓
  [COMPLETED]
```

---

## Assignment Methods

### Random Assignment

**Use Case**: Baseline comparison, no prior knowledge  
**Algorithm**: Uniform random selection  

```python
assignment_method=AssignmentMethod.RANDOM
```

**Behavior**:
- Each strategy has equal probability (1/N)
- No exploitation of performance data
- Good for unbiased comparison

**Example**:
```python
experiment = manager.create_experiment(
    name="Random Test",
    strategies=["s1", "s2", "s3"],
    assignment_method=AssignmentMethod.RANDOM
)
# Each strategy gets ~33.3% of assignments
```

### Epsilon-Greedy

**Use Case**: Balance exploration and exploitation  
**Algorithm**: Explore with probability ε, exploit 1-ε  

```python
assignment_method=AssignmentMethod.EPSILON_GREEDY
epsilon=0.1  # 10% exploration
```

**Behavior**:
- **Explore** (ε% of time): Random strategy
- **Exploit** ((1-ε)% of time): Best performing strategy
- Simple and effective
- Converges quickly to best strategy

**Example**:
```python
experiment = manager.create_experiment(
    name="Epsilon-Greedy Test",
    strategies=["s1", "s2"],
    assignment_method=AssignmentMethod.EPSILON_GREEDY,
    epsilon=0.2  # 20% exploration, 80% exploitation
)
# After 100 assignments:
# - Best strategy gets ~80% of new assignments
# - Other strategies get ~20% combined
```

**Tuning Epsilon**:
- **ε = 0.0**: Pure exploitation (never explore)
- **ε = 0.1**: Conservative (10% explore) - DEFAULT
- **ε = 0.3**: Balanced (30% explore)
- **ε = 1.0**: Pure exploration (always random)

### Thompson Sampling

**Use Case**: Optimal exploration/exploitation, Bayesian approach  
**Algorithm**: Sample from Beta distributions  

```python
assignment_method=AssignmentMethod.THOMPSON_SAMPLING
```

**Behavior**:
- Maintains Beta(α, β) distribution per strategy
- α = successes + 1, β = failures + 1 (Jeffrey's prior)
- Samples from each distribution
- Assigns strategy with highest sample
- Naturally balances exploration/exploitation

**Example**:
```python
experiment = manager.create_experiment(
    name="Thompson Sampling Test",
    strategies=["s1", "s2"],
    assignment_method=AssignmentMethod.THOMPSON_SAMPLING
)
# Automatically adapts assignment probabilities based on performance
```

**Advantages**:
- **Bayesian**: Incorporates uncertainty
- **Adaptive**: More exploration when uncertain
- **Optimal**: Minimizes regret in theory
- **No tuning**: No hyperparameters like epsilon

**Distribution Evolution**:
```
Initial: Both strategies Beta(1, 1) - uniform
After 10 assignments:
- s1: 7 successes, 3 failures → Beta(8, 4)
- s2: 4 successes, 6 failures → Beta(5, 7)
→ s1 gets ~70% of future assignments
```

### Comparison Table

| Method | Exploration | Convergence | Tuning | Use Case |
|--------|-------------|-------------|--------|----------|
| Random | 100% | Never | None | Baseline |
| Epsilon-Greedy | ε% fixed | Fast | Epsilon | Simple, fast |
| Thompson Sampling | Adaptive | Optimal | None | Best overall |

---

## Statistical Analysis

### Performance Metrics

For each strategy, we calculate:

```python
performances = manager.get_strategy_performances(experiment_id)

for strategy, perf in performances.items():
    print(f"{strategy}:")
    print(f"  Total: {perf.total_assignments}")
    print(f"  Success Rate: {perf.success_rate:.2%}")
    print(f"  95% CI: [{perf.success_rate_ci_lower:.2%}, "
          f"{perf.success_rate_ci_upper:.2%}]")
    print(f"  Avg Score: {perf.avg_outcome_score:.3f} ± "
          f"{perf.std_outcome_score:.3f}")
    print(f"  Avg Time: {perf.avg_execution_time:.1f}s")
```

### Confidence Intervals

**For Proportions (Success Rate)**:
```python
from src.executive.learning import calculate_proportion_confidence_interval

lower, upper = calculate_proportion_confidence_interval(
    successes=70,
    total=100,
    confidence_level=0.95
)
# Wilson score interval: (0.60, 0.79)
```

**For Continuous Metrics**:
```python
from src.executive.learning import calculate_confidence_interval

scores = [0.7, 0.8, 0.9, 0.75, 0.85]
lower, upper = calculate_confidence_interval(
    data=scores,
    confidence_level=0.95
)
# t-distribution interval: (0.71, 0.89)
```

### Hypothesis Testing

#### Chi-Square Test (Categorical)

**Use When**: Comparing success/failure rates

```python
from src.executive.learning import chi_square_test

result = chi_square_test(
    strategy_a_successes=70, strategy_a_failures=30,
    strategy_b_successes=50, strategy_b_failures=50,
    alpha=0.05
)

print(f"Chi-square: {result.test_statistic:.2f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Winner: {result.winner}")
print(result.interpretation)
```

**Output**:
```
Chi-square: 8.33
p-value: 0.0039
Significant: True
Winner: Strategy A
Strategy A has significantly better success rate (A: 70.00%, B: 50.00%, p=0.0039)
```

#### t-Test (Continuous, Parametric)

**Use When**: Comparing continuous metrics (scores), assuming normal distribution

```python
from src.executive.learning import t_test

result = t_test(
    strategy_a_scores=[0.8, 0.9, 0.85, 0.7, 0.95],
    strategy_b_scores=[0.6, 0.65, 0.7, 0.55, 0.6],
    alpha=0.05
)

print(f"t-statistic: {result.test_statistic:.2f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Cohen's d: {result.effect_size:.2f} ({result.effect_size_interpretation})")
```

#### Mann-Whitney U Test (Non-Parametric)

**Use When**: Comparing continuous metrics, **not** assuming normal distribution

```python
from src.executive.learning import mann_whitney_test

result = mann_whitney_test(
    strategy_a_scores=[...],
    strategy_b_scores=[...],
    alpha=0.05
)
```

**When to Use Each**:
- **Chi-square**: Success/failure counts
- **t-test**: Continuous scores, normal distribution
- **Mann-Whitney**: Continuous scores, skewed distribution

### Effect Size

**Cohen's d** measures standardized difference:

```python
from src.executive.learning import cohens_d, interpret_effect_size

d = cohens_d(group_a, group_b)
interpretation = interpret_effect_size(d)
```

**Interpretation**:
- **|d| < 0.2**: Negligible - differences too small to matter
- **0.2 ≤ |d| < 0.5**: Small - noticeable but minor
- **0.5 ≤ |d| < 0.8**: Medium - moderate practical significance
- **|d| ≥ 0.8**: Large - substantial practical significance

**Example**:
```python
# Strategy A: avg=0.85, std=0.1
# Strategy B: avg=0.65, std=0.1
d = cohens_d([0.85]*10, [0.65]*10)
# d = 2.0 → "large" effect
```

### Automated Recommendations

```python
analysis = manager.analyze_experiment(experiment_id)

print(f"Recommended: {analysis['recommended_strategy']}")
print(f"Confidence: {analysis['confidence']}")  # low/medium/high
print(f"Reason: {analysis['reason']}")
```

**Confidence Levels**:
- **High**: Statistically significant with large effect size
- **Medium**: Statistically significant with small/medium effect
- **Low**: Not significant or insufficient data

**Minimum Sample Size**: Default 30 per strategy before recommendations

---

## DecisionEngine Integration

### Setup

```python
from src.executive import DecisionEngine
from src.executive.learning import create_experiment_manager

# Create experiment manager
manager = create_experiment_manager()

# Create experiment
experiment = manager.create_experiment(
    name="Strategy Test",
    strategies=["weighted_scoring", "ahp", "pareto"]
)
manager.start_experiment(experiment.experiment_id)

# Create engine with experiment support
engine = DecisionEngine(
    enable_ml_predictions=True,
    experiment_manager=manager
)
```

### Making Decisions

```python
# Without experiment (manual strategy)
result = engine.make_decision(
    options=options,
    criteria=criteria,
    strategy="weighted_scoring"
)

# With experiment (auto-assigned strategy)
result = engine.make_decision(
    options=options,
    criteria=criteria,
    experiment_id=experiment.experiment_id,
    context={'goal_id': 'goal_123', 'decision_id': 'dec_456'}
)

# Check assigned strategy
assignment_info = result.metadata.get('experiment_assignment')
if assignment_info:
    print(f"Assigned: {assignment_info['assigned_strategy']}")
    print(f"Assignment ID: {assignment_info['assignment_id']}")
```

### Recording Outcomes

```python
# Get assignment ID from result
assignment_id = result.metadata['experiment_assignment']['assignment_id']

# Execute decision and record outcome
success, score, duration = execute_decision(result)

engine.record_experiment_outcome(
    assignment_id=assignment_id,
    success=success,
    outcome_score=score,
    execution_time_seconds=duration,
    metadata={'details': '...'}
)
```

### Full Integration Example

```python
from src.executive import DecisionEngine
from src.executive.learning import create_experiment_manager, AssignmentMethod

# Setup
manager = create_experiment_manager()
exp = manager.create_experiment(
    name="Task Selection Strategies",
    strategies=["weighted_scoring", "ahp"],
    assignment_method=AssignmentMethod.THOMPSON_SAMPLING
)
manager.start_experiment(exp.experiment_id)

engine = DecisionEngine(experiment_manager=manager)

# Run 100 decisions
for i in range(100):
    # Make decision with auto-assigned strategy
    result = engine.make_decision(
        options=task_options,
        criteria=task_criteria,
        experiment_id=exp.experiment_id,
        context={'goal_id': f'goal_{i}'}
    )
    
    # Execute
    task = result.recommended_option.data['task']
    success, score, duration = execute_task(task)
    
    # Record outcome
    assignment_id = result.metadata['experiment_assignment']['assignment_id']
    engine.record_experiment_outcome(
        assignment_id=assignment_id,
        success=success,
        outcome_score=score,
        execution_time_seconds=duration
    )

# Complete and analyze
manager.complete_experiment(exp.experiment_id)
analysis = manager.analyze_experiment(exp.experiment_id)

print(f"Winner: {analysis['recommended_strategy']}")
print(f"Confidence: {analysis['confidence']}")
```

---

## Advanced Usage

### Custom Storage Location

```python
from pathlib import Path

manager = create_experiment_manager(
    storage_dir=Path("experiments/production")
)
```

### Multiple Experiments

```python
# Create multiple experiments
exp1 = manager.create_experiment("Test A", ["s1", "s2"])
exp2 = manager.create_experiment("Test B", ["s1", "s3"])

manager.start_experiment(exp1.experiment_id)
manager.start_experiment(exp2.experiment_id)

# Run in parallel
for decision in decisions:
    if condition_a:
        assign_from(exp1.experiment_id)
    else:
        assign_from(exp2.experiment_id)
```

### Filtering and Querying

```python
# List experiments by status
from src.executive.learning import ExperimentStatus

active = manager.list_experiments(status=ExperimentStatus.ACTIVE)
completed = manager.list_experiments(status=ExperimentStatus.COMPLETED)

# Get experiment summary
summary = manager.get_experiment_summary(experiment_id)
print(f"Completion Rate: {summary['completion_rate']:.1%}")
print(f"Duration: {summary['duration_hours']:.1f} hours")

# Get detailed performances
performances = manager.get_strategy_performances(experiment_id)
```

### Persistence Across Sessions

```python
# Session 1: Create and run
manager1 = create_experiment_manager(storage_dir=Path("data/exp"))
exp = manager1.create_experiment("Test", ["s1", "s2"])
# ... run assignments ...

# Session 2: Load and continue
manager2 = create_experiment_manager(storage_dir=Path("data/exp"))
exp_loaded = manager2.get_experiment(exp.experiment_id)
# All data automatically loaded from JSON
```

---

## Best Practices

### 1. Sample Size

**Minimum**: 30 samples per strategy  
**Recommended**: 100+ samples per strategy  
**Rule of Thumb**: More strategies = more samples needed

```python
# Good
experiment = manager.create_experiment(
    strategies=["s1", "s2"],
    min_sample_size=50  # 50 per strategy = 100 total
)

# Be careful
experiment = manager.create_experiment(
    strategies=["s1", "s2", "s3", "s4", "s5"],
    min_sample_size=30  # 30 * 5 = 150 total minimum
)
```

### 2. Assignment Method Selection

**Use Random** when:
- Need unbiased baseline
- Don't care about performance during experiment
- Have unlimited samples

**Use Epsilon-Greedy** when:
- Want quick convergence
- Can tolerate some suboptimal decisions
- Understand your domain (can tune epsilon)

**Use Thompson Sampling** when:
- Want optimal long-term performance
- Can't waste many samples on bad strategies
- Don't want to tune hyperparameters

### 3. Experiment Duration

**Too Short**: Risk false conclusions  
**Too Long**: Waste resources on inferior strategies

**Recommended Duration**:
- **2 strategies**: 50-100 assignments each
- **3-5 strategies**: 100-200 assignments each
- **5+ strategies**: 200+ assignments each

### 4. Statistical Significance

Don't trust results if:
- `p_value > 0.05` (not significant at 95% confidence)
- Sample size < 30 per strategy
- Effect size is "negligible" (d < 0.2)

**Check All Three**:
```python
if (result.is_significant and 
    perf.total_assignments >= 30 and
    abs(result.effect_size) >= 0.2):
    # Trustworthy result
```

### 5. Context Metadata

Always include context for later analysis:

```python
assignment = manager.assign_strategy(
    experiment_id=exp.experiment_id,
    decision_id=decision.id,
    goal_id=goal.id,
    context={
        'user_id': user.id,
        'time_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'cognitive_load': system.get_cognitive_load()
    }
)
```

### 6. Monitoring

Check progress regularly:

```python
summary = manager.get_experiment_summary(exp.experiment_id)

if summary['total_assignments'] % 50 == 0:
    print(f"Progress: {summary['total_assignments']} assignments")
    
    for strategy, perf in summary['performance'].items():
        print(f"  {strategy}: {perf['success_rate']:.1%} success")
```

### 7. Early Stopping

Stop early if one strategy is clearly winning:

```python
if summary['total_assignments'] >= 100:
    analysis = manager.analyze_experiment(exp.experiment_id)
    
    if analysis['confidence'] == 'high':
        print(f"Clear winner: {analysis['recommended_strategy']}")
        manager.complete_experiment(exp.experiment_id)
```

---

## API Reference

### ExperimentManager

**Constructor**:
```python
manager = create_experiment_manager(storage_dir: Optional[Path] = None)
```

**Methods**:

`create_experiment(name, strategies, assignment_method, ...)` → StrategyExperiment  
`start_experiment(experiment_id)` → None  
`pause_experiment(experiment_id)` → None  
`complete_experiment(experiment_id)` → None  
`assign_strategy(experiment_id, decision_id, goal_id, context)` → ExperimentAssignment  
`record_outcome(assignment_id, success, outcome_score, ...)` → StrategyOutcome  
`get_experiment(experiment_id)` → Optional[StrategyExperiment]  
`list_experiments(status)` → List[StrategyExperiment]  
`get_experiment_summary(experiment_id)` → Dict  
`get_strategy_performances(experiment_id)` → Dict[str, StrategyPerformance]  
`analyze_experiment(experiment_id)` → Dict  

### StrategyExperiment

**Attributes**:
```python
experiment_id: str
name: str
strategies: List[str]
assignment_method: AssignmentMethod
status: ExperimentStatus
description: str = ""
epsilon: float = 0.1
confidence_level: float = 0.95
min_sample_size: int = 30
created_at: datetime
started_at: Optional[datetime]
completed_at: Optional[datetime]
metadata: Dict[str, Any]
```

### ExperimentAssignment

**Attributes**:
```python
assignment_id: str
experiment_id: str
decision_id: str
goal_id: str
assigned_strategy: str
assignment_timestamp: datetime
assignment_method: AssignmentMethod
metadata: Dict[str, Any]
```

### StrategyOutcome

**Attributes**:
```python
assignment_id: str
success: bool
outcome_score: float
execution_time_seconds: Optional[float]
decision_confidence: float
completion_timestamp: datetime
metadata: Dict[str, Any]
```

### Statistical Functions

```python
calculate_confidence_interval(data, confidence_level) → Tuple[float, float]
calculate_proportion_confidence_interval(successes, total, confidence_level) → Tuple[float, float]
cohens_d(group_a, group_b) → float
interpret_effect_size(d) → str
chi_square_test(a_succ, a_fail, b_succ, b_fail, alpha) → ComparisonResult
t_test(a_scores, b_scores, alpha) → ComparisonResult
mann_whitney_test(a_scores, b_scores, alpha) → ComparisonResult
recommend_strategy(performances, alpha, min_sample_size) → Dict
```

---

## Troubleshooting

### Problem: No strategy recommended

**Symptoms**:
```python
analysis['recommended_strategy'] is None
analysis['confidence'] == 'none'
```

**Causes**:
1. No performance data
2. Insufficient sample size

**Solutions**:
```python
# Check sample sizes
summary = manager.get_experiment_summary(exp.experiment_id)
for strategy, perf in summary['performance'].items():
    if perf['total_assignments'] < 30:
        print(f"Need more data for {strategy}")
```

### Problem: Low confidence recommendation

**Symptoms**:
```python
analysis['confidence'] == 'low'
```

**Causes**:
1. Strategies performing similarly
2. High variance in outcomes
3. Insufficient samples

**Solutions**:
- Collect more samples (100+ per strategy)
- Check if strategies are truly different
- Consider effect size (may not matter if d < 0.2)

### Problem: Assignment failed

**Symptoms**:
```python
ValueError: Experiment ... not active
```

**Cause**: Experiment not started

**Solution**:
```python
manager.start_experiment(experiment_id)
```

### Problem: Unexpected assignment distribution

**Symptoms**: One strategy getting all assignments with epsilon-greedy

**Cause**: One strategy performing much better

**Solution**: This is expected behavior! Epsilon-greedy exploits best strategy.

---

**Next**: See [Quick Reference Guide](WEEK_16_PHASE_4_QUICK_REF.md) for command cheat sheet.
