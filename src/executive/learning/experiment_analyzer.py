"""
Experiment Analyzer - Week 16 Phase 4

Statistical analysis for strategy experiments.
Provides significance testing, effect size calculation, confidence intervals,
and strategy recommendations based on empirical performance.

Key responsibilities:
- Calculate strategy performance metrics
- Statistical significance testing
- Effect size estimation
- Confidence intervals
- Strategy recommendations
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, cast
from enum import Enum

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SignificanceTest(Enum):
    """Type of statistical significance test."""
    CHI_SQUARE = "chi_square"  # For categorical outcomes (success/failure)
    T_TEST = "t_test"  # For continuous metrics (parametric)
    MANN_WHITNEY = "mann_whitney"  # For continuous metrics (non-parametric)


@dataclass
class StrategyPerformance:
    """
    Aggregated performance metrics for a strategy.
    
    Summarizes all outcomes for a strategy in an experiment.
    """
    strategy_name: str
    total_assignments: int
    success_count: int
    failure_count: int
    
    # Success metrics
    success_rate: float
    success_rate_ci_lower: float  # Lower bound of confidence interval
    success_rate_ci_upper: float  # Upper bound of confidence interval
    
    # Outcome score metrics
    avg_outcome_score: float
    std_outcome_score: float
    outcome_scores: List[float]  # All scores for analysis
    
    # Execution time metrics
    avg_execution_time: float
    std_execution_time: float
    
    # Decision confidence metrics
    avg_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'total_assignments': self.total_assignments,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_rate,
            'success_rate_ci_lower': self.success_rate_ci_lower,
            'success_rate_ci_upper': self.success_rate_ci_upper,
            'avg_outcome_score': self.avg_outcome_score,
            'std_outcome_score': self.std_outcome_score,
            'avg_execution_time': self.avg_execution_time,
            'std_execution_time': self.std_execution_time,
            'avg_confidence': self.avg_confidence
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing two strategies.
    
    Contains test statistics, p-values, effect sizes, and interpretation.
    """
    strategy_a: str
    strategy_b: str
    test_type: SignificanceTest
    
    # Test results
    test_statistic: float
    p_value: float
    is_significant: bool  # p < alpha
    alpha: float  # Significance level (e.g., 0.05)
    
    # Effect size
    effect_size: float  # Cohen's d or similar
    effect_size_interpretation: str  # "small", "medium", "large"
    
    # Interpretation
    winner: Optional[str]  # Strategy with better performance (if significant)
    interpretation: str  # Human-readable summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_a': self.strategy_a,
            'strategy_b': self.strategy_b,
            'test_type': self.test_type.value,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'alpha': self.alpha,
            'effect_size': self.effect_size,
            'effect_size_interpretation': self.effect_size_interpretation,
            'winner': self.winner,
            'interpretation': self.interpretation
        }


def calculate_confidence_interval(
    data: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset.
    
    Args:
        data: List of values
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        (lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0)
    
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of mean
    
    # t-distribution for small samples
    margin = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    
    return (mean - margin, mean + margin)


def calculate_proportion_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a proportion (success rate).
    
    Uses Wilson score interval (better for small samples than normal approximation).
    
    Args:
        successes: Number of successes
        total: Total trials
        confidence_level: Confidence level (e.g., 0.95)
        
    Returns:
        (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Measures standardized difference between two groups.
    
    Args:
        group_a: Data from group A
        group_b: Data from group B
        
    Returns:
        Cohen's d (standardized difference)
    """
    if not group_a or not group_b:
        return 0.0
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    var_a = np.var(group_a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(group_b, ddof=1) if n_b > 1 else 0.0
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean_a - mean_b) / pooled_std


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def chi_square_test(
    strategy_a_successes: int,
    strategy_a_failures: int,
    strategy_b_successes: int,
    strategy_b_failures: int,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Perform chi-square test for categorical outcomes.
    
    Tests if success rates differ significantly between strategies.
    
    Args:
        strategy_a_successes: Successes for strategy A
        strategy_a_failures: Failures for strategy A
        strategy_b_successes: Successes for strategy B
        strategy_b_failures: Failures for strategy B
        alpha: Significance level
        
    Returns:
        ComparisonResult
    """
    # Contingency table
    observed = np.array([
        [strategy_a_successes, strategy_a_failures],
        [strategy_b_successes, strategy_b_failures]
    ])
    
    # Chi-square test - cast to float to satisfy type checker
    result = stats.chi2_contingency(observed)
    chi2 = cast(float, result[0])
    p_value = cast(float, result[1])
    
    is_significant = p_value < alpha
    
    # Calculate success rates
    total_a = strategy_a_successes + strategy_a_failures
    total_b = strategy_b_successes + strategy_b_failures
    
    rate_a = strategy_a_successes / total_a if total_a > 0 else 0.0
    rate_b = strategy_b_successes / total_b if total_b > 0 else 0.0
    
    # Effect size (Cramér's V for 2x2 table simplifies to phi coefficient)
    n = total_a + total_b
    effect_size = np.sqrt(chi2 / n) if n > 0 else 0.0
    
    # Determine winner
    winner = None
    if is_significant:
        winner = "Strategy A" if rate_a > rate_b else "Strategy B"
    
    # Interpretation
    if is_significant:
        interpretation = (
            f"{winner} has significantly better success rate "
            f"(A: {rate_a:.2%}, B: {rate_b:.2%}, p={p_value:.4f})"
        )
    else:
        interpretation = (
            f"No significant difference in success rates "
            f"(A: {rate_a:.2%}, B: {rate_b:.2%}, p={p_value:.4f})"
        )
    
    return ComparisonResult(
        strategy_a="Strategy A",
        strategy_b="Strategy B",
        test_type=SignificanceTest.CHI_SQUARE,
        test_statistic=chi2,
        p_value=p_value,
        is_significant=is_significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=interpret_effect_size(effect_size),
        winner=winner,
        interpretation=interpretation
    )


def t_test(
    strategy_a_scores: List[float],
    strategy_b_scores: List[float],
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Perform independent samples t-test for continuous metrics.
    
    Tests if outcome scores differ significantly between strategies.
    Assumes normally distributed data.
    
    Args:
        strategy_a_scores: Scores for strategy A
        strategy_b_scores: Scores for strategy B
        alpha: Significance level
        
    Returns:
        ComparisonResult
    """
    if not strategy_a_scores or not strategy_b_scores:
        raise ValueError("Both strategies must have scores")
    
    # Independent samples t-test - cast to float to satisfy type checker
    result = stats.ttest_ind(strategy_a_scores, strategy_b_scores)
    t_stat = cast(float, result[0])
    p_value = cast(float, result[1])
    
    is_significant = p_value < alpha
    
    # Calculate means
    mean_a = np.mean(strategy_a_scores)
    mean_b = np.mean(strategy_b_scores)
    
    # Effect size (Cohen's d)
    effect_size = cohens_d(strategy_a_scores, strategy_b_scores)
    
    # Determine winner
    winner = None
    if is_significant:
        winner = "Strategy A" if mean_a > mean_b else "Strategy B"
    
    # Interpretation
    if is_significant:
        interpretation = (
            f"{winner} has significantly better scores "
            f"(A: {mean_a:.3f}, B: {mean_b:.3f}, p={p_value:.4f}, d={effect_size:.2f})"
        )
    else:
        interpretation = (
            f"No significant difference in scores "
            f"(A: {mean_a:.3f}, B: {mean_b:.3f}, p={p_value:.4f}, d={effect_size:.2f})"
        )
    
    return ComparisonResult(
        strategy_a="Strategy A",
        strategy_b="Strategy B",
        test_type=SignificanceTest.T_TEST,
        test_statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=interpret_effect_size(effect_size),
        winner=winner,
        interpretation=interpretation
    )


def mann_whitney_test(
    strategy_a_scores: List[float],
    strategy_b_scores: List[float],
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Perform Mann-Whitney U test for continuous metrics.
    
    Non-parametric alternative to t-test. Does not assume normal distribution.
    
    Args:
        strategy_a_scores: Scores for strategy A
        strategy_b_scores: Scores for strategy B
        alpha: Significance level
        
    Returns:
        ComparisonResult
    """
    if not strategy_a_scores or not strategy_b_scores:
        raise ValueError("Both strategies must have scores")
    
    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(
        strategy_a_scores,
        strategy_b_scores,
        alternative='two-sided'
    )
    
    is_significant = p_value < alpha
    
    # Calculate medians
    median_a = np.median(strategy_a_scores)
    median_b = np.median(strategy_b_scores)
    
    # Effect size (rank-biserial correlation)
    n_a = len(strategy_a_scores)
    n_b = len(strategy_b_scores)
    effect_size = 1 - (2 * u_stat) / (n_a * n_b)
    
    # Determine winner
    winner = None
    if is_significant:
        winner = "Strategy A" if median_a > median_b else "Strategy B"
    
    # Interpretation
    if is_significant:
        interpretation = (
            f"{winner} has significantly better scores "
            f"(A median: {median_a:.3f}, B median: {median_b:.3f}, p={p_value:.4f})"
        )
    else:
        interpretation = (
            f"No significant difference in scores "
            f"(A median: {median_a:.3f}, B median: {median_b:.3f}, p={p_value:.4f})"
        )
    
    return ComparisonResult(
        strategy_a="Strategy A",
        strategy_b="Strategy B",
        test_type=SignificanceTest.MANN_WHITNEY,
        test_statistic=u_stat,
        p_value=p_value,
        is_significant=is_significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=interpret_effect_size(effect_size),
        winner=winner,
        interpretation=interpretation
    )


def recommend_strategy(
    performances: Dict[str, StrategyPerformance],
    alpha: float = 0.05,
    min_sample_size: int = 30
) -> Dict[str, Any]:
    """
    Recommend the best strategy based on statistical analysis.
    
    Args:
        performances: Performance metrics for each strategy
        alpha: Significance level for testing
        min_sample_size: Minimum samples per strategy
        
    Returns:
        Dictionary with recommendation and analysis
    """
    if not performances:
        return {
            'recommended_strategy': None,
            'confidence': 'none',
            'reason': 'No performance data available'
        }
    
    # Check sample sizes
    insufficient_data = [
        name for name, perf in performances.items()
        if perf.total_assignments < min_sample_size
    ]
    
    if insufficient_data:
        # Insufficient data - return current leader
        best_strategy = max(
            performances.items(),
            key=lambda x: x[1].success_rate
        )
        
        return {
            'recommended_strategy': best_strategy[0],
            'confidence': 'low',
            'reason': (
                f'Insufficient data for statistical analysis. '
                f'Strategies with <{min_sample_size} samples: {", ".join(insufficient_data)}. '
                f'Recommending current leader: {best_strategy[0]} '
                f'({best_strategy[1].success_rate:.2%} success rate).'
            ),
            'strategy_performances': {
                name: perf.to_dict() for name, perf in performances.items()
            }
        }
    
    # Sufficient data - perform pairwise comparisons
    strategies = list(performances.keys())
    
    if len(strategies) == 2:
        # Compare two strategies
        perf_a = performances[strategies[0]]
        perf_b = performances[strategies[1]]
        
        # Chi-square test for success rates
        chi_result = chi_square_test(
            perf_a.success_count, perf_a.failure_count,
            perf_b.success_count, perf_b.failure_count,
            alpha=alpha
        )
        chi_result.strategy_a = strategies[0]
        chi_result.strategy_b = strategies[1]
        
        # t-test for outcome scores (if available)
        t_result = None
        if perf_a.outcome_scores and perf_b.outcome_scores:
            t_result = t_test(perf_a.outcome_scores, perf_b.outcome_scores, alpha=alpha)
            t_result.strategy_a = strategies[0]
            t_result.strategy_b = strategies[1]
        
        # Determine recommendation
        if chi_result.is_significant and chi_result.winner:
            # Parse winner string to get actual strategy name
            if "Strategy A" in chi_result.winner:
                recommended = strategies[0]
            else:
                recommended = strategies[1]
            confidence = 'high' if chi_result.effect_size > 0.5 else 'medium'
            reason = chi_result.interpretation
        else:
            # No significant difference - recommend higher success rate
            recommended = strategies[0] if perf_a.success_rate >= perf_b.success_rate else strategies[1]
            confidence = 'low'
            reason = 'No statistically significant difference. Recommending strategy with higher success rate.'
        
        return {
            'recommended_strategy': recommended,
            'confidence': confidence,
            'reason': reason,
            'chi_square_test': chi_result.to_dict(),
            't_test': t_result.to_dict() if t_result else None,
            'strategy_performances': {
                name: perf.to_dict() for name, perf in performances.items()
            }
        }
    
    else:
        # Multiple strategies - find best performer
        # For simplicity, use success rate + confidence intervals
        best_strategy = max(
            performances.items(),
            key=lambda x: x[1].success_rate
        )
        
        # Check if best is significantly better than others
        significant_wins = 0
        comparisons = []
        
        for strategy_name, perf in performances.items():
            if strategy_name == best_strategy[0]:
                continue
            
            chi_result = chi_square_test(
                best_strategy[1].success_count, best_strategy[1].failure_count,
                perf.success_count, perf.failure_count,
                alpha=alpha
            )
            chi_result.strategy_a = best_strategy[0]
            chi_result.strategy_b = strategy_name
            
            comparisons.append(chi_result.to_dict())
            
            if chi_result.is_significant and chi_result.winner and chi_result.winner.endswith("A"):
                significant_wins += 1
        
        # Determine confidence
        if significant_wins == len(strategies) - 1:
            confidence = 'high'
            reason = f'{best_strategy[0]} significantly outperforms all other strategies.'
        elif significant_wins > 0:
            confidence = 'medium'
            reason = f'{best_strategy[0]} significantly outperforms {significant_wins}/{len(strategies)-1} strategies.'
        else:
            confidence = 'low'
            reason = f'{best_strategy[0]} has highest success rate but not significantly better than others.'
        
        return {
            'recommended_strategy': best_strategy[0],
            'confidence': confidence,
            'reason': reason,
            'comparisons': comparisons,
            'strategy_performances': {
                name: perf.to_dict() for name, perf in performances.items()
            }
        }
