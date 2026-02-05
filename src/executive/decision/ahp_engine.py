"""
Analytic Hierarchy Process (AHP) Decision Engine

Implements the AHP algorithm for multi-criteria decision-making with:
- Pairwise comparison matrices
- Eigenvector method for weight calculation
- Consistency ratio validation
- Hierarchical criteria support

Reference: Saaty, T.L. (1980). The Analytic Hierarchy Process.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.linalg import eig
from dataclasses import dataclass
import logging
import time

from .base import (
    CriteriaHierarchy,
    EnhancedDecisionContext,
    EnhancedDecisionResult,
    DecisionStrategy,
)

# Initialize logger
logger = logging.getLogger(__name__)

# Metrics tracking (lazy import to avoid circular dependency)
_metrics_registry = None

def get_metrics_registry():
    """Lazy import of metrics registry from chat system"""
    global _metrics_registry
    if _metrics_registry is None:
        try:
            from src.memory.metrics import metrics_registry
            _metrics_registry = metrics_registry
        except ImportError:
            # Fallback to dummy metrics if chat system unavailable
            class DummyMetrics:
                def inc(self, name, value=1): pass
                def observe(self, name, ms): pass
                def observe_hist(self, name, value, max_len=500): pass
            _metrics_registry = DummyMetrics()
    return _metrics_registry


# AHP fundamental scale for pairwise comparisons
# 1 = Equal importance
# 3 = Moderate importance
# 5 = Strong importance
# 7 = Very strong importance
# 9 = Extreme importance
# 2, 4, 6, 8 = Intermediate values

# Random Index (RI) for consistency checking
# RI values for matrices of size n (from Saaty's research)
RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}


@dataclass
class AHPResult:
    """
    Result of AHP analysis
    
    Attributes:
        weights: Computed weights for criteria
        consistency_ratio: CR value (should be < 0.1)
        lambda_max: Maximum eigenvalue
        is_consistent: Whether the comparisons are consistent
        hierarchy: Full criteria hierarchy with weights
    """
    weights: Dict[str, float]
    consistency_ratio: float
    lambda_max: float
    is_consistent: bool
    hierarchy: CriteriaHierarchy


class AHPEngine:
    """
    Analytic Hierarchy Process implementation for multi-criteria decisions
    
    Usage:
        1. Create criteria hierarchy
        2. Provide pairwise comparisons
        3. Calculate weights using eigenvector method
        4. Validate consistency
        5. Apply weights to options
    """
    
    def __init__(self, consistency_threshold: float = 0.1):
        """
        Initialize AHP engine
        
        Args:
            consistency_threshold: Maximum acceptable CR (default 0.1)
        """
        self.consistency_threshold = consistency_threshold
    
    def build_pairwise_matrix(
        self,
        criteria: List[str],
        comparisons: Dict[Tuple[str, str], float]
    ) -> np.ndarray:
        """
        Build pairwise comparison matrix from comparisons
        
        Args:
            criteria: List of criterion names
            comparisons: Dict mapping (criterion_i, criterion_j) to comparison value
                        Value > 1 means i is more important than j
                        Value < 1 means j is more important than i
                        
        Returns:
            Square matrix M where M[i,j] is comparison of i to j
        """
        n = len(criteria)
        matrix = np.ones((n, n))
        
        for i, crit_i in enumerate(criteria):
            for j, crit_j in enumerate(criteria):
                if i == j:
                    matrix[i, j] = 1.0
                elif (crit_i, crit_j) in comparisons:
                    matrix[i, j] = comparisons[(crit_i, crit_j)]
                elif (crit_j, crit_i) in comparisons:
                    # Reciprocal relationship
                    matrix[i, j] = 1.0 / comparisons[(crit_j, crit_i)]
                # else: matrix[i, j] remains 1.0 (equal importance)
        
        return matrix
    
    def calculate_weights(self, matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate priority weights using eigenvector method
        
        Args:
            matrix: Pairwise comparison matrix
            
        Returns:
            Tuple of (normalized weights, lambda_max)
        """
        # Calculate eigenvalues and eigenvectors
        result = eig(matrix)
        eigenvalues = np.real(result[0])
        eigenvectors = np.real(result[1])
        
        # Find the principal eigenvector (corresponding to max eigenvalue)
        max_eigenvalue_idx = np.argmax(eigenvalues)
        lambda_max = eigenvalues[max_eigenvalue_idx]
        principal_eigenvector = eigenvectors[:, max_eigenvalue_idx]
        
        # Normalize to get weights (sum to 1.0)
        weights = principal_eigenvector / np.sum(principal_eigenvector)
        
        return weights, lambda_max
    
    def calculate_consistency_ratio(
        self,
        lambda_max: float,
        n: int
    ) -> float:
        """
        Calculate consistency ratio (CR)
        
        CR = CI / RI
        where CI = (lambda_max - n) / (n - 1)
        
        Args:
            lambda_max: Maximum eigenvalue
            n: Size of the matrix
            
        Returns:
            Consistency ratio (should be < 0.1 for acceptable consistency)
        """
        if n <= 2:
            return 0.0  # Perfect consistency for n <= 2
        
        # Consistency Index
        ci = (lambda_max - n) / (n - 1)
        
        # Random Index
        ri = RANDOM_INDEX.get(n, 1.59)  # Use 1.59 for n > 15
        
        # Consistency Ratio
        cr = ci / ri if ri > 0 else 0.0
        
        return cr
    
    def analyze_hierarchy(
        self,
        hierarchy: CriteriaHierarchy
    ) -> AHPResult:
        """
        Analyze criteria hierarchy and compute all weights
        
        Recursively computes weights for hierarchical criteria.
        
        Args:
            hierarchy: Criteria hierarchy with pairwise comparisons
            
        Returns:
            AHP result with weights and consistency metrics
        """
        # Get all criteria at this level
        if hierarchy.sub_criteria:
            criteria_names = [c.name for c in hierarchy.sub_criteria]
            
            # Build pairwise comparison matrix
            matrix = self.build_pairwise_matrix(
                criteria_names,
                hierarchy.pairwise_comparisons
            )
            
            # Calculate weights
            weights, lambda_max = self.calculate_weights(matrix)
            
            # Calculate consistency ratio
            cr = self.calculate_consistency_ratio(lambda_max, len(criteria_names))
            is_consistent = cr <= self.consistency_threshold
            
            # Assign weights to sub-criteria
            for i, sub_criterion in enumerate(hierarchy.sub_criteria):
                sub_criterion.weight = weights[i]
                
                # Recursively analyze sub-hierarchies
                if sub_criterion.sub_criteria:
                    sub_result = self.analyze_hierarchy(sub_criterion)
                    # Inherit worst consistency
                    if not sub_result.is_consistent:
                        is_consistent = False
                        cr = max(cr, sub_result.consistency_ratio)
            
            # Collect all leaf weights
            leaf_weights = {}
            for leaf in hierarchy.get_all_leaves():
                leaf_weights[leaf.name] = leaf.global_weight()
            
            hierarchy.consistency_ratio = cr
            
            return AHPResult(
                weights=leaf_weights,
                consistency_ratio=cr,
                lambda_max=lambda_max,
                is_consistent=is_consistent,
                hierarchy=hierarchy
            )
        else:
            # Leaf criterion, no analysis needed
            return AHPResult(
                weights={hierarchy.name: hierarchy.weight},
                consistency_ratio=0.0,
                lambda_max=1.0,
                is_consistent=True,
                hierarchy=hierarchy
            )
    
    def score_options(
        self,
        options: List[Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Score options using weighted criteria
        
        Args:
            options: List of options, each a dict of criterion -> score
            weights: Criterion weights from AHP
            
        Returns:
            Dict mapping option index to total score
        """
        scores = {}
        
        for idx, option in enumerate(options):
            total_score = 0.0
            for criterion, weight in weights.items():
                if criterion in option:
                    total_score += option[criterion] * weight
            scores[f"option_{idx}"] = total_score
        
        return scores


class AHPStrategy(DecisionStrategy):
    """
    Decision strategy using Analytic Hierarchy Process
    
    Applies AHP to compute criterion weights, then scores options.
    """
    
    def __init__(self, consistency_threshold: float = 0.1):
        """
        Initialize AHP strategy
        
        Args:
            consistency_threshold: Max acceptable consistency ratio
        """
        self.ahp_engine = AHPEngine(consistency_threshold)
        self.last_ahp_result: Optional[AHPResult] = None
    
    def decide(
        self,
        options: List[Dict[str, float]],
        criteria: CriteriaHierarchy,
        context: EnhancedDecisionContext
    ) -> EnhancedDecisionResult:
        """
        Make decision using AHP
        
        Args:
            options: List of options with criterion scores
            criteria: Hierarchical criteria with pairwise comparisons
            context: Decision context
            
        Returns:
            Enhanced decision result
        """
        start_time = time.time()
        metrics = get_metrics_registry()
        
        try:
            # Analyze hierarchy to get weights
            ahp_start = time.time()
            ahp_result = self.ahp_engine.analyze_hierarchy(criteria)
            ahp_duration = (time.time() - ahp_start) * 1000.0
            
            self.last_ahp_result = ahp_result
            
            # Track consistency metrics
            metrics.observe_hist('ahp_consistency_ratio', ahp_result.consistency_ratio)
            metrics.inc('ahp_decisions_total')
            if ahp_result.is_consistent:
                metrics.inc('ahp_consistent_decisions_total')
            else:
                metrics.inc('ahp_inconsistent_decisions_total')
                logger.warning(f"AHP inconsistent: CR={ahp_result.consistency_ratio:.3f} > 0.1")
            
            # Score all options
            scoring_start = time.time()
            option_scores = self.ahp_engine.score_options(options, ahp_result.weights)
            scoring_duration = (time.time() - scoring_start) * 1000.0
            
            # Find best option
            best_option_id = max(option_scores, key=lambda x: option_scores[x])
            
            # Calculate confidence based on consistency and score spread
            confidence = self._calculate_confidence(ahp_result, option_scores)
            metrics.observe_hist('ahp_confidence', confidence)
            
            # Generate rationale
            rationale = self._generate_rationale(ahp_result, option_scores, best_option_id)
            
            # Track performance metrics
            total_duration = (time.time() - start_time) * 1000.0
            metrics.observe('ahp_decision_latency_ms', total_duration)
            metrics.observe('ahp_hierarchy_analysis_ms', ahp_duration)
            metrics.observe('ahp_scoring_ms', scoring_duration)
            
            # Track counts
            metrics.inc('ahp_alternatives_processed_total', len(options))
            metrics.observe_hist('ahp_alternatives_count', len(options))
            
            logger.info(
                f"AHP decision complete: {len(options)} alternatives, "
                f"CR={ahp_result.consistency_ratio:.3f}, "
                f"confidence={confidence:.2f}, "
                f"latency={total_duration:.1f}ms"
            )
            
            return EnhancedDecisionResult(
                recommended_option_id=best_option_id,
                option_scores=option_scores,
                criterion_weights=ahp_result.weights,
                original_weights=ahp_result.weights.copy(),
                rationale=rationale,
                confidence=confidence,
                context=context,
                metadata={
                    'ahp_consistency_ratio': ahp_result.consistency_ratio,
                    'ahp_is_consistent': ahp_result.is_consistent,
                    'ahp_lambda_max': ahp_result.lambda_max,
                    'ahp_latency_ms': total_duration,
                    'ahp_hierarchy_analysis_ms': ahp_duration,
                    'ahp_scoring_ms': scoring_duration,
                }
            )
            
        except Exception as e:
            metrics.inc('ahp_errors_total')
            logger.error(f"AHP decision failed: {e}", exc_info=True)
            raise
    
    def explain(self, result: EnhancedDecisionResult) -> str:
        """
        Generate natural language explanation
        
        Args:
            result: Decision result
            
        Returns:
            Human-readable explanation
        """
        explanation = [f"Recommended option: {result.recommended_option_id}"]
        explanation.append(f"Confidence: {result.confidence:.2%}")
        
        if result.metadata.get('ahp_is_consistent'):
            explanation.append(
                f"AHP consistency ratio: {result.metadata['ahp_consistency_ratio']:.3f} (acceptable)"
            )
        else:
            explanation.append(
                f"⚠️ AHP consistency ratio: {result.metadata['ahp_consistency_ratio']:.3f} (inconsistent comparisons)"
            )
        
        explanation.append("\nCriterion weights:")
        sorted_weights = sorted(
            result.criterion_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for criterion, weight in sorted_weights[:5]:  # Top 5
            explanation.append(f"  • {criterion}: {weight:.3f}")
        
        explanation.append(f"\n{result.rationale}")
        
        return "\n".join(explanation)
    
    def _calculate_confidence(
        self,
        ahp_result: AHPResult,
        option_scores: Dict[str, float]
    ) -> float:
        """Calculate confidence based on consistency and score spread"""
        # Base confidence on consistency
        if ahp_result.is_consistent:
            consistency_confidence = 1.0 - ahp_result.consistency_ratio
        else:
            consistency_confidence = 0.5  # Reduced confidence for inconsistent comparisons
        
        # Factor in score spread (higher spread = more confident)
        scores = list(option_scores.values())
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            mean_score = np.mean(scores)
            spread_confidence = min(1.0, score_range / mean_score) if mean_score > 0 else 0.5
        else:
            spread_confidence = 1.0
        
        # Combine factors
        confidence = 0.6 * consistency_confidence + 0.4 * spread_confidence
        
        return float(max(0.0, min(1.0, confidence)))
    
    def _generate_rationale(
        self,
        ahp_result: AHPResult,
        option_scores: Dict[str, float],
        best_option_id: str
    ) -> str:
        """Generate rationale for the decision"""
        top_criteria = sorted(
            ahp_result.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        criteria_str = ", ".join([f"{name} ({weight:.2f})" for name, weight in top_criteria])
        
        rationale = (
            f"Based on AHP analysis with {len(ahp_result.weights)} criteria, "
            f"{best_option_id} scored highest ({option_scores[best_option_id]:.3f}). "
            f"Top weighted criteria: {criteria_str}."
        )
        
        return rationale
