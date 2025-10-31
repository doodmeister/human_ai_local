"""
Base classes and data structures for enhanced decision-making

This module defines the core abstractions for advanced decision strategies,
including context-aware weighting, outcome tracking, and hierarchical criteria.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import uuid


@dataclass
class EnhancedDecisionContext:
    """
    Rich context for decision-making with cognitive and environmental factors
    
    Attributes:
        cognitive_load: Current cognitive load (0.0 = fresh, 1.0 = exhausted)
        time_pressure: Urgency of decision (0.0 = relaxed, 1.0 = critical)
        risk_tolerance: Acceptable risk level (0.0 = risk-averse, 1.0 = risk-seeking)
        available_resources: List of available resources
        constraints: Hard constraints that must be satisfied
        past_decisions: Historical decisions for learning
        user_preferences: User-specific preferences
        domain_knowledge: Domain-specific context
    """
    cognitive_load: float = 0.5
    time_pressure: float = 0.5
    risk_tolerance: float = 0.5
    available_resources: List[str] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    past_decisions: List['DecisionOutcome'] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate context parameters"""
        if not (0.0 <= self.cognitive_load <= 1.0):
            raise ValueError("cognitive_load must be between 0.0 and 1.0")
        if not (0.0 <= self.time_pressure <= 1.0):
            raise ValueError("time_pressure must be between 0.0 and 1.0")
        if not (0.0 <= self.risk_tolerance <= 1.0):
            raise ValueError("risk_tolerance must be between 0.0 and 1.0")
    
    def is_high_pressure(self) -> bool:
        """Check if this is a high-pressure decision context"""
        return self.time_pressure > 0.7 or self.cognitive_load > 0.8


@dataclass
class CriteriaHierarchy:
    """
    Hierarchical structure of decision criteria for AHP
    
    Supports multi-level criteria where high-level criteria
    can be decomposed into sub-criteria.
    
    Attributes:
        name: Criterion name
        weight: Computed weight (0.0 to 1.0)
        sub_criteria: Child criteria
        parent: Parent criterion
        pairwise_comparisons: Matrix of pairwise comparisons
        consistency_ratio: AHP consistency ratio
    """
    name: str
    weight: float = 0.0
    sub_criteria: List['CriteriaHierarchy'] = field(default_factory=list)
    parent: Optional['CriteriaHierarchy'] = None
    pairwise_comparisons: Dict[Tuple[str, str], float] = field(default_factory=dict)
    consistency_ratio: float = 0.0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf criterion (no sub-criteria)"""
        return len(self.sub_criteria) == 0
    
    def get_all_leaves(self) -> List['CriteriaHierarchy']:
        """Get all leaf criteria in this hierarchy"""
        if self.is_leaf():
            return [self]
        leaves = []
        for sub in self.sub_criteria:
            leaves.extend(sub.get_all_leaves())
        return leaves
    
    def global_weight(self) -> float:
        """Calculate global weight (product of weights up the hierarchy)"""
        if self.parent is None:
            return self.weight
        return self.weight * self.parent.global_weight()


@dataclass
class ParetoSolution:
    """
    A solution on the Pareto frontier
    
    Represents a non-dominated solution in multi-objective optimization.
    
    Attributes:
        option_id: ID of the decision option
        objectives: Objective values (higher is better)
        is_dominated: Whether this solution is dominated by another
        domination_count: Number of solutions this dominates
        distance_to_ideal: Distance to ideal point
    """
    option_id: str
    objectives: Dict[str, float]
    is_dominated: bool = False
    domination_count: int = 0
    distance_to_ideal: float = float('inf')
    
    def dominates(self, other: 'ParetoSolution') -> bool:
        """
        Check if this solution dominates another
        
        A solution dominates another if it's better or equal on all
        objectives and strictly better on at least one.
        """
        better_in_any = False
        for obj_name in self.objectives:
            if self.objectives[obj_name] < other.objectives.get(obj_name, 0):
                return False  # Worse in this objective
            if self.objectives[obj_name] > other.objectives.get(obj_name, 0):
                better_in_any = True
        return better_in_any


@dataclass
class EnhancedDecisionResult:
    """
    Result of enhanced decision-making process
    
    Extends basic decision results with:
    - Pareto frontier for multi-objective decisions
    - Sensitivity analysis
    - Confidence intervals
    - Explanation generation
    
    Attributes:
        decision_id: Unique identifier
        recommended_option_id: Top recommended option
        option_scores: Scores for all options
        criterion_weights: Final weights used (potentially adjusted by context)
        original_weights: Original weights before context adjustment
        pareto_frontier: Pareto-optimal solutions
        rationale: Detailed explanation
        confidence: Confidence level (0.0 to 1.0)
        sensitivity: Sensitivity to weight changes
        trade_offs: Trade-off analysis for top options
        timestamp: When decision was made
        context: Decision context used
        metadata: Additional metadata
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recommended_option_id: Optional[str] = None
    option_scores: Dict[str, float] = field(default_factory=dict)
    criterion_weights: Dict[str, float] = field(default_factory=dict)
    original_weights: Dict[str, float] = field(default_factory=dict)
    pareto_frontier: List[ParetoSolution] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    sensitivity: Dict[str, float] = field(default_factory=dict)
    trade_offs: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[EnhancedDecisionContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'decision_id': self.decision_id,
            'recommended_option_id': self.recommended_option_id,
            'option_scores': self.option_scores,
            'criterion_weights': self.criterion_weights,
            'pareto_frontier_size': len(self.pareto_frontier),
            'rationale': self.rationale,
            'confidence': self.confidence,
            'sensitivity': self.sensitivity,
            'trade_offs': self.trade_offs,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def get_top_n_options(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N options by score"""
        sorted_options = sorted(
            self.option_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_options[:n]


@dataclass
class DecisionOutcome:
    """
    Track outcomes of decisions for learning
    
    Links decisions to their real-world outcomes to enable
    machine learning and strategy improvement.
    
    Attributes:
        decision_id: ID of the decision
        option_chosen: Option that was chosen
        outcome_metrics: Measured outcomes (e.g., goal progress, time taken)
        success: Whether decision was successful
        feedback: Qualitative feedback
        timestamp: When outcome was recorded
        context_at_decision: Context when decision was made
    """
    decision_id: str
    option_chosen: str
    outcome_metrics: Dict[str, float]
    success: bool
    feedback: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context_at_decision: Optional[EnhancedDecisionContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert outcome to dictionary for serialization"""
        return {
            'decision_id': self.decision_id,
            'option_chosen': self.option_chosen,
            'outcome_metrics': self.outcome_metrics,
            'success': self.success,
            'feedback': self.feedback,
            'timestamp': self.timestamp.isoformat()
        }


class DecisionStrategy(ABC):
    """
    Abstract base class for decision strategies
    
    Different strategies implement different decision algorithms:
    - AHP: Analytic Hierarchy Process
    - Pareto: Multi-objective Pareto optimization
    - ML: Machine learning-based decisions
    - Hybrid: Combination of multiple strategies
    """
    
    @abstractmethod
    def decide(
        self,
        options: List[Any],  # Can be DecisionOption or any option type
        criteria: Any,  # Can be list, hierarchy, or other structure
        context: EnhancedDecisionContext
    ) -> EnhancedDecisionResult:
        """
        Make a decision using this strategy
        
        Args:
            options: List of options to choose from
            criteria: Decision criteria (structure depends on strategy)
            context: Rich decision context
            
        Returns:
            Enhanced decision result with recommendations
        """
        pass
    
    @abstractmethod
    def explain(self, result: EnhancedDecisionResult) -> str:
        """
        Generate human-readable explanation of the decision
        
        Args:
            result: Decision result to explain
            
        Returns:
            Natural language explanation
        """
        pass
    
    def learn_from_outcome(self, outcome: DecisionOutcome) -> None:
        """
        Learn from decision outcome (optional)
        
        Strategies that support learning can override this to
        update their parameters based on outcomes.
        
        Args:
            outcome: Outcome of a previous decision
        """
        pass  # Default: no learning


class FeatureFlags:
    """
    Feature flags for gradual rollout of enhanced decision features
    
    Allows enabling/disabling specific enhancements without
    requiring code changes.
    """
    
    def __init__(self):
        self.use_ahp: bool = False
        self.use_pareto: bool = False
        self.use_ml_learning: bool = False
        self.use_context_adjustment: bool = False
        self.fallback_to_legacy: bool = True
        
    def enable_all(self) -> None:
        """Enable all enhanced features"""
        self.use_ahp = True
        self.use_pareto = True
        self.use_ml_learning = True
        self.use_context_adjustment = True
        self.fallback_to_legacy = False
    
    def disable_all(self) -> None:
        """Disable all enhanced features (use legacy)"""
        self.use_ahp = False
        self.use_pareto = False
        self.use_ml_learning = False
        self.use_context_adjustment = False
        self.fallback_to_legacy = True
    
    def to_dict(self) -> Dict[str, bool]:
        """Export flags as dictionary"""
        return {
            'use_ahp': self.use_ahp,
            'use_pareto': self.use_pareto,
            'use_ml_learning': self.use_ml_learning,
            'use_context_adjustment': self.use_context_adjustment,
            'fallback_to_legacy': self.fallback_to_legacy
        }


# Global feature flags instance
_feature_flags = FeatureFlags()


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance"""
    return _feature_flags
