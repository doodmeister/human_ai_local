"""
Decision Module - Enhanced decision-making with AHP and Pareto optimization

This module provides advanced decision-making capabilities using:
- Analytic Hierarchy Process (AHP) for criteria weighting
- Pareto optimization for multi-objective decisions
- Context-aware decision strategies
- Machine learning from decision outcomes

Architecture:
- base.py: Core interfaces and data structures
- ahp_engine.py: AHP implementation
- pareto_optimizer.py: Pareto frontier calculation
- context_analyzer.py: Dynamic weight adjustment
- ml_decision_model.py: Learning from outcomes
"""

from .base import (
    EnhancedDecisionContext,
    EnhancedDecisionResult,
    DecisionOutcome,
    DecisionStrategy,
    CriteriaHierarchy,
    ParetoSolution,
    FeatureFlags,
    get_feature_flags,
)

from .ahp_engine import (
    AHPEngine,
    AHPStrategy,
    AHPResult,
    RANDOM_INDEX,
)

from .pareto_optimizer import (
    ParetoOptimizer,
    ParetoStrategy,
    ParetoFrontier,
)

from .context_analyzer import (
    ContextAnalyzer,
    ContextAdjustment,
)

from .ml_decision_model import (
    MLDecisionModel,
    DecisionFeatures,
)

__all__ = [
    # Base classes
    "EnhancedDecisionContext",
    "EnhancedDecisionResult",
    "DecisionOutcome",
    "DecisionStrategy",
    "CriteriaHierarchy",
    "ParetoSolution",
    "FeatureFlags",
    "get_feature_flags",
    # AHP
    "AHPEngine",
    "AHPStrategy",
    "AHPResult",
    "RANDOM_INDEX",
    # Pareto
    "ParetoOptimizer",
    "ParetoStrategy",
    "ParetoFrontier",
    # Context
    "ContextAnalyzer",
    "ContextAdjustment",
    # ML
    "MLDecisionModel",
    "DecisionFeatures",
]

