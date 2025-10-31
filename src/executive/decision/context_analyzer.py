"""
Context Analyzer - Dynamic weight adjustment based on decision context

Adjusts criterion weights based on cognitive and environmental factors:
- High cognitive load → favor simpler criteria
- Time pressure → favor quick-to-evaluate criteria
- Risk tolerance → adjust uncertainty penalties
- Past outcomes → learn from experience
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import time

from .base import EnhancedDecisionContext

# Initialize logger
logger = logging.getLogger(__name__)

# Metrics tracking (lazy import to avoid circular dependency)
_metrics_registry = None

def get_metrics_registry():
    """Lazy import of metrics registry from chat system"""
    global _metrics_registry
    if _metrics_registry is None:
        try:
            from src.chat.metrics import metrics_registry
            _metrics_registry = metrics_registry
        except ImportError:
            # Fallback to dummy metrics if chat system unavailable
            class DummyMetrics:
                def inc(self, name, value=1): pass
                def observe(self, name, ms): pass
                def observe_hist(self, name, value, max_len=500): pass
            _metrics_registry = DummyMetrics()
    return _metrics_registry


@dataclass
class ContextAdjustment:
    """
    Record of context-based adjustment
    
    Attributes:
        criterion: Criterion being adjusted
        original_weight: Original weight
        adjusted_weight: Weight after context adjustment
        adjustment_factor: Multiplier applied
        reason: Why adjustment was made
    """
    criterion: str
    original_weight: float
    adjusted_weight: float
    adjustment_factor: float
    reason: str


class ContextAnalyzer:
    """
    Analyzes decision context and adjusts criterion weights accordingly
    
    Factors considered:
    - Cognitive load: Under high load, reduce weight of complex criteria
    - Time pressure: Under time pressure, favor quick decisions
    - Risk tolerance: Adjust uncertainty-related criteria
    - User preferences: Apply user-specific adjustments
    """
    
    def __init__(self):
        """Initialize context analyzer"""
        self.adjustment_history: List[ContextAdjustment] = []
    
    def adjust_weights(
        self,
        weights: Dict[str, float],
        context: EnhancedDecisionContext,
        criteria_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> tuple[Dict[str, float], List[ContextAdjustment]]:
        """
        Adjust criterion weights based on context
        
        Args:
            weights: Original criterion weights
            context: Decision context with cognitive/environmental factors
            criteria_metadata: Optional metadata about criteria (complexity, etc.)
            
        Returns:
            Tuple of (adjusted weights, list of adjustments made)
        """
        start_time = time.time()
        metrics = get_metrics_registry()
        
        try:
            adjusted_weights = weights.copy()
            adjustments = []
            
            if criteria_metadata is None:
                criteria_metadata = {}
            
            # Track context values
            metrics.observe_hist('context_cognitive_load', context.cognitive_load)
            metrics.observe_hist('context_time_pressure', context.time_pressure)
            metrics.observe_hist('context_risk_tolerance', context.risk_tolerance)
            
            # Adjustment 1: Cognitive load
            if context.cognitive_load > 0.7:
                load_adjustments = self._adjust_for_cognitive_load(
                    adjusted_weights,
                    context.cognitive_load,
                    criteria_metadata
                )
                adjustments.extend(load_adjustments)
                if load_adjustments:
                    metrics.inc('context_cognitive_load_adjustments_total', len(load_adjustments))
            
            # Adjustment 2: Time pressure
            if context.time_pressure > 0.7:
                time_adjustments = self._adjust_for_time_pressure(
                    adjusted_weights,
                    context.time_pressure,
                    criteria_metadata
                )
                adjustments.extend(time_adjustments)
                if time_adjustments:
                    metrics.inc('context_time_pressure_adjustments_total', len(time_adjustments))
            
            # Adjustment 3: Risk tolerance
            risk_adjustments = self._adjust_for_risk_tolerance(
                adjusted_weights,
                context.risk_tolerance,
                criteria_metadata
            )
            adjustments.extend(risk_adjustments)
            if risk_adjustments:
                metrics.inc('context_risk_tolerance_adjustments_total', len(risk_adjustments))
            
            # Adjustment 4: User preferences
            if context.user_preferences:
                pref_adjustments = self._adjust_for_preferences(
                    adjusted_weights,
                    context.user_preferences
                )
                adjustments.extend(pref_adjustments)
                if pref_adjustments:
                    metrics.inc('context_preference_adjustments_total', len(pref_adjustments))
            
            # Normalize weights to sum to 1.0
            total = sum(adjusted_weights.values())
            if total > 0:
                adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
            
            # Track total adjustments
            metrics.inc('context_adjustments_total', len(adjustments))
            metrics.observe_hist('context_adjustment_count', len(adjustments))
            
            # Track performance
            duration = (time.time() - start_time) * 1000.0
            metrics.observe('context_analysis_latency_ms', duration)
            
            # Track weight changes
            if adjustments:
                max_change = max(abs(adj.adjusted_weight - adj.original_weight) for adj in adjustments)
                metrics.observe_hist('context_max_weight_change', max_change)
            
            self.adjustment_history.extend(adjustments)
            
            logger.debug(
                f"Context analysis complete: {len(adjustments)} adjustments, "
                f"cognitive_load={context.cognitive_load:.2f}, "
                f"time_pressure={context.time_pressure:.2f}, "
                f"latency={duration:.1f}ms"
            )
            
            return adjusted_weights, adjustments
            
        except Exception as e:
            metrics.inc('context_analysis_errors_total')
            logger.error(f"Context analysis failed: {e}", exc_info=True)
            raise
    
    def _adjust_for_cognitive_load(
        self,
        weights: Dict[str, float],
        cognitive_load: float,
        criteria_metadata: Dict[str, Dict[str, Any]]
    ) -> List[ContextAdjustment]:
        """Reduce weight of complex criteria under high cognitive load"""
        adjustments = []
        
        # Dampen complex criteria (if metadata available)
        for criterion, weight in weights.items():
            complexity = criteria_metadata.get(criterion, {}).get('complexity', 0.5)
            
            if complexity > 0.6:  # Complex criterion
                # Reduce weight proportional to cognitive load and complexity
                reduction_factor = 1.0 - (cognitive_load * complexity * 0.3)
                original_weight = weight
                weights[criterion] = weight * reduction_factor
                
                adjustments.append(ContextAdjustment(
                    criterion=criterion,
                    original_weight=original_weight,
                    adjusted_weight=weights[criterion],
                    adjustment_factor=reduction_factor,
                    reason=f"High cognitive load ({cognitive_load:.2f}), complex criterion"
                ))
        
        return adjustments
    
    def _adjust_for_time_pressure(
        self,
        weights: Dict[str, float],
        time_pressure: float,
        criteria_metadata: Dict[str, Dict[str, Any]]
    ) -> List[ContextAdjustment]:
        """Favor quick-to-evaluate criteria under time pressure"""
        adjustments = []
        
        for criterion, weight in weights.items():
            evaluation_time = criteria_metadata.get(criterion, {}).get('eval_time', 0.5)
            
            if evaluation_time > 0.6:  # Slow to evaluate
                # Reduce weight proportional to time pressure
                reduction_factor = 1.0 - (time_pressure * 0.4)
                original_weight = weight
                weights[criterion] = weight * reduction_factor
                
                adjustments.append(ContextAdjustment(
                    criterion=criterion,
                    original_weight=original_weight,
                    adjusted_weight=weights[criterion],
                    adjustment_factor=reduction_factor,
                    reason=f"High time pressure ({time_pressure:.2f}), slow criterion"
                ))
            elif evaluation_time < 0.4:  # Fast to evaluate
                # Boost weight slightly
                boost_factor = 1.0 + (time_pressure * 0.2)
                original_weight = weight
                weights[criterion] = weight * boost_factor
                
                adjustments.append(ContextAdjustment(
                    criterion=criterion,
                    original_weight=original_weight,
                    adjusted_weight=weights[criterion],
                    adjustment_factor=boost_factor,
                    reason=f"High time pressure ({time_pressure:.2f}), fast criterion"
                ))
        
        return adjustments
    
    def _adjust_for_risk_tolerance(
        self,
        weights: Dict[str, float],
        risk_tolerance: float,
        criteria_metadata: Dict[str, Dict[str, Any]]
    ) -> List[ContextAdjustment]:
        """Adjust uncertainty-related criteria based on risk tolerance"""
        adjustments = []
        
        for criterion, weight in weights.items():
            is_uncertainty_criterion = criteria_metadata.get(
                criterion, {}
            ).get('uncertainty_related', False)
            
            if is_uncertainty_criterion:
                if risk_tolerance < 0.3:
                    # Risk-averse: boost safety/certainty criteria
                    boost_factor = 1.0 + ((1.0 - risk_tolerance) * 0.3)
                    original_weight = weight
                    weights[criterion] = weight * boost_factor
                    
                    adjustments.append(ContextAdjustment(
                        criterion=criterion,
                        original_weight=original_weight,
                        adjusted_weight=weights[criterion],
                        adjustment_factor=boost_factor,
                        reason=f"Low risk tolerance ({risk_tolerance:.2f}), boosting safety"
                    ))
                elif risk_tolerance > 0.7:
                    # Risk-seeking: reduce weight of safety criteria
                    reduction_factor = 1.0 - (risk_tolerance * 0.2)
                    original_weight = weight
                    weights[criterion] = weight * reduction_factor
                    
                    adjustments.append(ContextAdjustment(
                        criterion=criterion,
                        original_weight=original_weight,
                        adjusted_weight=weights[criterion],
                        adjustment_factor=reduction_factor,
                        reason=f"High risk tolerance ({risk_tolerance:.2f}), reducing safety weight"
                    ))
        
        return adjustments
    
    def _adjust_for_preferences(
        self,
        weights: Dict[str, float],
        preferences: Dict[str, Any]
    ) -> List[ContextAdjustment]:
        """Apply user-specific preference adjustments"""
        adjustments = []
        
        # Check for explicit criterion preferences
        criterion_prefs = preferences.get('criterion_preferences', {})
        
        for criterion, weight in weights.items():
            if criterion in criterion_prefs:
                pref_multiplier = criterion_prefs[criterion]
                original_weight = weight
                weights[criterion] = weight * pref_multiplier
                
                adjustments.append(ContextAdjustment(
                    criterion=criterion,
                    original_weight=original_weight,
                    adjusted_weight=weights[criterion],
                    adjustment_factor=pref_multiplier,
                    reason=f"User preference multiplier: {pref_multiplier:.2f}"
                ))
        
        return adjustments
    
    def explain_adjustments(
        self,
        adjustments: List[ContextAdjustment]
    ) -> str:
        """
        Generate human-readable explanation of adjustments
        
        Args:
            adjustments: List of adjustments made
            
        Returns:
            Natural language explanation
        """
        if not adjustments:
            return "No context-based adjustments were made."
        
        explanation = [f"Made {len(adjustments)} context-based adjustments:"]
        
        for adj in adjustments:
            change_pct = ((adj.adjusted_weight - adj.original_weight) 
                         / adj.original_weight * 100)
            direction = "increased" if change_pct > 0 else "decreased"
            
            explanation.append(
                f"  • {adj.criterion}: {direction} by {abs(change_pct):.1f}% - {adj.reason}"
            )
        
        return "\n".join(explanation)
