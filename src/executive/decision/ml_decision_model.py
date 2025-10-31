"""
Machine Learning Decision Model - Learn from decision outcomes

Implements a simple decision tree that learns from past decisions:
- Tracks decision → outcome pairs
- Trains on successful decisions
- Suggests weight adjustments
- Provides confidence estimates
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pickle
import logging
import time

from .base import DecisionOutcome, EnhancedDecisionContext

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
class DecisionFeatures:
    """
    Features extracted from decision context for ML
    
    Attributes:
        cognitive_load: Current cognitive load
        time_pressure: Decision urgency
        risk_tolerance: Risk acceptance level
        num_options: Number of options considered
        num_criteria: Number of criteria used
        context_hash: Hash of broader context
    """
    cognitive_load: float
    time_pressure: float
    risk_tolerance: float
    num_options: int
    num_criteria: int
    context_hash: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for sklearn"""
        return np.array([
            self.cognitive_load,
            self.time_pressure,
            self.risk_tolerance,
            float(self.num_options),
            float(self.num_criteria),
            float(self.context_hash % 10000) / 10000.0,  # Normalize hash
        ])


class MLDecisionModel:
    """
    Machine learning model that learns from decision outcomes
    
    Uses decision trees to predict decision success and suggest
    criterion weight adjustments based on historical data.
    """
    
    def __init__(self, min_samples_to_train: int = 10):
        """
        Initialize ML decision model
        
        Args:
            min_samples_to_train: Minimum outcomes before training
        """
        self.min_samples_to_train = min_samples_to_train
        self.outcomes: List[DecisionOutcome] = []
        self.model: Optional[DecisionTreeClassifier] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances: Dict[str, float] = {}
    
    def record_outcome(self, outcome: DecisionOutcome) -> None:
        """
        Record a decision outcome for learning
        
        Args:
            outcome: Decision outcome to record
        """
        metrics = get_metrics_registry()
        
        self.outcomes.append(outcome)
        metrics.inc('ml_outcomes_recorded_total')
        metrics.observe_hist('ml_outcomes_count', len(self.outcomes))
        
        # Track success rate
        if outcome.success:
            metrics.inc('ml_successful_outcomes_total')
        else:
            metrics.inc('ml_failed_outcomes_total')
        
        logger.debug(f"Recorded ML outcome: success={outcome.success}, total_outcomes={len(self.outcomes)}")
        
        # Retrain if we have enough samples
        if len(self.outcomes) >= self.min_samples_to_train:
            if len(self.outcomes) % 5 == 0:  # Retrain every 5 outcomes
                self.train()
    
    def train(self) -> bool:
        """
        Train model on recorded outcomes
        
        Returns:
            True if training was successful
        """
        start_time = time.time()
        metrics = get_metrics_registry()
        
        if len(self.outcomes) < self.min_samples_to_train:
            logger.debug(f"Insufficient samples for training: {len(self.outcomes)} < {self.min_samples_to_train}")
            return False
        
        try:
            # Extract features and labels
            X = []
            y = []
            
            for outcome in self.outcomes:
                if outcome.context_at_decision:
                    features = self._extract_features(
                        outcome.context_at_decision,
                        len(outcome.outcome_metrics),  # Proxy for num options
                        len(outcome.outcome_metrics)   # Proxy for num criteria
                    )
                    X.append(features.to_array())
                    y.append(1 if outcome.success else 0)
            
            if len(X) < self.min_samples_to_train:
                logger.debug(f"Insufficient valid samples: {len(X)} < {self.min_samples_to_train}")
                return False
            
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_array)
            
            # Train decision tree
            self.model = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X_scaled, y_array)
            
            # Store feature importances
            feature_names = [
                'cognitive_load',
                'time_pressure',
                'risk_tolerance',
                'num_options',
                'num_criteria',
                'context_hash'
            ]
            self.feature_importances = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
            
            # Track accuracy
            accuracy = self.model.score(X_scaled, y_array)
            metrics.observe_hist('ml_training_accuracy', accuracy)
            metrics.inc('ml_training_sessions_total')
            
            # Track training time
            duration = (time.time() - start_time) * 1000.0
            metrics.observe('ml_training_latency_ms', duration)
            
            self.is_trained = True
            
            logger.info(
                f"ML model trained: {len(X)} samples, "
                f"accuracy={accuracy:.2%}, "
                f"latency={duration:.1f}ms"
            )
            
            return True
            
        except Exception as e:
            metrics.inc('ml_training_errors_total')
            logger.error(f"ML training failed: {e}", exc_info=True)
            return False
    
    def predict_success(
        self,
        context: EnhancedDecisionContext,
        num_options: int,
        num_criteria: int
    ) -> Tuple[float, float]:
        """
        Predict probability of decision success
        
        Args:
            context: Decision context
            num_options: Number of options being considered
            num_criteria: Number of criteria being used
            
        Returns:
            Tuple of (success probability, confidence)
        """
        if not self.is_trained or self.model is None:
            return 0.5, 0.0  # No prediction, no confidence
        
        features = self._extract_features(context, num_options, num_criteria)
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probability
        proba = self.model.predict_proba(X_scaled)[0]
        success_prob = proba[1] if len(proba) > 1 else 0.5
        
        # Calculate confidence (based on tree depth and sample size)
        confidence = min(1.0, len(self.outcomes) / (self.min_samples_to_train * 5))
        
        return success_prob, confidence
    
    def suggest_weight_adjustments(
        self,
        context: EnhancedDecisionContext,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Suggest weight adjustments based on learned patterns
        
        Args:
            context: Current decision context
            current_weights: Current criterion weights
            
        Returns:
            Suggested weight adjustments (multipliers)
        """
        if not self.is_trained:
            return {k: 1.0 for k in current_weights}  # No adjustments
        
        # Analyze which features are most important for success
        adjustments = {}
        
        # Simple heuristic: if cognitive_load is important and current load is high,
        # suggest reducing complex criteria weights
        if self.feature_importances.get('cognitive_load', 0) > 0.15:
            if context.cognitive_load > 0.7:
                # Suggest reducing all weights slightly (will be normalized)
                adjustments = {k: 0.9 for k in current_weights}
        
        # If time_pressure is important and pressure is high,
        # suggest focusing weights (increase top, decrease bottom)
        if self.feature_importances.get('time_pressure', 0) > 0.15:
            if context.time_pressure > 0.7:
                sorted_weights = sorted(
                    current_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                # Boost top 3, reduce rest
                for i, (criterion, _) in enumerate(sorted_weights):
                    if i < 3:
                        adjustments[criterion] = 1.2
                    else:
                        adjustments[criterion] = 0.8
        
        # Default: no adjustment
        for criterion in current_weights:
            if criterion not in adjustments:
                adjustments[criterion] = 1.0
        
        return adjustments
    
    def get_insights(self) -> Dict[str, Any]:
        """
        Get insights from learned model
        
        Returns:
            Dict with model insights
        """
        if not self.is_trained:
            return {
                'is_trained': False,
                'num_outcomes': len(self.outcomes),
                'min_samples_needed': self.min_samples_to_train
            }
        
        # Calculate success rate
        successes = sum(1 for o in self.outcomes if o.success)
        success_rate = successes / len(self.outcomes) if self.outcomes else 0.0
        
        return {
            'is_trained': True,
            'num_outcomes': len(self.outcomes),
            'success_rate': success_rate,
            'feature_importances': self.feature_importances,
            'most_important_feature': max(
                self.feature_importances.items(),
                key=lambda x: x[1]
            )[0] if self.feature_importances else None,
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'outcomes': self.outcomes,
                    'is_trained': self.is_trained,
                    'feature_importances': self.feature_importances,
                }, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.outcomes = data['outcomes']
                self.is_trained = data['is_trained']
                self.feature_importances = data['feature_importances']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _extract_features(
        self,
        context: EnhancedDecisionContext,
        num_options: int,
        num_criteria: int
    ) -> DecisionFeatures:
        """Extract features from context"""
        # Simple hash of domain knowledge
        context_hash = hash(str(context.domain_knowledge)) if context.domain_knowledge else 0
        
        return DecisionFeatures(
            cognitive_load=context.cognitive_load,
            time_pressure=context.time_pressure,
            risk_tolerance=context.risk_tolerance,
            num_options=num_options,
            num_criteria=num_criteria,
            context_hash=context_hash
        )
