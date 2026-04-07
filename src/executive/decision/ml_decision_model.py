"""
Machine Learning Decision Model - Learn from decision outcomes

Implements a simple decision tree that learns from past decisions:
- Tracks decision → outcome pairs
- Trains on successful decisions
- Suggests weight adjustments
- Provides confidence estimates
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import logging
import time

from .base import DecisionOutcome, EnhancedDecisionContext, get_metrics_registry

# Initialize logger
logger = logging.getLogger(__name__)

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
        domain_knowledge_signal: Continuous signal derived from domain knowledge structure
    """
    cognitive_load: float
    time_pressure: float
    risk_tolerance: float
    num_options: int
    num_criteria: int
    domain_knowledge_signal: float = 0.0

    @property
    def context_hash(self) -> float:
        """Backward-compatible alias for older callers expecting the old feature name."""
        return self.domain_knowledge_signal
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for sklearn"""
        return np.array([
            self.cognitive_load,
            self.time_pressure,
            self.risk_tolerance,
            float(self.num_options),
            float(self.num_criteria),
            self.domain_knowledge_signal,
        ])


class MLDecisionModel:
    """
    Machine learning model that learns from decision outcomes
    
    Uses decision trees to predict decision success and suggest
    criterion weight adjustments based on historical data.
    """
    
    def __init__(self, min_samples_to_train: int = 10, max_outcomes: int = 1000):
        """
        Initialize ML decision model
        
        Args:
            min_samples_to_train: Minimum outcomes before training
        """
        self.min_samples_to_train = min_samples_to_train
        self.max_outcomes = max_outcomes
        self.outcomes: List[DecisionOutcome] = []
        self.model: Optional[DecisionTreeClassifier] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances: Dict[str, float] = {}
        self.training_accuracy: Optional[float] = None
        self.validation_accuracy: Optional[float] = None
    
    def record_outcome(self, outcome: DecisionOutcome) -> None:
        """
        Record a decision outcome for learning
        
        Args:
            outcome: Decision outcome to record
        """
        metrics = get_metrics_registry()
        
        self.outcomes.append(outcome)
        if self.max_outcomes > 0 and len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes:]
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
                    num_options, num_criteria = self._infer_decision_shape(outcome)
                    features = self._extract_features(
                        outcome.context_at_decision,
                        num_options,
                        num_criteria,
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
            self.model = self._build_classifier()
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
            
            # Track training and validation accuracy separately.
            training_accuracy = self.model.score(X_scaled, y_array)
            validation_accuracy = self._estimate_validation_accuracy(X_array, y_array)
            self.training_accuracy = float(training_accuracy)
            self.validation_accuracy = validation_accuracy

            metrics.observe_hist('ml_training_accuracy', training_accuracy)
            if validation_accuracy is not None:
                metrics.observe_hist('ml_validation_accuracy', validation_accuracy)
            metrics.inc('ml_training_sessions_total')
            
            # Track training time
            duration = (time.time() - start_time) * 1000.0
            metrics.observe('ml_training_latency_ms', duration)
            
            self.is_trained = True

            validation_accuracy_str = (
                f"{validation_accuracy:.2%}"
                if validation_accuracy is not None
                else "n/a"
            )
            
            logger.info(
                f"ML model trained: {len(X)} samples, "
                f"training_accuracy={training_accuracy:.2%}, "
                f"validation_accuracy={validation_accuracy_str}, "
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
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
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
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'outcomes': self.outcomes,
                'is_trained': self.is_trained,
                'feature_importances': self.feature_importances,
                'training_accuracy': self.training_accuracy,
                'validation_accuracy': self.validation_accuracy,
            }, filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}", exc_info=True)
            return False
    
    def load_model(self, filepath: str, trusted_source: bool = False) -> bool:
        """
        Load model from file.

        Model artifacts are serialized Python objects. Only load them from
        trusted sources by explicitly opting in with trusted_source=True.
        
        Args:
            filepath: Path to load model from
            trusted_source: Whether the artifact source is trusted
            
        Returns:
            True if successful
        """
        if not trusted_source:
            logger.error(
                f"Refusing to load model from untrusted source: {filepath}. "
                "Pass trusted_source=True only for verified local artifacts."
            )
            return False

        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.outcomes = data['outcomes']
            self.is_trained = data['is_trained']
            self.feature_importances = data['feature_importances']
            self.training_accuracy = data.get('training_accuracy')
            self.validation_accuracy = data.get('validation_accuracy')
            return True
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}", exc_info=True)
            return False

    def _estimate_validation_accuracy(
        self,
        X_array: np.ndarray,
        y_array: np.ndarray,
    ) -> Optional[float]:
        """Estimate generalization accuracy with scaler fitting isolated to each fold."""
        labels, counts = np.unique(y_array, return_counts=True)
        if len(labels) < 2:
            return None

        min_class_count = int(np.min(counts))
        n_splits = min(5, min_class_count, len(y_array))
        if n_splits < 2:
            return None

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        validation_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self._build_classifier()),
        ])
        scores = cross_val_score(validation_pipeline, X_array, y_array, cv=cv)
        return float(np.mean(scores))

    def _build_classifier(self) -> DecisionTreeClassifier:
        """Create the classifier used for both training and fold-local validation."""
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )

    def _infer_decision_shape(self, outcome: DecisionOutcome) -> Tuple[int, int]:
        """Infer decision dimensions from outcome metadata.

        When no decision-shape metadata is available, the fallback duplicates a
        conservative proxy based on outcome metric count. That last-resort path
        is intentionally documented here because the engine needs richer stored
        decision metadata to infer independent option and criteria counts.
        """
        context = outcome.context_at_decision
        if context is None:
            return self._fallback_decision_shape(outcome)

        domain_knowledge = context.domain_knowledge or {}
        user_preferences = context.user_preferences or {}

        num_options = self._coerce_positive_int(
            domain_knowledge.get('num_options')
            or domain_knowledge.get('decision_num_options')
            or user_preferences.get('num_options')
            or user_preferences.get('decision_num_options')
        )
        if num_options is None:
            alternatives = domain_knowledge.get('alternatives') or user_preferences.get('alternatives')
            if isinstance(alternatives, dict):
                num_options = len(alternatives)
            elif isinstance(alternatives, list):
                num_options = len(alternatives)

        criteria = (
            domain_knowledge.get('criteria')
            or user_preferences.get('criteria')
            or user_preferences.get('criterion_preferences')
            or user_preferences.get('objective_weights')
        )
        if isinstance(criteria, dict):
            num_criteria = len(criteria)
        elif isinstance(criteria, list):
            num_criteria = len(criteria)
        else:
            num_criteria = self._coerce_positive_int(
                domain_knowledge.get('num_criteria')
                or domain_knowledge.get('decision_num_criteria')
                or user_preferences.get('num_criteria')
                or user_preferences.get('decision_num_criteria')
            )

        if num_options is not None and num_criteria is not None:
            return num_options, num_criteria
        if num_options is not None:
            return num_options, num_criteria or 1
        if num_criteria is not None:
            return 1, num_criteria

        return self._fallback_decision_shape(outcome)

    def _fallback_decision_shape(self, outcome: DecisionOutcome) -> Tuple[int, int]:
        """Last-resort shape inference when the original decision metadata is unavailable."""
        fallback = max(1, len(outcome.outcome_metrics))
        logger.debug(
            "Falling back to duplicated decision-shape proxy for outcome %s; richer decision metadata is unavailable",
            outcome.decision_id,
        )
        return fallback, fallback

    def _coerce_positive_int(self, value: Any) -> Optional[int]:
        """Convert a candidate size value into a positive integer when possible."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    
    def _extract_features(
        self,
        context: EnhancedDecisionContext,
        num_options: int,
        num_criteria: int
    ) -> DecisionFeatures:
        """Extract features from context"""
        domain_knowledge_signal = self._extract_domain_knowledge_signal(context.domain_knowledge)
        
        return DecisionFeatures(
            cognitive_load=context.cognitive_load,
            time_pressure=context.time_pressure,
            risk_tolerance=context.risk_tolerance,
            num_options=num_options,
            num_criteria=num_criteria,
            domain_knowledge_signal=domain_knowledge_signal,
        )

    def _extract_domain_knowledge_signal(self, domain_knowledge: Dict[str, Any]) -> float:
        """Convert domain knowledge structure into a smooth complexity-like feature."""
        if not domain_knowledge:
            return 0.0

        normalized_context = json.dumps(
            domain_knowledge,
            sort_keys=True,
            default=str,
        )
        breadth = min(1.0, len(domain_knowledge) / 12.0)
        content_length = min(1.0, len(normalized_context) / 600.0)
        depth = min(1.0, self._estimate_structure_depth(domain_knowledge) / 6.0)

        return float((0.4 * breadth) + (0.4 * content_length) + (0.2 * depth))

    def _estimate_structure_depth(
        self,
        value: Any,
        current_depth: int = 0,
        max_depth: int = 32,
    ) -> int:
        """Estimate nested depth for dict/list structures used in domain knowledge."""
        if current_depth >= max_depth:
            return max_depth

        if isinstance(value, dict):
            if not value:
                return current_depth + 1
            return max(
                self._estimate_structure_depth(child, current_depth + 1, max_depth)
                for child in value.values()
            )
        if isinstance(value, list):
            if not value:
                return current_depth + 1
            return max(
                self._estimate_structure_depth(child, current_depth + 1, max_depth)
                for child in value
            )
        return current_depth
