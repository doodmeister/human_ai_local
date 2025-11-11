"""
Model Predictor - Week 16 Phase 3

Inference service for making predictions using trained ML models.
Loads models from disk and provides prediction interface.

Key responsibilities:
- Load trained models
- Make predictions on new goals
- Provide confidence scores
- Support batch predictions
"""

import logging
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np

from .model_trainer import ModelType, ModelMetadata
from .feature_extractor import FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from model prediction."""
    model_type: ModelType
    prediction: Any
    confidence: float
    model_version: str
    model_accuracy: float


class ModelPredictor:
    """
    Service for making predictions using trained models.
    
    Loads models from disk and provides prediction interface.
    Supports:
    - Strategy prediction (best decision strategy)
    - Success prediction (goal success probability)
    - Time prediction (time accuracy ratio)
    - Outcome prediction (outcome quality score)
    
    Usage:
        predictor = ModelPredictor()
        strategy = predictor.predict_strategy(feature_vector)
        success_prob = predictor.predict_success(feature_vector)
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir or Path("data/models/learning")
        
        # Lazy-loaded models
        self._models: Dict[ModelType, Any] = {}
        self._metadata: Dict[ModelType, ModelMetadata] = {}
        
        logger.info(f"ModelPredictor initialized (models_dir={self.models_dir})")
    
    def predict_strategy(
        self,
        feature_vector: FeatureVector,
        return_probabilities: bool = False,
    ) -> PredictionResult:
        """
        Predict best decision strategy.
        
        Args:
            feature_vector: Features for prediction
            return_probabilities: Return class probabilities
        
        Returns:
            PredictionResult with predicted strategy
        """
        model, metadata = self._load_model('strategy_classifier')
        
        # Extract features (without decision features)
        X = self._extract_single_feature(feature_vector, include_decision=False)
        
        # Predict
        prediction = model.predict([X])[0]
        
        # Get probabilities for confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([X])[0]
            confidence = float(np.max(proba))
            
            if return_probabilities:
                # Return full probability distribution
                classes = model.classes_
                prediction = dict(zip(classes, proba))
        else:
            confidence = 0.5  # Default confidence if no probabilities
        
        return PredictionResult(
            model_type='strategy_classifier',
            prediction=prediction,
            confidence=confidence,
            model_version=metadata.version,
            model_accuracy=metadata.test_accuracy,
        )
    
    def predict_success(
        self,
        feature_vector: FeatureVector,
    ) -> PredictionResult:
        """
        Predict goal success probability.
        
        Args:
            feature_vector: Features for prediction
        
        Returns:
            PredictionResult with success probability
        """
        model, metadata = self._load_model('success_classifier')
        
        # Extract features
        X = self._extract_single_feature(feature_vector, include_decision=True)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([X])[0]
            # Probability of success (class 1)
            success_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Fallback to binary prediction
            success_prob = float(model.predict([X])[0])
        
        return PredictionResult(
            model_type='success_classifier',
            prediction=success_prob,
            confidence=abs(success_prob - 0.5) * 2,  # Distance from 0.5, scaled to 0-1
            model_version=metadata.version,
            model_accuracy=metadata.test_accuracy,
        )
    
    def predict_time_accuracy(
        self,
        feature_vector: FeatureVector,
    ) -> PredictionResult:
        """
        Predict time accuracy ratio (actual/predicted time).
        
        Args:
            feature_vector: Features for prediction
        
        Returns:
            PredictionResult with time accuracy ratio
        """
        model, metadata = self._load_model('time_regressor')
        
        # Extract features
        X = self._extract_single_feature(feature_vector, include_decision=True)
        
        # Predict
        time_ratio = float(model.predict([X])[0])
        
        # Confidence based on model R²
        confidence = max(0.0, metadata.test_accuracy)
        
        return PredictionResult(
            model_type='time_regressor',
            prediction=time_ratio,
            confidence=confidence,
            model_version=metadata.version,
            model_accuracy=metadata.test_accuracy,
        )
    
    def predict_outcome_score(
        self,
        feature_vector: FeatureVector,
    ) -> PredictionResult:
        """
        Predict outcome quality score (0-1).
        
        Args:
            feature_vector: Features for prediction
        
        Returns:
            PredictionResult with outcome score
        """
        model, metadata = self._load_model('outcome_regressor')
        
        # Extract features
        X = self._extract_single_feature(feature_vector, include_decision=True)
        
        # Predict
        outcome_score = float(model.predict([X])[0])
        
        # Clip to valid range
        outcome_score = np.clip(outcome_score, 0.0, 1.0)
        
        # Confidence based on model R²
        confidence = max(0.0, metadata.test_accuracy)
        
        return PredictionResult(
            model_type='outcome_regressor',
            prediction=outcome_score,
            confidence=confidence,
            model_version=metadata.version,
            model_accuracy=metadata.test_accuracy,
        )
    
    def predict_all(
        self,
        feature_vector: FeatureVector,
    ) -> Dict[ModelType, PredictionResult]:
        """
        Make predictions with all available models.
        
        Args:
            feature_vector: Features for prediction
        
        Returns:
            Dict mapping model type to prediction result
        """
        results = {}
        
        try:
            results['strategy_classifier'] = self.predict_strategy(feature_vector)
        except Exception as e:
            logger.warning(f"Strategy prediction failed: {e}")
        
        try:
            results['success_classifier'] = self.predict_success(feature_vector)
        except Exception as e:
            logger.warning(f"Success prediction failed: {e}")
        
        try:
            results['time_regressor'] = self.predict_time_accuracy(feature_vector)
        except Exception as e:
            logger.warning(f"Time prediction failed: {e}")
        
        try:
            results['outcome_regressor'] = self.predict_outcome_score(feature_vector)
        except Exception as e:
            logger.warning(f"Outcome prediction failed: {e}")
        
        return results
    
    def batch_predict_strategy(
        self,
        feature_vectors: List[FeatureVector],
    ) -> List[PredictionResult]:
        """
        Batch predict strategies for multiple goals.
        
        Args:
            feature_vectors: List of features
        
        Returns:
            List of prediction results
        """
        model, metadata = self._load_model('strategy_classifier')
        
        # Extract features
        X = np.array([
            self._extract_single_feature(fv, include_decision=False)
            for fv in feature_vectors
        ])
        
        # Predict
        predictions = model.predict(X)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            confidences = np.max(probas, axis=1)
        else:
            confidences = np.full(len(predictions), 0.5)
        
        # Build results
        results = []
        for pred, conf in zip(predictions, confidences):
            results.append(PredictionResult(
                model_type='strategy_classifier',
                prediction=pred,
                confidence=float(conf),
                model_version=metadata.version,
                model_accuracy=metadata.test_accuracy,
            ))
        
        return results
    
    def batch_predict_success(
        self,
        feature_vectors: List[FeatureVector],
    ) -> List[PredictionResult]:
        """
        Batch predict success probabilities.
        
        Args:
            feature_vectors: List of features
        
        Returns:
            List of prediction results
        """
        model, metadata = self._load_model('success_classifier')
        
        # Extract features
        X = np.array([
            self._extract_single_feature(fv, include_decision=True)
            for fv in feature_vectors
        ])
        
        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            success_probs = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        else:
            success_probs = model.predict(X)
        
        # Build results
        results = []
        for prob in success_probs:
            results.append(PredictionResult(
                model_type='success_classifier',
                prediction=float(prob),
                confidence=abs(float(prob) - 0.5) * 2,
                model_version=metadata.version,
                model_accuracy=metadata.test_accuracy,
            ))
        
        return results
    
    def get_model_info(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """
        Get metadata for a model.
        
        Args:
            model_type: Type of model
        
        Returns:
            ModelMetadata or None if model not found
        """
        try:
            _, metadata = self._load_model(model_type)
            return metadata
        except FileNotFoundError:
            return None
    
    def list_available_models(self) -> List[ModelType]:
        """
        List available trained models.
        
        Returns:
            List of model types that are available
        """
        available: List[ModelType] = []
        
        model_types: List[ModelType] = [
            'strategy_classifier',
            'success_classifier',
            'time_regressor',
            'outcome_regressor'
        ]
        
        for model_type in model_types:
            try:
                self._load_model(model_type)
                available.append(model_type)
            except FileNotFoundError:
                pass
        
        return available
    
    def clear_cache(self) -> None:
        """Clear cached models (force reload on next prediction)."""
        self._models.clear()
        self._metadata.clear()
        logger.debug("Cleared model cache")
    
    # Helper methods
    
    def _load_model(self, model_type: ModelType) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata from disk (with caching)."""
        # Check cache
        if model_type in self._models:
            return self._models[model_type], self._metadata[model_type]
        
        # Find most recent model file
        model_files = sorted(
            self.models_dir.glob(f"{model_type}_*.joblib"),
            reverse=True
        )
        
        if not model_files:
            raise FileNotFoundError(f"No trained model found for {model_type}")
        
        model_file = model_files[0]
        metadata_file = model_file.with_suffix('.json')
        
        # Load model
        model = joblib.load(model_file)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Cache
        self._models[model_type] = model
        self._metadata[model_type] = metadata
        
        logger.debug(f"Loaded {model_type} from {model_file}")
        
        return model, metadata
    
    def _extract_single_feature(
        self,
        feature_vector: FeatureVector,
        include_decision: bool = True,
    ) -> np.ndarray:
        """Extract feature array from single feature vector."""
        features = []
        
        # Decision features
        if include_decision:
            # One-hot encode strategy
            for strategy in ['weighted_scoring', 'ahp', 'pareto']:
                features.append(1 if feature_vector.decision_strategy == strategy else 0)
            
            features.append(feature_vector.decision_confidence)
            features.append(feature_vector.decision_time_ms)
        
        # Planning features
        features.extend([
            feature_vector.plan_length,
            feature_vector.plan_cost,
            feature_vector.planning_time_ms,
            feature_vector.nodes_expanded,
        ])
        
        # Scheduling features
        features.extend([
            feature_vector.predicted_makespan_minutes,
            feature_vector.task_count,
        ])
        
        # Context features
        features.extend([
            feature_vector.hour_of_day,
            feature_vector.day_of_week,
        ])
        
        return np.array(features)


def create_model_predictor(models_dir: Optional[Path] = None) -> ModelPredictor:
    """Factory function for creating ModelPredictor."""
    return ModelPredictor(models_dir=models_dir)
