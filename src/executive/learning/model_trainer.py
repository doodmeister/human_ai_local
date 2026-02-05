"""
Model Trainer - Week 16 Phase 3

Trains ML models from execution outcomes for continuous improvement.
Implements strategy classification, success prediction, and time estimation.

Key responsibilities:
- Train models from feature vectors
- Evaluate model performance
- Persist trained models
- Support hyperparameter tuning
"""

import json
import logging
import joblib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from .feature_extractor import FeatureVector

logger = logging.getLogger(__name__)


ModelType = Literal['strategy_classifier', 'success_classifier', 'time_regressor', 'outcome_regressor']


@dataclass
class ModelMetadata:
    """Metadata for trained model."""
    model_type: ModelType
    version: str
    training_date: datetime
    n_samples: int
    n_features: int
    test_accuracy: float
    test_metrics: Dict[str, Any]
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type,
            'version': self.version,
            'training_date': self.training_date.isoformat(),
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'test_accuracy': self.test_accuracy,
            'test_metrics': self.test_metrics,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data['training_date'] = datetime.fromisoformat(data['training_date'])
        return ModelMetadata(**data)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1  # Use all CPUs
    
    # Strategy classifier
    strategy_n_estimators: int = 100
    strategy_max_depth: Optional[int] = 10
    
    # Success classifier
    success_n_estimators: int = 100
    success_learning_rate: float = 0.1
    success_max_depth: int = 5
    
    # Time regressor
    time_n_estimators: int = 100
    time_max_depth: Optional[int] = 15
    
    # Outcome regressor
    outcome_n_estimators: int = 100
    outcome_learning_rate: float = 0.1
    outcome_max_depth: int = 5


@dataclass
class TrainingResult:
    """Results from model training."""
    model_type: ModelType
    success: bool
    metadata: Optional[ModelMetadata] = None
    error: Optional[str] = None
    cv_scores: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None


class ModelTrainer:
    """
    Service for training ML models from execution outcomes.
    
    Trains four model types:
    1. Strategy Classifier - Predict best decision strategy
    2. Success Classifier - Predict goal success probability
    3. Time Regressor - Predict time accuracy ratio
    4. Outcome Regressor - Predict outcome quality score
    
    Usage:
        trainer = ModelTrainer()
        result = trainer.train_strategy_classifier(feature_vectors)
        if result.success:
            print(f"Accuracy: {result.metadata.test_accuracy:.1%}")
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration
            models_dir: Directory for storing trained models
        """
        self.config = config or TrainingConfig()
        self.models_dir = models_dir or Path("data/models/learning")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelTrainer initialized (models_dir={self.models_dir})")
    
    def train_strategy_classifier(
        self,
        feature_vectors: List[FeatureVector],
        tune_hyperparameters: bool = False,
    ) -> TrainingResult:
        """
        Train model to predict best decision strategy.
        
        Args:
            feature_vectors: Training data
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            TrainingResult with model metadata
        """
        try:
            # Extract features and target
            X, feature_names = self._extract_features(
                feature_vectors,
                include_decision=False,  # Don't include strategy as feature
                include_targets=False
            )
            y = np.array([fv.decision_strategy for fv in feature_vectors])
            
            # Check sufficient data
            if len(X) < 10:
                return TrainingResult(
                    model_type='strategy_classifier',
                    success=False,
                    error=f"Insufficient data: {len(X)} samples (need >= 10)"
                )
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Create model
            if tune_hyperparameters:
                model = self._tune_strategy_classifier(X_train, y_train)
            else:
                model = RandomForestClassifier(
                    n_estimators=self.config.strategy_n_estimators,
                    max_depth=self.config.strategy_max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=min(self.config.cv_folds, len(X)),
                n_jobs=self.config.n_jobs
            )
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Build metadata
            metadata = ModelMetadata(
                model_type='strategy_classifier',
                version='1.0',
                training_date=datetime.now(),
                n_samples=len(X),
                n_features=len(feature_names),
                test_accuracy=float(accuracy),
                test_metrics={
                    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                },
                feature_names=feature_names,
                hyperparameters=model.get_params(),
            )
            
            # Save model
            self._save_model(model, metadata)
            
            logger.info(
                f"Strategy classifier trained: accuracy={accuracy:.1%}, "
                f"cv={np.mean(cv_scores):.1%}±{np.std(cv_scores):.1%}"
            )
            
            return TrainingResult(
                model_type='strategy_classifier',
                success=True,
                metadata=metadata,
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            logger.error(f"Failed to train strategy classifier: {e}")
            return TrainingResult(
                model_type='strategy_classifier',
                success=False,
                error=str(e)
            )
    
    def train_success_classifier(
        self,
        feature_vectors: List[FeatureVector],
        tune_hyperparameters: bool = False,
    ) -> TrainingResult:
        """
        Train model to predict goal success probability.
        
        Args:
            feature_vectors: Training data
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            TrainingResult with model metadata
        """
        try:
            # Extract features and target
            X, feature_names = self._extract_features(
                feature_vectors,
                include_decision=True,
                include_targets=False
            )
            y = np.array([fv.success for fv in feature_vectors])
            
            # Check sufficient data
            if len(X) < 10:
                return TrainingResult(
                    model_type='success_classifier',
                    success=False,
                    error=f"Insufficient data: {len(X)} samples (need >= 10)"
                )
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Create model
            if tune_hyperparameters:
                model = self._tune_success_classifier(X_train, y_train)
            else:
                model = GradientBoostingClassifier(
                    n_estimators=self.config.success_n_estimators,
                    learning_rate=self.config.success_learning_rate,
                    max_depth=self.config.success_max_depth,
                    random_state=self.config.random_state,
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=min(self.config.cv_folds, len(X)),
                n_jobs=self.config.n_jobs
            )
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Build metadata
            metadata = ModelMetadata(
                model_type='success_classifier',
                version='1.0',
                training_date=datetime.now(),
                n_samples=len(X),
                n_features=len(feature_names),
                test_accuracy=float(accuracy),
                test_metrics={
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                },
                feature_names=feature_names,
                hyperparameters=model.get_params(),
            )
            
            # Save model
            self._save_model(model, metadata)
            
            logger.info(
                f"Success classifier trained: accuracy={accuracy:.1%}, "
                f"cv={np.mean(cv_scores):.1%}±{np.std(cv_scores):.1%}"
            )
            
            return TrainingResult(
                model_type='success_classifier',
                success=True,
                metadata=metadata,
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            logger.error(f"Failed to train success classifier: {e}")
            return TrainingResult(
                model_type='success_classifier',
                success=False,
                error=str(e)
            )
    
    def train_time_regressor(
        self,
        feature_vectors: List[FeatureVector],
        tune_hyperparameters: bool = False,
    ) -> TrainingResult:
        """
        Train model to predict time accuracy ratio.
        
        Args:
            feature_vectors: Training data
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            TrainingResult with model metadata
        """
        try:
            # Extract features and target
            X, feature_names = self._extract_features(
                feature_vectors,
                include_decision=True,
                include_targets=False
            )
            y = np.array([fv.time_accuracy_ratio for fv in feature_vectors])
            
            # Check sufficient data
            if len(X) < 10:
                return TrainingResult(
                    model_type='time_regressor',
                    success=False,
                    error=f"Insufficient data: {len(X)} samples (need >= 10)"
                )
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Create model
            if tune_hyperparameters:
                model = self._tune_time_regressor(X_train, y_train)
            else:
                model = RandomForestRegressor(
                    n_estimators=self.config.time_n_estimators,
                    max_depth=self.config.time_max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation (negative MSE)
            cv_scores = -cross_val_score(
                model, X, y,
                cv=min(self.config.cv_folds, len(X)),
                scoring='neg_mean_squared_error',
                n_jobs=self.config.n_jobs
            )
            cv_rmse = np.sqrt(cv_scores)
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Build metadata
            metadata = ModelMetadata(
                model_type='time_regressor',
                version='1.0',
                training_date=datetime.now(),
                n_samples=len(X),
                n_features=len(feature_names),
                test_accuracy=float(r2),  # Use R² as accuracy
                test_metrics={
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'cv_rmse_mean': float(np.mean(cv_rmse)),
                    'cv_rmse_std': float(np.std(cv_rmse)),
                },
                feature_names=feature_names,
                hyperparameters=model.get_params(),
            )
            
            # Save model
            self._save_model(model, metadata)
            
            logger.info(
                f"Time regressor trained: R²={r2:.2f}, RMSE={rmse:.3f}, "
                f"cv_rmse={np.mean(cv_rmse):.3f}±{np.std(cv_rmse):.3f}"
            )
            
            return TrainingResult(
                model_type='time_regressor',
                success=True,
                metadata=metadata,
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            logger.error(f"Failed to train time regressor: {e}")
            return TrainingResult(
                model_type='time_regressor',
                success=False,
                error=str(e)
            )
    
    def train_outcome_regressor(
        self,
        feature_vectors: List[FeatureVector],
        tune_hyperparameters: bool = False,
    ) -> TrainingResult:
        """
        Train model to predict outcome quality score.
        
        Args:
            feature_vectors: Training data
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            TrainingResult with model metadata
        """
        try:
            # Extract features and target
            X, feature_names = self._extract_features(
                feature_vectors,
                include_decision=True,
                include_targets=False
            )
            y = np.array([fv.outcome_score for fv in feature_vectors])
            
            # Check sufficient data
            if len(X) < 10:
                return TrainingResult(
                    model_type='outcome_regressor',
                    success=False,
                    error=f"Insufficient data: {len(X)} samples (need >= 10)"
                )
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Create model
            if tune_hyperparameters:
                model = self._tune_outcome_regressor(X_train, y_train)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=self.config.outcome_n_estimators,
                    learning_rate=self.config.outcome_learning_rate,
                    max_depth=self.config.outcome_max_depth,
                    random_state=self.config.random_state,
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = -cross_val_score(
                model, X, y,
                cv=min(self.config.cv_folds, len(X)),
                scoring='neg_mean_squared_error',
                n_jobs=self.config.n_jobs
            )
            cv_rmse = np.sqrt(cv_scores)
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Build metadata
            metadata = ModelMetadata(
                model_type='outcome_regressor',
                version='1.0',
                training_date=datetime.now(),
                n_samples=len(X),
                n_features=len(feature_names),
                test_accuracy=float(r2),
                test_metrics={
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'cv_rmse_mean': float(np.mean(cv_rmse)),
                    'cv_rmse_std': float(np.std(cv_rmse)),
                },
                feature_names=feature_names,
                hyperparameters=model.get_params(),
            )
            
            # Save model
            self._save_model(model, metadata)
            
            logger.info(
                f"Outcome regressor trained: R²={r2:.2f}, RMSE={rmse:.3f}, "
                f"cv_rmse={np.mean(cv_rmse):.3f}±{np.std(cv_rmse):.3f}"
            )
            
            return TrainingResult(
                model_type='outcome_regressor',
                success=True,
                metadata=metadata,
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            logger.error(f"Failed to train outcome regressor: {e}")
            return TrainingResult(
                model_type='outcome_regressor',
                success=False,
                error=str(e)
            )
    
    def train_all_models(
        self,
        feature_vectors: List[FeatureVector],
        tune_hyperparameters: bool = False,
    ) -> Dict[ModelType, TrainingResult]:
        """
        Train all model types.
        
        Args:
            feature_vectors: Training data
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            Dict mapping model type to training result
        """
        results = {}
        
        results['strategy_classifier'] = self.train_strategy_classifier(
            feature_vectors, tune_hyperparameters
        )
        
        results['success_classifier'] = self.train_success_classifier(
            feature_vectors, tune_hyperparameters
        )
        
        results['time_regressor'] = self.train_time_regressor(
            feature_vectors, tune_hyperparameters
        )
        
        results['outcome_regressor'] = self.train_outcome_regressor(
            feature_vectors, tune_hyperparameters
        )
        
        # Summary
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Trained {successful}/4 models successfully")
        
        return results
    
    # Helper methods
    
    def _extract_features(
        self,
        feature_vectors: List[FeatureVector],
        include_decision: bool = True,
        include_targets: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract feature matrix and feature names."""
        features = []
        feature_names = []
        
        for fv in feature_vectors:
            row = []
            
            # Decision features
            if include_decision:
                # One-hot encode strategy
                for strategy in ['weighted_scoring', 'ahp', 'pareto']:
                    row.append(1 if fv.decision_strategy == strategy else 0)
                    if not feature_names or len(feature_names) < len(row):
                        feature_names.append(f'strategy_{strategy}')
                
                row.append(fv.decision_confidence)
                if not feature_names or len(feature_names) < len(row):
                    feature_names.append('decision_confidence')
                
                row.append(fv.decision_time_ms)
                if not feature_names or len(feature_names) < len(row):
                    feature_names.append('decision_time_ms')
            
            # Planning features
            row.extend([
                fv.plan_length,
                fv.plan_cost,
                fv.planning_time_ms,
                fv.nodes_expanded,
            ])
            if not feature_names or len(feature_names) < len(row):
                feature_names.extend([
                    'plan_length', 'plan_cost',
                    'planning_time_ms', 'nodes_expanded'
                ])
            
            # Scheduling features
            row.extend([
                fv.predicted_makespan_minutes,
                fv.task_count,
            ])
            if not feature_names or len(feature_names) < len(row):
                feature_names.extend([
                    'predicted_makespan_minutes', 'task_count'
                ])
            
            # Context features
            row.extend([
                fv.hour_of_day,
                fv.day_of_week,
            ])
            if not feature_names or len(feature_names) < len(row):
                feature_names.extend(['hour_of_day', 'day_of_week'])
            
            # Targets (if requested)
            if include_targets:
                row.extend([
                    fv.success,
                    fv.outcome_score,
                    fv.time_accuracy_ratio,
                    fv.plan_adherence_score,
                ])
                if not feature_names or len(feature_names) < len(row):
                    feature_names.extend([
                        'success', 'outcome_score',
                        'time_accuracy_ratio', 'plan_adherence_score'
                    ])
            
            features.append(row)
        
        return np.array(features), feature_names
    
    def _tune_strategy_classifier(self, X_train, y_train):
        """Tune hyperparameters for strategy classifier."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.config.random_state),
            param_grid,
            cv=3,
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _tune_success_classifier(self, X_train, y_train):
        """Tune hyperparameters for success classifier."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
        
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=self.config.random_state),
            param_grid,
            cv=3,
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _tune_time_regressor(self, X_train, y_train):
        """Tune hyperparameters for time regressor."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.config.random_state),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _tune_outcome_regressor(self, X_train, y_train):
        """Tune hyperparameters for outcome regressor."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=self.config.random_state),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _save_model(self, model: Any, metadata: ModelMetadata) -> None:
        """Save model and metadata to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = self.models_dir / f"{metadata.model_type}_{timestamp}.joblib"
        metadata_file = self.models_dir / f"{metadata.model_type}_{timestamp}.json"
        
        # Save model
        joblib.dump(model, model_file)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.debug(f"Saved model to {model_file}")


def create_model_trainer(
    config: Optional[TrainingConfig] = None,
    models_dir: Optional[Path] = None,
) -> ModelTrainer:
    """Factory function for creating ModelTrainer."""
    return ModelTrainer(config=config, models_dir=models_dir)
