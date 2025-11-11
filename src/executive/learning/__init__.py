"""
Executive Learning Module - Week 16

Learning infrastructure for continuous improvement:
- Outcome tracking: Record and analyze execution results
- Feature extraction: Convert outcomes to ML training data
- Model training: Train ML models from outcomes
- Model prediction: Make predictions on new goals
- A/B testing: Compare strategies scientifically
"""

from src.executive.learning.outcome_tracker import (
    OutcomeTracker,
    OutcomeRecord,
    AccuracyMetrics,
)
from src.executive.learning.feature_extractor import (
    FeatureExtractor,
    FeatureVector,
    create_feature_extractor,
)
from src.executive.learning.model_trainer import (
    ModelTrainer,
    ModelMetadata,
    TrainingConfig,
    TrainingResult,
    create_model_trainer,
)
from src.executive.learning.model_predictor import (
    ModelPredictor,
    PredictionResult,
    create_model_predictor,
)
from src.executive.learning.experiment_manager import (
    ExperimentManager,
    StrategyExperiment,
    ExperimentAssignment,
    StrategyOutcome,
    ExperimentStatus,
    AssignmentMethod,
    create_experiment_manager,
)
from src.executive.learning.experiment_analyzer import (
    StrategyPerformance,
    ComparisonResult,
    SignificanceTest,
    calculate_confidence_interval,
    calculate_proportion_confidence_interval,
    cohens_d,
    interpret_effect_size,
    chi_square_test,
    t_test,
    mann_whitney_test,
    recommend_strategy,
)

__all__ = [
    "OutcomeTracker",
    "OutcomeRecord", 
    "AccuracyMetrics",
    "FeatureExtractor",
    "FeatureVector",
    "create_feature_extractor",
    "ModelTrainer",
    "ModelMetadata",
    "TrainingConfig",
    "TrainingResult",
    "create_model_trainer",
    "ModelPredictor",
    "PredictionResult",
    "create_model_predictor",
    # Phase 4: Experiment management
    "ExperimentManager",
    "StrategyExperiment",
    "ExperimentAssignment",
    "StrategyOutcome",
    "ExperimentStatus",
    "AssignmentMethod",
    "create_experiment_manager",
    # Phase 4: Statistical analysis
    "StrategyPerformance",
    "ComparisonResult",
    "SignificanceTest",
    "calculate_confidence_interval",
    "calculate_proportion_confidence_interval",
    "cohens_d",
    "interpret_effect_size",
    "chi_square_test",
    "t_test",
    "mann_whitney_test",
    "recommend_strategy",
]

