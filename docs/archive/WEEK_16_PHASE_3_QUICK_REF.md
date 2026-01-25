# Week 16 Phase 3 - Quick Reference

**ML Training Pipeline Cheat Sheet**

## Quick Start

### 1. Train Models (one-time setup)

```python
from src.executive.learning import (
    OutcomeTracker, FeatureExtractor, create_model_trainer
)

# Get training data
tracker = OutcomeTracker()
extractor = FeatureExtractor()
outcomes = tracker.get_recent_outcomes(limit=100)
features = [extractor.extract_from_outcome(o) for o in outcomes]

# Train all models
trainer = create_model_trainer()
results = trainer.train_all_models(features)
```

### 2. Use ML-Enhanced Decisions

```python
from src.executive.decision_engine import DecisionEngine, DecisionOption, DecisionCriterion

# Create engine with ML
engine = DecisionEngine(enable_ml_predictions=True)

# Make decision with context
result = engine.make_decision(
    options, criteria,
    context={'goal_id': 'analyze_data', 'plan_length': 5, 'task_count': 5}
)

# Check ML boost
print(f"Confidence: {result.confidence:.2%}")
print(f"ML boost: {result.metadata.get('ml_confidence_boost', 0):+.2%}")
```

## Common Commands

### Training

```python
# Train single model
trainer.train_strategy_classifier(features)
trainer.train_success_classifier(features)
trainer.train_time_regressor(features)
trainer.train_outcome_regressor(features)

# Train with hyperparameter tuning (slower)
trainer.train_all_models(features, tune_hyperparameters=True)
```

### Prediction

```python
from src.executive.learning import create_model_predictor

predictor = create_model_predictor()

# Single predictions
strategy = predictor.predict_strategy(features)
success = predictor.predict_success(features)
time = predictor.predict_time_accuracy(features)
outcome = predictor.predict_outcome_score(features)

# All at once
all_preds = predictor.predict_all(features)

# Batch predictions
strategies = predictor.batch_predict_strategy(features_list)
successes = predictor.batch_predict_success(features_list)
```

### Model Management

```python
# List available models
models = predictor.list_available_models()

# Get model info
info = predictor.get_model_info('strategy_classifier')
print(f"Accuracy: {info.test_accuracy:.2%}")

# Clear cached models
predictor.clear_cache()
```

## Feature Vector Construction

```python
from src.executive.learning import FeatureVector
from datetime import datetime

features = FeatureVector(
    # Identification
    record_id="temp_001",
    goal_id="analyze_data",
    timestamp=datetime.now(),
    
    # Decision features
    decision_strategy="weighted_scoring",
    decision_confidence=0.7,
    decision_time_ms=100.0,
    
    # Planning features
    plan_length=5,
    plan_cost=10.0,
    planning_time_ms=200.0,
    nodes_expanded=50,
    
    # Scheduling features
    predicted_makespan_minutes=30.0,
    task_count=5,
    
    # Context features
    hour_of_day=14,
    day_of_week=1,
    
    # Target variables
    success=1,
    outcome_score=0.5,
    time_accuracy_ratio=1.0,
    plan_adherence_score=1.0
)
```

## Model Types Quick Reference

| Model | Type | Purpose | Output |
|-------|------|---------|--------|
| Strategy Classifier | RandomForest | Best strategy | weighted_scoring/ahp/pareto |
| Success Classifier | GradientBoosting | Success probability | 0-1 float |
| Time Regressor | RandomForest | Time accuracy | Ratio (actual/predicted) |
| Outcome Regressor | GradientBoosting | Outcome quality | 0-1 float (clipped) |

## Configuration Options

### TrainingConfig

```python
from src.executive.learning import TrainingConfig

config = TrainingConfig(
    test_size=0.2,        # 80/20 split
    cv_folds=5,           # 5-fold CV
    random_state=42,      # Reproducibility
    hyperparameters={...} # GridSearchCV params
)
```

### DecisionEngine

```python
# Enable ML predictions (default)
engine = DecisionEngine(enable_ml_predictions=True)

# Disable ML predictions
engine = DecisionEngine(enable_ml_predictions=False)
```

## Confidence Boosting Formula

```python
# Base confidence from decision strategy
confidence = base_confidence

# Strategy match: +5% boost
if predicted_strategy == selected_strategy:
    confidence += 0.05 * prediction_confidence

# Success prediction: +15% boost (high) or -10% penalty (low)
if predicted_success > 0.5:
    confidence += 0.15 * (predicted_success - 0.5) * 2
else:
    confidence -= 0.10 * (0.5 - predicted_success) * 2

# Clamp to [0.05, 0.95]
confidence = min(0.95, max(0.05, confidence))
```

## File Locations

```
src/executive/learning/
├── model_trainer.py       # Training service
├── model_predictor.py     # Inference service
├── feature_extractor.py   # Feature extraction
└── outcome_tracker.py     # Outcome recording

tests/
├── test_model_training.py           # Training tests (34/35)
└── test_decision_ml_integration.py  # Integration tests (7/7)

data/models/learning/
├── strategy_classifier_*.joblib     # Trained models
├── strategy_classifier_*.json       # Model metadata
└── ...
```

## Common Issues

### No Models Available
```python
# Check models
predictor.list_available_models()

# Train if empty
trainer.train_all_models(features)
```

### Low Accuracy
```python
# Need more data (100+ samples recommended)
outcomes = tracker.get_recent_outcomes(limit=200)

# Enable hyperparameter tuning
trainer.train_all_models(features, tune_hyperparameters=True)
```

### No ML Boost
```python
# Must provide context with planning metrics
context = {
    'goal_id': 'test',
    'plan_length': 5,
    'task_count': 5,
    'predicted_makespan_minutes': 30.0
}
result = engine.make_decision(options, criteria, context=context)
```

## Testing

```bash
# All training tests
pytest tests/test_model_training.py -v

# Integration tests
pytest tests/test_decision_ml_integration.py -v

# Quick smoke test
pytest tests/test_model_training.py::TestModelTrainer::test_trainer_initialization -v
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Train 1 model | 2-5s | 50 samples, no tuning |
| Train all 4 | 8-20s | 50 samples, no tuning |
| Train with tuning | 30-60s | Per model, GridSearchCV |
| Single prediction | <10ms | Cached model |
| Batch (100) | <100ms | Cached model |
| Model loading | 50-200ms | First use only |

## API Reference

### Factories

```python
from src.executive.learning import (
    create_model_trainer,
    create_model_predictor
)

trainer = create_model_trainer(model_dir="custom/path")
predictor = create_model_predictor(model_dir="custom/path")
```

### Result Objects

```python
# TrainingResult
result.success          # bool
result.metadata        # ModelMetadata
result.error           # Optional[str]
result.cv_scores       # List[float]
result.feature_importance  # Dict[str, float]

# PredictionResult
pred.model_type        # str
pred.prediction        # Union[str, float]
pred.confidence        # float
pred.model_version     # str
pred.model_accuracy    # float

# ModelMetadata
meta.model_type        # str
meta.version           # str
meta.training_date     # datetime
meta.n_samples         # int
meta.test_accuracy     # float
meta.test_metrics      # Dict[str, float]
meta.hyperparameters   # Dict[str, Any]
```

## Exports

```python
from src.executive.learning import (
    # Phase 1: Outcome tracking
    OutcomeTracker,
    OutcomeRecord,
    AccuracyMetrics,
    
    # Phase 2: Feature extraction
    FeatureExtractor,
    FeatureVector,
    
    # Phase 3: Training
    ModelTrainer,
    ModelMetadata,
    TrainingConfig,
    TrainingResult,
    
    # Phase 3: Prediction
    ModelPredictor,
    PredictionResult,
    
    # Factories
    create_model_trainer,
    create_model_predictor,
    create_outcome_tracker,
    create_feature_extractor
)
```

## Dependencies

```bash
pip install scikit-learn>=1.3.0
pip install joblib>=1.3.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
```

## Next Phase

**Week 17 Phase 4: A/B Testing**
- Strategy comparison framework
- Randomized strategy selection
- Performance tracking
- Statistical significance testing
- Automated optimization

---

**Full Documentation**: `docs/archive/WEEK_16_PHASE_3_TRAINING_PIPELINE.md`  
**Completion Summary**: `docs/archive/WEEK_16_PHASE_3_COMPLETION_SUMMARY.md`  
**Status**: ✅ COMPLETE (40/41 tests passing, 98%)
