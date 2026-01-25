# Week 16 Phase 2 Quick Reference

**Feature Extraction for ML Training Data**

## One-Liners

```python
# Get training dataset as CSV
features = OutcomeTracker().get_training_dataset(limit=100, export_path=Path("train.csv"))

# Extract and normalize
extractor = FeatureExtractor()
features = extractor.extract_dataset(outcomes)
normalized, params = extractor.normalize_features(features, method='standard')

# DataFrame for sklearn
df = extractor.to_dataframe(features, include_targets=True)
X, y = df.drop(columns=['success', ...]), df['success']
```

## Feature Schema (23 fields)

**Decision** (3): strategy, confidence, decision_time_ms  
**Planning** (4): plan_length, plan_cost, planning_time_ms, nodes_expanded  
**Scheduling** (2): predicted_makespan_minutes, task_count  
**Context** (2): hour_of_day (0-23), day_of_week (0-6)  
**Targets** (4): success (0/1), outcome_score (0-1), time_accuracy_ratio, plan_adherence_score

## Common Operations

### Export Formats
```python
extractor.export_csv(features, Path("train.csv"))       # CSV
extractor.export_json(features, Path("train.json"))     # JSON
extractor.export_parquet(features, Path("train.parquet")) # Parquet (fast)
```

### Normalization
```python
# Standard (mean=0, std=1)
normalized, params = extractor.normalize_features(features, method='standard')

# MinMax (0-1 range)
normalized, params = extractor.normalize_features(features, method='minmax')
```

### Statistics
```python
stats = extractor.get_feature_statistics(features)
print(f"Samples: {stats['n_samples']}, Success rate: {stats['success_rate']:.1%}")
```

### Missing Values
```python
filled = extractor.handle_missing_values(features, strategy='median')  # or 'mean', 'zero'
```

## Integration with OutcomeTracker

```python
tracker = OutcomeTracker()

# Filter and export
features = tracker.get_training_dataset(
    limit=100,                          # Recent 100
    strategy="ahp",                     # AHP decisions only
    success_only=True,                  # Successes only
    export_path=Path("ahp_success.csv"),
    export_format='csv'                 # or 'json', 'parquet'
)
```

## sklearn Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Get features
tracker = OutcomeTracker()
features = tracker.get_training_dataset(limit=500)

# Convert to DataFrame
df = extractor.to_dataframe(features, include_targets=True)
X = df.drop(columns=['success', 'outcome_score', 'time_accuracy_ratio', 'plan_adherence_score'])
y = df['success']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.1%}")
```

## File Locations

**Source**: `src/executive/learning/feature_extractor.py`  
**Tests**: `tests/test_feature_extraction.py`  
**Docs**: `docs/archive/WEEK_16_PHASE_2_FEATURE_EXTRACTION.md`  
**Data**: `data/outcomes/` (OutcomeRecord JSON files)

## Import Paths

```python
from src.executive.learning import (
    FeatureExtractor,
    FeatureVector,
    OutcomeTracker,
    create_feature_extractor,
)
```

## Performance

- Extract: ~1ms/outcome
- DataFrame: ~5ms/100 features
- Export CSV: ~20ms/100 features
- Normalize: ~10ms/100 features

## Tests

```bash
# All feature extraction tests (21)
pytest tests/test_feature_extraction.py -v

# All learning tests (41 = 21 Phase 2 + 20 Phase 1)
pytest tests/test_feature_extraction.py tests/test_outcome_tracking.py -v
```

**Status**: 21/21 tests passing ✅
