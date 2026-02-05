# Changelog

All notable changes to the Human-AI Cognition Framework.

---

## [January 2026] - API Compatibility & Deprecation Policy

### API surface stabilization ✅
- **Canonical endpoints**: Prefer unprefixed routes like `/agent/chat`, `/agent/reminders*`, `/memory/*`, `/procedure/*`, `/executive/*`.
- **Legacy aliases**: `/api/*` routes remain available temporarily but return deprecation headers.
- **Deprecation signaling**: Deprecated endpoints return `Deprecation: true` and `Link: <successor>; rel="successor-version"`.
- **Removal milestone**: Deprecated aliases planned for removal on **2026-04-01** (or first release after).

## [November 2025] - Learning Infrastructure (Week 16)

### Phase 4: A/B Testing ✅
- **ExperimentManager**: 3 assignment methods (Random, Epsilon-Greedy, Thompson Sampling)
- **Statistical Analysis**: Chi-square, t-test, Mann-Whitney, Cohen's d
- **DecisionEngine Integration**: Automatic strategy assignment and outcome tracking
- **Recommendation System**: Confidence-based strategy selection
- Tests: 23/23 passing | Docs: `docs/archive/WEEK_16_PHASE_4_AB_TESTING.md`

### Phase 3: ML Training Pipeline ✅
- **Model Trainer**: 4 ML models (strategy classifier, success predictor, time/outcome regressors)
- **Model Predictor**: Batch predictions with caching
- **DecisionEngine ML Boost**: +5% to +15% confidence adjustments
- Tests: 40/41 passing | Docs: `docs/archive/WEEK_16_PHASE_3_TRAINING_PIPELINE.md`

### Phase 2: Feature Extraction ✅
- **FeatureVector**: 23-field structured features for ML training
- **Multi-format Export**: CSV, JSON, Parquet
- **Normalization**: StandardScaler, MinMaxScaler
- Tests: 21/21 passing | Docs: `docs/archive/WEEK_16_PHASE_2_FEATURE_EXTRACTION.md`

### Phase 1: Outcome Tracking ✅
- **OutcomeTracker**: Records execution results with accuracy analysis
- **AccuracyMetrics**: Time, plan adherence, goal achievement tracking
- **Persistent Storage**: JSON files in `data/outcomes/`
- Tests: 20/20 passing | Docs: `docs/archive/WEEK_16_PHASE_1_OUTCOME_TRACKING.md`

---

## [November 2025] - System Integration (Week 15)

### Executive System Integration ✅
- **ExecutiveSystem**: Unified orchestrator for Goal→Decision→Plan→Schedule pipeline
- **ExecutionContext**: Full state tracking with timing metrics
- **System Health**: Monitoring and execution history
- **Pipeline Latency**: 12-15s end-to-end
- Tests: 17/24 passing (core 100% functional) | Docs: `docs/archive/WEEK_15_COMPLETION_SUMMARY.md`

---

## [October 2025] - Dynamic Scheduling (Week 14)

### Real-Time Schedule Adaptation ✅
- **Quality Metrics**: 8 methods (critical path, slack, robustness, etc.)
- **ScheduleMonitor**: Real-time disruption detection
- **ScheduleAnalyzer**: Proactive warnings (resource contention, deadline risks)
- **ScheduleVisualizer**: 7 export formats (Gantt, timeline, resource utilization, etc.)
- Tests: 36/36 passing | Docs: `docs/archive/WEEK_14_COMPLETION_SUMMARY.md`

---

## [October 2025] - Constraint Scheduling (Week 12)

### CP-SAT Scheduler ✅
- **Google OR-Tools Integration**: Constraint satisfaction solver
- **Constraint Types**: Precedence, resource capacity, deadlines, time windows, cognitive load
- **Optimization**: Minimize makespan, maximize priority, weighted objectives
- **30s Solver Timeout**: 4 workers for parallel search
- Tests: 17/17 passing | Docs: `docs/archive/WEEK_12_COMPLETION_SUMMARY.md`

---

## [September 2025] - GOAP Planning (Phase 2)

### Goal-Oriented Action Planning ✅
- **GOAPPlanner**: A* search with 5 heuristics (goal_distance, relaxed_plan, weighted, composite, zero)
- **WorldState**: Immutable state representation with precondition/effect matching
- **ActionLibrary**: 10 predefined actions (gather_data, analyze_data, create_document, etc.)
- **Constraints**: 5 types (resource, temporal, dependency, state, custom)
- **ReplanningEngine**: Failure detection and plan repair
- **Telemetry**: 10 metrics tracked via metrics_registry
- Tests: 96 total (71 unit + 16 adapter + 9 integration) | Docs: `docs/archive/PHASE_2_FINAL_COMPLETE.md`

---

## [August 2025] - Enhanced Decision Making (Phase 1)

### Advanced Decision Algorithms ✅
- **AHP Engine**: Analytic Hierarchy Process with eigenvector method
- **Pareto Optimizer**: Multi-objective frontier analysis
- **Context Analyzer**: Dynamic weight adjustment (cognitive load, time pressure, risk)
- **ML Decision Model**: Learn from outcomes with decision trees
- **Feature Flags**: Gradual rollout with legacy fallback
- Tests: Comprehensive coverage | Docs: `docs/archive/PHASE_1_COMPLETION_SUMMARY.md`

---

## [August 2025] - Consolidation & Metacognition

### Memory Consolidation Pipeline ✅
- **STM→LTM Promotion**: Age and rehearsal gating
- **Provenance Tracking**: Source system identification
- **Metrics Counters**: Promotion tracking via metrics_registry
- **Threshold Configuration**: Dynamic consolidation criteria
- Docs: `docs/metacog_features.md`

### Metacognitive Self-Monitoring ✅
- **Adaptive Retrieval**: Load-based strategy adjustment
- **Performance Telemetry**: Executive function metrics
- **Attention Tracking**: Fatigue and capacity monitoring
- Docs: `docs/executive_telemetry.md`

---

## [July 2025] - STM & Attention Integration

### Production STM System ✅
- **VectorShortTermMemory**: ChromaDB-backed with sentence-transformers
- **7-Item Capacity**: Biological working memory limit
- **Activation/Decay**: Recency, frequency, salience weighting
- **4 Decay Modes**: Linear, exponential, power-law, sigmoid
- **LRU Eviction**: Least recently used item removal
- Docs: `docs/vector_stm_integration_complete.md`

### Attention Mechanism ✅
- **Fatigue Tracking**: 0-100 capacity with decay
- **Salience Scoring**: Relevance, novelty, emotional weighting
- **Capacity Limits**: Cognitive load thresholds
- Docs: `docs/stm_decay_adaptive_activation.md`

---

## [June 2025] - Enhanced LTM & Episodic Memory

### Long-Term Memory Enhancements ✅
- **Ebbinghaus Forgetting**: Exponential decay with half-life (30 days default)
- **Salience/Recency Weighting**: Dynamic retrieval scoring
- **Emotional Consolidation**: Emotion-based priority boosting
- **Importance Preservation**: High-importance memories resist decay
- **Consolidation Tracking**: STM promotion counters
- Docs: `docs/enhanced_ltm_summary.md`

### Episodic Memory System ✅
- **Rich Contextual Metadata**: Emotional valence, cognitive load, participants
- **Temporal Clustering**: Autobiographical life periods
- **Cross-System References**: Links to STM/LTM memories
- **Semantic Search**: ChromaDB vector search with fallback
- **Vividness & Confidence Tracking**: Memory quality metrics
- Tests: Comprehensive coverage with test isolation

---

## [June 2025] - Prospective Memory (Week 11)

### Time-Based Reminders ✅
- [Week 12 Completion Summary](docs/archive/WEEK_12_COMPLETION_SUMMARY.md)
- Tests: Comprehensive | Docs: `docs/archive/WEEK_11_COMPLETION_SUMMARY.md`

---

 [Week 15 Completion Summary](docs/archive/WEEK_15_COMPLETION_SUMMARY.md)

### Memory Foundation
- **BaseMemorySystem**: Abstract interface for all memory types
- **Memory Consolidation**: Initial STM→LTM transfer logic
 [Week 14 Completion Summary](docs/archive/WEEK_14_COMPLETION_SUMMARY.md)
- **Streamlit UI**: Interactive chat interface

---

 [Week 16 Phase 4 A/B Testing](docs/archive/WEEK_16_PHASE_4_AB_TESTING.md)

### Executive Settings
 [Week 16 Phase 3 Training Pipeline](docs/archive/WEEK_16_PHASE_3_TRAINING_PIPELINE.md)
- `MAX_ATTENTION_CAPACITY=100` - Attention limit
- `COGNITIVE_LOAD_THRESHOLD=0.75` - Overload detection
 [Week 16 Phase 2 Feature Extraction](docs/archive/WEEK_16_PHASE_2_FEATURE_EXTRACTION.md)

### November 2025
 [Week 16 Phase 1 Outcome Tracking](docs/archive/WEEK_16_PHASE_1_OUTCOME_TRACKING.md)
- `CPScheduler` requires `SchedulerConfig` for initialization

### September 2025
- GOAP `WorldState` is now immutable (frozen dataclass) - use `.set()` for updates
 [Phase 2 Final Complete](docs/archive/PHASE_2_FINAL_COMPLETE.md)
- **Learning System**: See `docs/archive/WEEK_16_PHASE_4_AB_TESTING.md` for experiment setup
---

For detailed documentation on each feature, see the `docs/` directory.
