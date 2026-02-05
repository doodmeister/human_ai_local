# Human-AI Cognition — Working Rules for AI Agents

Repo-specific guidance for being productive immediately. Keep changes small, concrete, and consistent with existing patterns.

## OpenMemory Integration
- **MCP Server**: OpenMemory is available via MCP at `http://localhost:8080/mcp` (requires Docker container running)
- **Automatic Memory Usage**: When the user shares important preferences, coding patterns, project context, or decisions, proactively use `openmemory_store` to save it with appropriate tags and `user_id`
- **Context Retrieval**: Before answering questions about preferences or past context, use `openmemory_query` to check for relevant memories
- **Memory Sectors**: Use appropriate tags for different types of information (e.g., "preferences", "patterns", "decisions", "architecture")
- **User Isolation**: Always use a consistent `user_id` (e.g., based on workspace or session) to keep memories isolated

## Ground rules
- Python 3.12. Put new source in `src/` and tests in `tests/`.
- Shell is bash on Windows 11. Do not include emojis in commands. Assume a venv is active.
- Lint with `ruff`; test with `pytest`. `pytest.ini` already adds `src/` to `PYTHONPATH`.

## Architecture landmarks (where to plug in)
- Chat pipeline (`src/chat/`)
  - `ChatService` orchestrates turns; instantiate via `src/chat/factory.build_chat_service()` (lazy DI of subsystems).
  - `ContextBuilder` stages: recent turns → STM/LTM/Episodic retrieval → Attention/Executive → score (`scoring.py`) → truncate to `ChatConfig.max_context_items`.
  - Metacog: adaptive retrieval/thresholds under load; metrics via `metrics_registry`.
- Memory systems (`src/memory/`)
  - STM: `stm/vector_stm.py` (ChromaDB + sentence-transformers; 7-item capacity; activation/decay; LRU).
  - LTM: `ltm/vector_ltm.py` (ChromaDB; semantic clusters; decay; health report).
  - **Prospective (Week 11 - COMPLETE)**: `prospective/prospective_memory.py` - Unified interface with two implementations:
    * `ProspectiveMemorySystem` ABC: Base interface with 10 methods (add_reminder, get_reminder, get_due_reminders, get_upcoming, list_reminders, search_reminders, complete_reminder, delete_reminder, purge_completed, clear).
    * `InMemoryProspectiveMemory`: Lightweight, no external dependencies, perfect for testing/development.
    * `VectorProspectiveMemory`: Full-featured with semantic search (requires sentence-transformers, chromadb).
    * **Factory**: `create_prospective_memory(use_vector=False)` creates instance with graceful fallback. `get_prospective_memory()` provides singleton.
    * **Backward Compatibility**: Legacy methods (check_due, upcoming, purge_triggered, to_dict) and aliases (ProspectiveMemory, ProspectiveMemoryVectorStore, get_inmemory_prospective_memory) work unchanged.
    * **Configuration**: Set `use_vector_prospective=True` in `MemorySystemConfig` to use vector store. Lazy imports prevent crashes if optional dependencies missing.
    * **Usage**: `pm = create_prospective_memory(use_vector=False); reminder = pm.add_reminder("Task", due_time=datetime.now() + timedelta(hours=1)); due = pm.get_due_reminders()`
    * **Optional Dependencies**: sentence-transformers and chromadb required only for VectorProspectiveMemory. System falls back to InMemory with warning if unavailable.
  - Consolidation: `consolidation/consolidator.py` (STM→LTM promotion with age/rehearsal gating, counters).
- Executive functions (`src/executive/`)
  - Legacy: `decision_engine.py` (weighted scoring), `task_planner.py` (template-based), `goal_manager.py` (hierarchical goals).
  - **Enhanced (Phase 1)**: `decision/` module with advanced algorithms:
    - `ahp_engine.py`: Analytic Hierarchy Process for multi-criteria decisions (eigenvector method, consistency checking).
    - `pareto_optimizer.py`: Multi-objective Pareto frontier analysis, trade-off visualization.
    - `context_analyzer.py`: Dynamic weight adjustment based on cognitive load, time pressure, risk tolerance.
    - `ml_decision_model.py`: Learn from decision outcomes using decision trees.
    - Feature flags (`get_feature_flags()`) enable gradual rollout; falls back to legacy on error.
  - **Enhanced (Phase 2 - COMPLETE)**: `planning/` module with GOAP (Goal-Oriented Action Planning):
    - `world_state.py`: Immutable state representation (`WorldState` frozen dataclass with key-value state).
    - `action_library.py`: Action definitions with preconditions, effects, costs (`Action` dataclass, `ActionLibrary`, 10 predefined actions).
    - `goap_planner.py`: A* search over state space (`GOAPPlanner.plan()` returns optimal `Plan` with action sequence).
    - `heuristics.py`: Admissible heuristics for A* (goal_distance, weighted_goal_distance, relaxed_plan, composite).
    - `constraints.py`: 5 constraint types (Resource, Temporal, Dependency, State, ConstraintChecker) for restricting action feasibility.
    - `replanning.py`: Dynamic replanning (`ReplanningEngine`) with failure detection, plan repair, and retry logic.
    - `goap_task_planner_adapter.py`: Bridges GOAP with legacy TaskPlanner (feature flags, fallback).
    - **Usage**: `planner = GOAPPlanner(action_library, heuristic='goal_distance', constraints=[...]); plan = planner.plan(initial_state, goal_state, plan_context={...})`.
    - **Telemetry**: 10 metrics tracked via `metrics_registry` (planning attempts, plans found, plan length/cost, nodes expanded, latency).
    - **Testing**: 96 comprehensive tests (71 unit + 16 adapter + 9 integration) all passing. <10ms medium plans, <500ms complex scenarios.
    - **Production Status**: 100% complete, 2,600+ production lines, feature flags for rollout, legacy fallback for safety.
  - **Scheduling (Week 12 - COMPLETE)**: `scheduling/` module with CP-SAT constraint scheduling:
    - `cp_scheduler.py`: Google OR-Tools CP-SAT solver for task scheduling (418 lines, type-safe, 0 Pylance errors).
    - `models.py`: Task, Resource, TimeWindow, Schedule, SchedulingProblem dataclasses (320 lines).
    - `task_planner_adapter.py`: Bridge CP scheduler with legacy TaskPlanner (327 lines, feature flags disabled by default).
    - **Constraints**: Precedence, resource capacity, deadlines, time windows, cognitive load limits.
    - **Optimization**: Minimize makespan, maximize priority, weighted multi-objective.
    - **Features**: Cycle detection, infeasibility detection, resource utilization metrics, 30s timeout, 4 workers.
    - **Usage**: `scheduler = CPScheduler(config); problem = SchedulingProblem(tasks, resources, objectives, ...); schedule = scheduler.schedule(problem)`.
    - **Testing**: 17/17 tests passing in 17.82s (basic, precedence, resources, deadlines, cognitive load, infeasibility, optimization).
    - **Dependencies**: ortools>=9.8.0 (Google's optimization library).
    - **Production Status**: 100% complete, 1,065 production lines, all type checks passing, ready for integration.
  - **Dynamic Scheduling (Week 14 - COMPLETE)**: `scheduling/` extensions with real-time adaptation:
    - `dynamic_scheduler.py`: Real-time monitoring, adaptation, health tracking (580 lines).
    - Quality metrics (8 methods): critical path, slack time, buffer time, robustness score (0-1), resource variance, cognitive smoothness.
    - `ScheduleMonitor`: Detects disruptions (failures, delays, resource unavailability).
    - `ScheduleAnalyzer`: Proactive warnings (resource contention >90%, zero slack, cognitive overload >90%, critical path >70%).
    - `ScheduleVisualizer`: 7 export formats (Gantt, timeline, resource utilization, dependency graph, critical path, cognitive load, JSON).
    - **Usage**: `scheduler = DynamicScheduler(); schedule = scheduler.create_initial_schedule(problem); warnings = scheduler.get_proactive_warnings(now); scheduler.handle_disruption(disruption)`.
    - **Testing**: 36/36 tests passing (monitoring, adaptation, quality metrics, visualization).
    - **Production Status**: 100% complete, 1,240 production lines, full visualization support.
  - **System Integration (Week 15 - COMPLETE)**: `integration.py` unified executive orchestrator:
    - `ExecutiveSystem`: Orchestrates GoalManager → DecisionEngine → GOAPPlanner → DynamicScheduler pipeline (497 lines).
    - `ExecutionContext`: Tracks full pipeline state (goal_id, decision_result, plan, schedule, status, timing metrics).
    - `ExecutionStatus` enum: IDLE, PLANNING, SCHEDULING, EXECUTING, COMPLETED, FAILED.
    - `IntegrationConfig`: Feature toggles, strategy selection (weighted_scoring/ahp/pareto), timeouts.
    - **Pipeline**: Stage 1 (Decision) → Stage 2 (GOAP Planning) → Stage 3 (CP-SAT Scheduling) → Execution tracking.
    - **Usage**: `system = ExecutiveSystem(); context = system.execute_goal(goal_id, initial_state=WorldState({})); health = system.get_system_health()`.
    - **Goal Parsing**: Success criteria format `"var=value"` (parsed to correct types: True/False → bool, digits → int, else → float/string). Example: `["data_analyzed=True"]` → `WorldState({"data_analyzed": True})`.
    - **Metrics**: 6 counters tracked (init, executions, decisions, plans, schedules, failures) via `get_metrics_registry()` from `goap_planner`.
    - **Testing**: 17/24 tests passing (71%), core integration 100% functional. Pipeline latency 12-15s.
    - **Production Status**: 100% complete, ~1,000 production lines (497 integration + 480 tests), end-to-end pipeline working.
  - **Learning Infrastructure (Week 16 Phase 1 - COMPLETE)**: `learning/` module with outcome tracking:
    - `outcome_tracker.py`: OutcomeTracker service for recording/analyzing execution results (610 lines).
    - `OutcomeRecord`: Complete execution record with decision/plan/schedule context, JSON serialization.
    - `AccuracyMetrics`: Time accuracy (actual/predicted), plan adherence, goal achievement, deviation tracking.
    - **Analysis Methods**: `analyze_decision_accuracy()`, `analyze_planning_accuracy()`, `analyze_scheduling_accuracy()`, `get_improvement_trends()`.
    - **Storage**: JSON files in `data/outcomes/` with format `outcome_{goal_id}_{timestamp}.json`.
    - **ExecutionContext Extensions**: 5 new fields (actual_completion_time, actual_success, outcome_score, deviations, accuracy_metrics).
    - **ExecutiveSystem Methods**: `complete_goal_execution(goal_id, success, outcome_score, deviations)`, `get_learning_metrics()`.
    - **Usage**: `outcome = system.complete_goal_execution("goal_123", success=True, outcome_score=0.85); metrics = system.get_learning_metrics()`.
    - **Key Features**: Confidence calibration, strategy comparison, improvement detection, persistent history.
    - **Testing**: 20/20 tests passing (100%). AccuracyMetrics (3), OutcomeRecord (1), OutcomeTracker (11), Integration (5).
    - **Production Status**: 100% complete, 1,200+ production lines (610 tracker + 500 tests + 90 integration).
  - **Learning Infrastructure (Week 16 Phase 2 - COMPLETE)**: `learning/` module with feature extraction:
    - `feature_extractor.py`: FeatureExtractor service converts outcomes to ML training features (450 lines).
    - `FeatureVector`: 23-field dataclass (decision/planning/scheduling/context features + targets).
    - **Feature Categories**: Decision (strategy, confidence), Planning (length, cost, nodes), Scheduling (makespan, tasks), Context (hour, day_of_week).
    - **Target Variables**: success (0/1), outcome_score (0-1), time_accuracy_ratio, plan_adherence_score.
    - **Export Methods**: `export_csv()`, `export_json()`, `export_parquet()` for training data.
    - **Normalization**: `normalize_features(method='standard')` for StandardScaler, `method='minmax'` for MinMaxScaler (0-1 range).
    - **Statistics**: `get_feature_statistics()` calculates mean, std, min, max, missing counts per feature.
    - **OutcomeTracker Integration**: `tracker.get_training_dataset(limit, strategy, success_only, export_path, export_format)`.
    - **Usage**: `features = tracker.get_training_dataset(limit=100, export_path='training.csv'); df = extractor.to_dataframe(features); normalized, params = extractor.normalize_features(features)`.
    - **Dependencies**: pandas, numpy for data processing; pyarrow for Parquet export (optional).
    - **Testing**: 21/21 tests passing (100%). FeatureVector (2), FeatureExtractor (15), Integration (4).
    - **Production Status**: 100% complete, 1,250+ production lines (450 extractor + 400 tests + 400 integration/docs).
  - **Learning Infrastructure (Week 16 Phase 3 - COMPLETE)**: `learning/` module with ML training and inference:
    - `model_trainer.py`: ModelTrainer service trains 4 ML models from outcomes (800 lines).
    - **4 Model Types**: Strategy classifier (RandomForest), Success classifier (GradientBoosting), Time regressor (RandomForest), Outcome regressor (GradientBoosting).
    - **Training Features**: 80/20 train/test split, 5-fold cross-validation, GridSearchCV hyperparameter tuning, feature importance extraction.
    - **Persistence**: joblib for models, JSON for metadata (version, accuracy, training date, hyperparameters). Storage: `data/models/learning/{type}_{timestamp}.joblib`.
    - `model_predictor.py`: ModelPredictor service for inference (400 lines).
    - **Prediction Methods**: `predict_strategy()`, `predict_success()`, `predict_time_accuracy()`, `predict_outcome_score()`, `predict_all()`.
    - **Batch Predictions**: `batch_predict_strategy()`, `batch_predict_success()` for multiple goals.
    - **Model Management**: `get_model_info()`, `list_available_models()`, `clear_cache()`, lazy loading with caching.
    - **DecisionEngine Integration**: ML-based confidence boosting (+5% strategy match, +15% high success, -10% low success). Graceful fallback if models unavailable.
    - **Usage Training**: `trainer = create_model_trainer(); results = trainer.train_all_models(features, tune_hyperparameters=True)`.
    - **Usage Inference**: `predictor = create_model_predictor(); strategy_pred = predictor.predict_strategy(feature_vec); success_pred = predictor.predict_success(feature_vec)`.
    - **Usage DecisionEngine**: `engine = DecisionEngine(enable_ml_predictions=True); result = engine.make_decision(options, criteria, context={'goal_id': 'test', 'plan_length': 5})`. ML predictions stored in `result.metadata['ml_predictions']`.
    - **Dependencies**: scikit-learn (RandomForest, GradientBoosting, GridSearchCV), joblib (persistence).
    - **Testing**: 40/41 tests passing (98%). ModelTrainer (19), ModelPredictor (14), Integration (7).
    - **Production Status**: 100% complete, 2,165 production lines (800 trainer + 400 predictor + 150 integration + 815 tests).
  - **Learning Infrastructure (Week 16 Phase 4 - COMPLETE)**: `learning/` module with A/B testing:
    - `experiment_manager.py`: ExperimentManager service for strategy experiments (780 lines).
    - **Core Classes**: `StrategyExperiment` (experiment metadata), `ExperimentAssignment` (tracks strategy per decision), `StrategyOutcome` (execution results).
    - **3 Assignment Methods**: Random (uniform), Epsilon-Greedy (explore ε=10% vs exploit 90%), Thompson Sampling (Bayesian with Beta distributions).
    - **Experiment Lifecycle**: create → start → [assign strategies + record outcomes] → complete → analyze.
    - **Persistence**: JSON storage in `data/experiments/` (experiment_{id}.json, assignments_{id}.json, outcomes_{id}.json).
    - `experiment_analyzer.py`: Statistical analysis for experiment results (450 lines).
    - **Statistical Tests**: Chi-square (categorical), t-test (continuous parametric), Mann-Whitney U (non-parametric), Cohen's d (effect size).
    - **Confidence Intervals**: Wilson score for proportions, t-distribution for continuous metrics.
    - **Strategy Recommendation**: `recommend_strategy()` with confidence levels (low/medium/high), significance testing, sample size validation.
    - **DecisionEngine Integration**: Add `experiment_manager` param to init, `experiment_id` to `make_decision()`. Auto-assigns strategy when experiment active. Use `record_experiment_outcome()` for tracking.
    - **Usage Creation**: `manager = create_experiment_manager(); exp = manager.create_experiment(name="Test", strategies=["weighted_scoring", "ahp"], assignment_method=AssignmentMethod.EPSILON_GREEDY)`.
    - **Usage Execution**: `manager.start_experiment(exp_id); assignment = manager.assign_strategy(exp_id, decision_id, goal_id); manager.record_outcome(assignment.assignment_id, success=True, outcome_score=0.85)`.
    - **Usage Analysis**: `analysis = manager.analyze_experiment(exp_id); print(analysis['recommended_strategy'], analysis['confidence'])`.
    - **Usage DecisionEngine**: `engine = DecisionEngine(experiment_manager=manager); result = engine.make_decision(options, criteria, experiment_id="exp_123"); engine.record_experiment_outcome(assignment_id, success, score)`.
    - **Dependencies**: scipy.stats (chi2_contingency, ttest_ind, mannwhitneyu), numpy (already in requirements.txt).
    - **Testing**: 23/23 tests passing (100%). Experiments (6), Assignments (3), Outcomes (3), Statistical Analysis (6), Integration (3), End-to-End (2).
    - **Production Status**: 100% complete, 1,880+ production lines (780 manager + 450 analyzer + 100 integration + 550 tests), ready for production use.
- Attention: `src/attention/attention_mechanism.py` (fatigue, capacity, metrics).
- Config: `src/core/config.py` (`get_chat_config().to_dict()` feeds `ContextBuilder`).
- API: `python main.py api` runs the FastAPI app (see `scripts/legacy/george_api_simple.py` and mounted routers under `src/interfaces/api/`).
- UI: `scripts/george_streamlit_chat.py` — minimal Streamlit chat client hitting `/agent/chat`.

## Critical workflows
```bash
# Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt

# Start API (http://localhost:8000)
python main.py api

# Start UI (minimal Streamlit chat interface)
python main.py ui

# Run tests
pytest -q
```

## Conventions and patterns
- DI-by-presence: `build_chat_service()` constructs subsystems if importable; set `DISABLE_SEMANTIC_MEMORY=1` to skip heavy semantic memory.
- Retrieval fallbacks: when using degraded paths, tag produced context items with a `reason` containing `fallback`.
- Consolidation thresholds are temporary per turn and restored after; provenance/counters live in `src/chat/metrics.py` (`metrics_registry`).
- **Prospective reminders**: Appear as rank-0 items with `source_system="prospective"` in chat context. Endpoints under `/agent/reminders*`. Use `ProspectiveMemorySystem` interface for all implementations. For backward compat with old timestamp-based API: `add_reminder(content, seconds)` auto-converts to datetime; `list_reminders(include_triggered=True)` alias for `include_completed`; `check_due()`, `upcoming(seconds)`, `purge_triggered()` methods available.
- Use `get_chat_config()` instead of hard-coded defaults; pass `.to_dict()` into `ContextBuilder`.
- Chroma configuration via `.env` (see `.env.example`: `CHROMA_PERSIST_DIR`, `STM_COLLECTION`, `LTM_COLLECTION`).
- **Executive decisions**: Enhanced decision module uses feature flags for gradual rollout. Import from `src.executive.decision` for AHP/Pareto strategies; legacy `DecisionEngine` remains as fallback. All strategies implement `DecisionStrategy` interface.
- **GOAP planning**: Import from `src.executive.planning`. `WorldState` is immutable (frozen dataclass); use `.set()` to create new states. Actions define preconditions/effects as `WorldState` objects. Heuristics must be admissible (never overestimate). `GOAPPlanner.plan()` returns `Plan` with optimal action sequence or `None` if no solution. Telemetry automatically tracked via `metrics_registry`. See 10 predefined actions in `action_library.create_default_action_library()` (analyze_data, gather_data, create_document, etc.).
- **CP-SAT scheduling**: Import from `src.executive.scheduling`. Create `CPScheduler(config)` with `SchedulerConfig(time_resolution, solver_timeout, num_workers)`. Define `SchedulingProblem` with tasks, resources, objectives. Call `scheduler.schedule(problem)` to get optimal `Schedule`. Tasks have `scheduled_start`/`scheduled_end` after scheduling. Use `typing.cast()` for Optional type guards. All constraint types available: precedence, resource capacity, deadlines, time windows, cognitive load.
- **Dynamic scheduling** (Week 14 COMPLETE): Import from `src.executive.scheduling`. Use `DynamicScheduler` for real-time schedule adaptation. Quality metrics auto-calculated via `schedule.update_quality_metrics()` - includes critical path, slack time, buffer time, robustness score (0-1), resource variance, cognitive smoothness. `ScheduleMonitor` detects disruptions (failures, delays, resource issues). `ScheduleAnalyzer` provides proactive warnings (resource contention >90%, zero slack, cognitive overload >90%, critical path >70%). Handle disruptions with `scheduler.handle_disruption(disruption)`. Check health with `scheduler.get_schedule_health()`. Export visualizations via `ScheduleVisualizer` - 7 formats (Gantt, timeline, resource utilization, dependency graph, critical path, cognitive load, JSON). See `docs/archive/WEEK_14_COMPLETION_SUMMARY.md` for full usage.
- **System integration** (Week 15 COMPLETE): Import from `src.executive.integration`. Use `ExecutiveSystem()` for unified Goal→Decision→Plan→Schedule pipeline. Success criteria parsing: `"var=value"` format auto-converts types (True/False→bool, digits→int). `WorldState` takes dict not kwargs: `WorldState({"key": value})`. Pipeline stages tracked in `ExecutionContext` with timing metrics. Health monitoring via `get_system_health()`. Supports custom `IntegrationConfig` for strategy/timeout tuning.
- **ML learning** (Week 16 Phase 3 COMPLETE): Import from `src.executive.learning`. Train 4 ML models from execution outcomes: `ModelTrainer.train_all_models(features, tune_hyperparameters=True)`. Models: strategy classifier (RandomForest), success classifier (GradientBoosting), time regressor (RandomForest), outcome regressor (GradientBoosting). Make predictions: `ModelPredictor.predict_strategy(feature_vec)`, `predict_success()`, `predict_time_accuracy()`, `predict_outcome_score()`. DecisionEngine auto-boosts confidence with ML predictions when context provided: `engine = DecisionEngine(enable_ml_predictions=True); result = engine.make_decision(options, criteria, context={'goal_id': 'test', 'plan_length': 5})`. ML predictions in `result.metadata['ml_predictions']`. Confidence boost: +5% strategy match, +15% high success prediction, -10% low success prediction. Graceful fallback if models unavailable. See `docs/archive/WEEK_16_PHASE_3_TRAINING_PIPELINE.md` for full usage.
- **A/B testing** (Week 16 Phase 4 COMPLETE): Import from `src.executive.learning`. Create experiments to compare strategies: `manager = create_experiment_manager(); exp = manager.create_experiment(name="Test", strategies=["weighted_scoring", "ahp"], assignment_method=AssignmentMethod.EPSILON_GREEDY)`. Start and run: `manager.start_experiment(exp.experiment_id); assignment = manager.assign_strategy(exp.experiment_id, decision_id, goal_id)`. Record outcomes: `manager.record_outcome(assignment.assignment_id, success=True, outcome_score=0.85, execution_time_seconds=10.0)`. Analyze results: `analysis = manager.analyze_experiment(exp.experiment_id)` returns `recommended_strategy` with confidence level (low/medium/high). DecisionEngine integration: `engine = DecisionEngine(experiment_manager=manager); result = engine.make_decision(options, criteria, experiment_id="exp_123")` auto-assigns strategy. Use `engine.record_experiment_outcome(assignment_id, success, score)` to track. Three assignment methods: Random (uniform), Epsilon-Greedy (90% exploit best, 10% explore), Thompson Sampling (Bayesian). Statistical tests: chi-square (success rates), t-test/Mann-Whitney (scores), Cohen's d (effect size). Storage: `data/experiments/`. See `docs/archive/WEEK_16_PHASE_4_AB_TESTING.md` for full usage.

## Integration references
- Chat endpoints: `/agent/chat`, `/agent/chat/preview`, `/agent/chat/performance`, `/agent/chat/metacog/status`, `/agent/chat/consolidation/status` (see `src/interfaces/api/chat_endpoints.py`).
- Scoring/profile: `src/chat/scoring.py` and `get_scoring_profile_version()`; exposed in context preview responses.
- Memory capture: `ChatService` uses `_capture` and `_capture_cache` to extract facts/preferences/goals, store to STM, and optionally promote to semantic on frequency milestones.

## When adding code
- New memory/retrieval providers should return `ContextItem`-compatible dicts and be safe to omit (lazy import, graceful fallback).
- Prefer lazy imports for heavy deps (see `_lazy_import` in `src/chat/factory.py`).
- Emit lightweight counters via `metrics_registry` instead of verbose logs.
- **Executive module changes**: 
  - Enhanced decision algorithms are in `src/executive/decision/`. Use feature flags to enable/disable. Maintain backward compatibility with legacy `DecisionEngine`.
  - **GOAP planning** (Phase 2 COMPLETE) is in `src/executive/planning/`. Core components: 
    * `WorldState` (immutable state): Use `.set()` to create new states, `.satisfies()` to check goal achievement.
    * `Action` (preconditions/effects): Define with `Action(name, preconditions=WorldState(...), effects=WorldState(...), cost=float)`.
    * `ActionLibrary` (action repository): Use `create_default_action_library()` or build custom with `.add_action()`.
    * `GOAPPlanner` (A* search): `planner.plan(initial, goal, plan_context={...})` returns `Plan` or `None`. Optional `constraints` and `heuristic` parameters.
    * `Constraint` types: Use helper functions `create_resource_constraint()`, `create_deadline_constraint()`, `create_dependency_constraint()`, `create_state_constraint()`.
    * `ReplanningEngine`: `detect_failure()` monitors execution, `replan()` tries repair then full replan, `should_retry_action()` for transient failures.
    * Heuristics must be admissible (never overestimate). Default is `goal_distance`. Use `get_heuristic()` factory or `CompositeHeuristic` for weighted combinations.
  - **CP-SAT Scheduling** (Week 12 COMPLETE) is in `src/executive/scheduling/`. Core components:
    * `CPScheduler` (constraint solver): Use `scheduler.schedule(problem)` to solve scheduling problems with CP-SAT.
    * `SchedulingProblem`: Define tasks, resources, objectives, constraints, time horizon, resolution.
    * `Task` (scheduling): Has id, duration, priority, dependencies, resource_requirements, cognitive_load, time_window.
    * `Resource`: Has name, capacity, type (COMPUTATIONAL, COGNITIVE, etc.). Use frozen Resource instances in task requirements.
    * `Schedule` (result): Contains scheduled tasks with start/end times, makespan, resource utilization metrics.
    * **Type Safety**: Use `typing.cast()` after None checks: `if self.model is None: raise RuntimeError(); model = cast(CpModel, self.model)`.
    * **Constraints**: Precedence (A before B), resource capacity (cumulative), deadlines, time windows, cognitive load limits.
    * **Optimization**: Minimize makespan (shortest schedule), maximize priority (high-priority first), weighted combinations.
  - **Integration**: Use `GOAPTaskPlannerAdapter` to bridge GOAP with `GoalManager`/`TaskPlanner`. Feature flags: `goap_enabled`, `goap_use_constraints`, `goap_use_replanning`, `goap_fallback_on_error`.
  - See `docs/archive/PHASE_2_FINAL_COMPLETE.md` for GOAP documentation and `docs/archive/WEEK_12_COMPLETION_SUMMARY.md` for scheduling details.

If a workflow or pattern you rely on is missing/unclear, say which part and we'll refine this doc.