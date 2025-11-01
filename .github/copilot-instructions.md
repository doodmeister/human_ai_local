# Human-AI Cognition — Working Rules for AI Agents

Repo-specific guidance for being productive immediately. Keep changes small, concrete, and consistent with existing patterns.

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
  - Prospective: `prospective/prospective_memory.py` (in-memory reminders injected into chat context).
  - Consolidation: `consolidation/consolidator.py` (STM→LTM promotion with age/rehearsal gating, counters).
- Executive functions (`src/executive/`)
  - Legacy: `decision_engine.py` (weighted scoring), `task_planner.py` (template-based), `goal_manager.py` (hierarchical goals).
  - **Enhanced (Phase 1)**: `decision/` module with advanced algorithms:
    - `ahp_engine.py`: Analytic Hierarchy Process for multi-criteria decisions (eigenvector method, consistency checking).
    - `pareto_optimizer.py`: Multi-objective Pareto frontier analysis, trade-off visualization.
    - `context_analyzer.py`: Dynamic weight adjustment based on cognitive load, time pressure, risk tolerance.
    - `ml_decision_model.py`: Learn from decision outcomes using decision trees.
    - Feature flags (`get_feature_flags()`) enable gradual rollout; falls back to legacy on error.
  - **Enhanced (Phase 2)**: `planning/` module with GOAP (Goal-Oriented Action Planning):
    - `world_state.py`: Immutable state representation (`WorldState` frozen dataclass with key-value state).
    - `action_library.py`: Action definitions with preconditions, effects, costs (`Action` dataclass, `ActionLibrary`, 10 predefined actions).
    - `goap_planner.py`: A* search over state space (`GOAPPlanner.plan()` returns optimal `Plan` with action sequence).
    - `heuristics.py`: Admissible heuristics for A* (goal_distance, weighted_goal_distance, relaxed_plan, composite).
    - **Usage**: `planner = GOAPPlanner(action_library, heuristic='goal_distance'); plan = planner.plan(initial_state, goal_state, max_iterations=1000)`.
    - **Telemetry**: 10 metrics tracked via `metrics_registry` (planning attempts, plans found, plan length/cost, nodes expanded, latency).
    - **Testing**: 40 comprehensive tests in `tests/test_executive_goap_planner.py` (all passing, <10ms medium plans).
- Attention: `src/attention/attention_mechanism.py` (fatigue, capacity, metrics).
- Config: `src/core/config.py` (`get_chat_config().to_dict()` feeds `ContextBuilder`).
- API: `start_server.py` loads FastAPI app (`george_api_simple.py`) and mounts chat routes from `src/interfaces/api/chat_endpoints.py`.
- UI: `scripts/george_streamlit_chat.py` — minimal Streamlit chat client hitting `/agent/chat`.

## Critical workflows
```bash
# Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt

# Start API (http://localhost:8000)
python start_server.py

# Start UI (minimal Streamlit chat interface)
python start_george.py

# Run tests
pytest -q
```

## Conventions and patterns
- DI-by-presence: `build_chat_service()` constructs subsystems if importable; set `DISABLE_SEMANTIC_MEMORY=1` to skip heavy semantic memory.
- Retrieval fallbacks: when using degraded paths, tag produced context items with a `reason` containing `fallback`.
- Consolidation thresholds are temporary per turn and restored after; provenance/counters live in `src/chat/metrics.py` (`metrics_registry`).
- Prospective reminders appear as rank-0 items with `source_system="prospective"`; endpoints under `/agent/reminders*`.
- Use `get_chat_config()` instead of hard-coded defaults; pass `.to_dict()` into `ContextBuilder`.
- Chroma configuration via `.env` (see `.env.example`: `CHROMA_PERSIST_DIR`, `STM_COLLECTION`, `LTM_COLLECTION`).
- **Executive decisions**: Enhanced decision module uses feature flags for gradual rollout. Import from `src.executive.decision` for AHP/Pareto strategies; legacy `DecisionEngine` remains as fallback. All strategies implement `DecisionStrategy` interface.
- **GOAP planning**: Import from `src.executive.planning`. `WorldState` is immutable (frozen dataclass); use `.set()` to create new states. Actions define preconditions/effects as `WorldState` objects. Heuristics must be admissible (never overestimate). `GOAPPlanner.plan()` returns `Plan` with optimal action sequence or `None` if no solution. Telemetry automatically tracked via `metrics_registry`. See 10 predefined actions in `action_library.create_default_action_library()` (analyze_data, gather_data, create_document, etc.).

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
  - GOAP planning is in `src/executive/planning/`. Core components: `WorldState` (immutable state), `Action` (preconditions/effects), `ActionLibrary` (action repository), `GOAPPlanner` (A* search), `heuristics` (guidance functions).
  - See `docs/executive_refactoring_plan.md` for architecture and Phase 1-5 roadmap.

If a workflow or pattern you rely on is missing/unclear, say which part and we'll refine this doc.