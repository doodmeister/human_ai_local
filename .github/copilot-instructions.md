# Human-AI Cognition - Working Rules for AI Agents

Be concise. Keep changes small and consistent with existing patterns.

## Ground rules
- Python 3.12. New source in `src/`, tests in `tests/`.
- Shell is bash on Windows 11. No emojis in commands. Assume a venv is active.
- Lint with `ruff`; test with `pytest`. `pytest.ini` adds `src/` to `PYTHONPATH`.

## Where to plug in
- Chat pipeline: `src/chat/` (`ChatService`, `ContextBuilder`, `scoring.py`, `metrics_registry`).
- Memory: `src/memory/` (STM/LTM + prospective reminders in `prospective/prospective_memory.py`).
- Executive: `src/executive/` (decision, planning, scheduling, learning, integration).
- Attention: `src/attention/attention_mechanism.py`.
- Config: `src/core/config.py` (`get_chat_config().to_dict()` for `ContextBuilder`).
- API/UI: `python main.py api` (FastAPI), `python main.py chainlit` (Chainlit UI).

## Directory map
- `src/`: production code (all new modules go here).
- `tests/`: pytest tests.
- `docs/`: design notes and deep dives.
- `scripts/`: dev utilities, smoke tests, helpers.
- `data/`: runtime artifacts (models, outcomes, memory stores).
- `test_data/`: fixtures and test corpora.
- `temp_test_semantic_memory/`: local test scratch for semantic memory.
- `.github/`: repo automation and Copilot instructions.

## Critical workflows
```bash
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python main.py api
python main.py chainlit
pytest -q
```

## Key conventions
- DI-by-presence: `build_chat_service()` constructs subsystems when importable; `DISABLE_SEMANTIC_MEMORY=1` skips heavy semantic memory.
- Retrieval fallbacks: tag context items with `reason` containing `fallback`.
- Consolidation thresholds reset per turn; counters live in `src/chat/metrics.py` (`metrics_registry`).
- Prospective reminders: rank-0 items with `source_system="prospective"`. Use `ProspectiveMemorySystem` interface; legacy aliases still supported.
- Chroma config via `.env` (`CHROMA_PERSIST_DIR`, `STM_COLLECTION`, `LTM_COLLECTION`).

## Executive module cheatsheet
- Decision: `src/executive/decision` (feature flags; legacy `DecisionEngine` fallback).
- Planning (GOAP): `src/executive/planning` (`WorldState` immutable, use `.set()`, heuristics must be admissible).
- Scheduling: `src/executive/scheduling` (CP-SAT + dynamic scheduler).
- Integration: `src/executive/integration` (`ExecutiveSystem`, success criteria `"var=value"`).
- Learning: `src/executive/learning` (outcomes, features, models, experiments).

## When adding code
- New memory/retrieval providers must return `ContextItem`-compatible dicts and be safe to omit (lazy import, graceful fallback).
- Prefer lazy imports for heavy deps (`_lazy_import` in `src/chat/factory.py`).
- Emit lightweight counters via `metrics_registry` instead of verbose logs.

If a workflow or pattern is missing, call it out and we will refine this doc.