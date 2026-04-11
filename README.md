# Human-AI Cognition Framework

A production-grade cognitive architecture for AI systems with persistent memory, executive control, adaptive attention, and a metacognitive runtime.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

The repository combines:

- Memory systems for short-term, long-term, episodic, autobiographical, procedural, and prospective recall
- Executive systems for goal management, decision support, GOAP planning, scheduling, and learning from outcomes
- Attention and cognition layers for salience, fatigue, focus management, and self-model support
- A metacognitive loop that observes the agent's state, schedules follow-up cognition, persists traces, and exposes diagnostics through the API

The canonical runtime entrypoints are:

- `python main.py chat` for the interactive CLI
- `python main.py api` for FastAPI
- `python main.py chainlit` for the Chainlit UI
- `python main.py ui` for the legacy Streamlit UI

---

## Recent Changes

The past week landed several major runtime changes:

- A full `src/orchestration/metacognition/` subsystem with typed models, interfaces, controller, goal manager, policy engine, executor, critic, scheduler, event bus, presenters, scorecard, and filesystem-backed cycle tracing
- Background cognition support with persisted scheduled tasks, idle reflection, contradiction-audit scheduling, persisted reflection episodes, and persisted self-model snapshots
- New metacognition API surfaces for status, background activity, scorecards, dashboard views, last-cycle inspection, goals, tasks, self-model state, and reflection history
- Attention refactoring in `src/cognition/attention/` to separate lifecycle, state, models, and exceptions from the main mechanism implementation
- Memory lineage cleanup that standardizes `source_memory_ids` while remaining backward-compatible with older `source_event_ids` payloads
- Executive and learning improvements around decision history, feature handling, outcome-tracking fidelity, and validation-ready ML decision features
- Chainlit and Streamlit client updates, plus refreshed UI/API quickstart documentation

For the phased metacognition plan and completion status, see `cognition.md`.

---

## Key Capabilities

### Memory

- STM with decay, capacity limits, and adaptive utilization tracking
- Vector-backed LTM and retrieval scoring
- Episodic and autobiographical memory with promotion, provenance, and contradiction-aware normalization
- Procedural memory and recall feedback pathways
- Prospective memory with reminder management and optional vector-backed storage

### Executive Functions

- Hierarchical goals with dependencies, deadlines, and execution state
- Multiple decision strategies including weighted scoring, AHP, Pareto optimization, and ML-assisted scoring
- GOAP planning and CP-SAT scheduling
- Outcome tracking, feature extraction, experiment analysis, and model training hooks
- Executive telemetry for lifecycle events and metrics integration

### Attention And Cognition

- Attention lifecycle/state modeling and adaptive cognitive load handling
- Felt-sense, relational, narrative, self-model, and emergent-pattern cognitive layers
- Dream-state consolidation and cognitive break support

### Metacognition

- Per-turn snapshots and adaptive chat advisories under load or degraded performance
- A controller-driven cognition loop with explicit goals, candidate acts, execution results, critic reports, and self-model updates
- Background scheduling for follow-up cognition, contradiction audits, and idle reflections
- Persisted traces, reflections, tasks, and scorecards exposed through stable presenter-backed APIs

---

## Current Architecture

The canonical runtime path is:

- `main.py` for user-facing entrypoints
- `src/orchestration/runtime/bootstrap.py` for FastAPI app creation
- `src/orchestration/runtime/app_container.py` for shared runtime composition
- `src/orchestration/cognitive_agent.py` as the public cognitive facade
- `src/orchestration/chat/chat_service.py` for chat orchestration
- `src/orchestration/metacognition/` for the metacognitive loop and persistence
- `src/memory/memory_system.py` as the public memory facade
- `src/executive/integration.py` as the public executive orchestration surface

The active refactor and rollout notes live in `phase3.md` and `cognition.md`.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/human_ai_local.git
cd human_ai_local
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash; use venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### Start The System

Option 1: Chainlit UI with backend

```bash
python main.py chainlit --with-backend
```

Chainlit runs on `http://localhost:8501` and the API runs on `http://127.0.0.1:8000`.

Option 2: API only

```bash
python main.py api
```

API docs are available at `http://127.0.0.1:8000/docs`.

Option 3: Split terminals

```bash
python main.py api
python main.py chainlit
```

Option 4: Interactive CLI

```bash
python main.py chat
```

Option 5: Legacy Streamlit UI

```bash
python main.py ui
```

See `STARTUP_GUIDE.md` for startup troubleshooting and `docs/UI_DEVELOPER_API_QUICKSTART.md` for frontend integration details.

### API Compatibility

`python main.py api` is the canonical API startup path.

Notes:

- Prefer unprefixed routes such as `/agent/chat`, `/agent/metacog/*`, `/memory/...`, and `/executive/...`
- Legacy `/api/*` aliases are removed
- When pointing a UI at the backend, set `GEORGE_API_BASE_URL` to the server root, for example `http://127.0.0.1:8000`

For a quick endpoint smoke check, run:

```bash
python scripts/smoke_api_compat.py --base http://localhost:8000
```

### Memory Quality Gate

For the deterministic memory/personality regression scorecard, run:

```bash
python scripts/generate_memory_scorecard.py --fail-on-gate
```

This summarizes retrieval quality, longitudinal continuity, contradiction repair, false-memory resistance, and policy-behavior stability, and exits nonzero if any configured gate fails.

---

## API Reference

### Chat And Metacognition

```text
POST   /agent/chat                         # Send message, optionally stream, optionally override consolidation threshold
GET    /agent/chat/preview                 # Deterministic context preview
GET    /agent/chat/metrics                 # Chat metrics snapshot
GET    /agent/chat/performance             # Latency/performance status
GET    /agent/chat/metacog/status          # Last chat metacog snapshot
GET    /agent/chat/consolidation/status    # Consolidation health and recent events

GET    /agent/metacog/status               # Normalized metacognitive status
GET    /agent/metacog/background           # Background scheduler and reflection state
GET    /agent/metacog/scorecard            # Persisted metacognition scorecard
GET    /agent/metacog/dashboard            # Combined status/background/scorecard view
GET    /agent/metacog/last-cycle           # Last persisted metacognitive cycle
GET    /agent/metacog/goals                # Active metacognitive goals
GET    /agent/metacog/self-model           # Persisted self-model
GET    /agent/metacog/tasks                # Scheduled cognitive tasks
GET    /agent/metacog/reflections          # Reflection episode history
POST   /agent/metacog/reflect              # Trigger an immediate reflection report

GET    /agent/init-status                  # Agent initialization state
GET    /agent/status                       # Basic cognitive status snapshot
GET    /health                             # API health
GET    /health/detailed                    # Component-level health
GET    /telemetry                          # Resilience telemetry snapshot
GET    /circuit-breakers                   # Circuit breaker states
```

### Memory And Reminders

```text
POST   /memory/{system}/store              # Store memory in STM/LTM/Episodic
GET    /memory/{system}/retrieve/{id}      # Retrieve by ID
POST   /memory/{system}/search             # Search by query
GET    /status                             # Memory system status

POST   /agent/reminders                    # Create reminder
GET    /agent/reminders                    # List reminders
GET    /agent/reminders/due                # Retrieve due reminders
DELETE /agent/reminders/triggered          # Purge fired reminders
POST   /agent/reminders/{id}/complete      # Mark reminder complete
DELETE /agent/reminders/{id}               # Delete reminder
```

### Executive

```text
POST   /executive/goals                    # Create goal
GET    /executive/goals                    # List goals
GET    /executive/goals/{goal_id}          # Goal details
POST   /executive/goals/{goal_id}/execute  # Decision -> plan -> schedule pipeline
GET    /executive/goals/{goal_id}/status   # Execution context
GET    /executive/goals/{goal_id}/plan     # GOAP plan
GET    /executive/goals/{goal_id}/schedule # CP-SAT schedule
GET    /executive/status                   # Executive status
GET    /executive/system/health            # Extended health metrics
```

Full API docs: `http://127.0.0.1:8000/docs` when the server is running.

---

## Usage Examples

### Chat With Memory

```python
from src.orchestration.chat.factory import build_chat_service

service = build_chat_service()

service.process_user_message("My favorite color is blue", session_id="user1")
result = service.process_user_message("What's my favorite color?", session_id="user1")
print(result["response"])
```

### Execute A Goal

```python
import requests

response = requests.post(
    "http://localhost:8000/executive/goals",
    json={
        "title": "Analyze sales data",
        "description": "Generate Q4 insights",
        "priority": "HIGH",
        "success_criteria": ["data_analyzed=True", "report_created=True"],
    },
)
goal_id = response.json()["goal"]["id"]

execution = requests.post(f"http://localhost:8000/executive/goals/{goal_id}/execute")
context = execution.json()["execution_context"]

print(context["status"])
print(context["planning_time_ms"], context["scheduled_tasks"])
```

### Inspect Metacognition

```python
import requests

dashboard = requests.get("http://localhost:8000/agent/metacog/dashboard")
payload = dashboard.json()

print(payload["available"])
print(payload["background"]["scheduler_running"])
print(payload["scorecard"].get("summary", {}))
```

---

## Configuration

Create a `.env` file for provider and storage settings:

```bash
# LLM provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4

# Memory storage
CHROMA_PERSIST_DIR=./data/memory_stores
STM_COLLECTION=stm
LTM_COLLECTION=ltm
STM_CAPACITY=7

# Feature flags
GOAP_ENABLED=1
USE_VECTOR_PROSPECTIVE=0
DISABLE_SEMANTIC_MEMORY=1
```

For chat runtime configuration, `ContextBuilder` consumes `get_chat_config().to_dict()` from `src/core/config.py`.

---

## Testing

```bash
# Maintained default suite
python -m pytest -q

# Lint source and tests
python -m ruff check src tests

# Focused metacognition validation
python -m pytest tests/test_metacognition_api_endpoints.py -q
python -m pytest tests/test_metacognition_persistence.py -q
python -m pytest tests/test_metacognition_runtime_integration.py -q
python -m pytest tests/test_metacognition_scorecard.py -q

# Focused memory and executive validation
python -m pytest tests/test_enhanced_ltm_comprehensive.py -q
python -m pytest tests/test_episodic_memory_integration.py -q
python -m pytest tests/test_executive_decision_support.py -q
python -m pytest tests/test_outcome_tracking.py -q
```

The default `pytest -q` target is the maintained `tests/` suite via `pytest.ini`.

Legacy manual-only coverage remains opt-in under `archived_tests/manual_legacy/`.

---

## Documentation

Active docs:

- `cognition.md` - Metacognition architecture, phases, and rollout status
- `phase3.md` - Runtime refactor plan and architecture notes
- `docs/metacog_features.md` - Chat-layer metacognitive snapshot behavior
- `docs/executive_telemetry.md` - Executive event and metrics wiring
- `docs/memory_personality_architecture.md` - Memory/personality design
- `docs/memory_personality_roadmap.md` - Roadmap and sequencing
- `docs/memory_personality_implementation_tickets.md` - Implementation slices
- `docs/memory_personality_issue_backlog.md` - Remaining follow-up work
- `docs/goap_architecture.md` and `docs/goap_quick_reference.md` - Planning reference
- `docs/UI_DEVELOPER_API_QUICKSTART.md` - Frontend/API integration notes
- `docs/README.md` - Documentation index

Historical summaries remain under `docs/archive/`.

---

## Project Structure

```text
human_ai_local/
├── src/
│   ├── cognition/attention/         # Attention lifecycle, state, models
│   ├── orchestration/chat/          # Chat orchestration and context building
│   ├── orchestration/metacognition/ # Controller, scheduler, tracer, presenters
│   ├── memory/                      # STM, LTM, episodic, autobiographical, procedural, prospective
│   ├── executive/                   # Goals, decisions, planning, scheduling, learning
│   ├── interfaces/api/              # FastAPI routers
│   └── core/                        # Config and resilience
├── tests/                           # Maintained pytest suite
├── archived_tests/                  # Manual-only legacy coverage
├── scripts/                         # Utilities, smoke tests, UI entry modules
├── docs/                            # Active documentation and archive
└── data/                            # Persistent stores, traces, outcomes, models
```

---

## Status

Production-ready core areas:

- Memory systems including prospective and autobiographical flows
- Executive orchestration including decision, planning, scheduling, and learning
- Chat interface with adaptive retrieval, consolidation controls, and performance diagnostics
- Metacognition runtime through persistence, observability, and background cognition phases
- Chainlit-backed interactive UI and canonical FastAPI runtime

Still evolving:

- Richer frontend visualization for plans, schedules, and metacognitive state
- Additional monitoring and operator dashboards
- Continued memory/personality quality work and follow-up tickets from the active roadmap

---

## Built With

- ChromaDB
- sentence-transformers
- Google OR-Tools
- FastAPI
- Chainlit
- Streamlit
- scikit-learn
- scipy
