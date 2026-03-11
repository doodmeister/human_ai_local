# Human-AI Cognition Framework

A production-grade cognitive architecture for AI systems with human-like memory, reasoning, and executive control.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Complete cognitive framework with:
- **Memory Systems**: Short-term (STM), long-term (LTM), episodic, and prospective memory
- **Executive Functions**: Goal management, decision-making, GOAP planning, constraint scheduling
- **Learning**: ML-powered outcome tracking, A/B testing, continuous improvement
- **Attention**: Adaptive mechanisms, cognitive load tracking, metacognition

---

## ✨ Key Capabilities

### Memory
- **STM**: 7-item capacity, activation decay, LRU eviction
- **LTM**: Vector-based semantic memory with forgetting curves
- **Episodic**: Contextual memories with emotional valence
- **Prospective**: Time-based and event-based reminders

### Executive Functions
- **Goals**: Hierarchical goals with dependencies and deadlines
- **Decisions**: Weighted scoring, AHP, Pareto optimization with ML predictions
- **Planning**: GOAP (A* search) with 10 predefined actions
- **Scheduling**: Google OR-Tools CP-SAT constraint solver
- **Learning**: Outcome tracking, A/B testing, 4 ML models

### Phase 2 Cognitive Layers (Implemented)
- **Layer 0: Drives** with implicit learning and internal conflict detection
- **Layer 1: Felt Sense** with mood derivation
- **Layer 2: Relational Field** for relationship-aware cognition
- **Layer 3: Emergent Patterns** including Big Five descriptions
- **Layer 4: Self-Model** with blind spots and self-discovery
- **Layer 5: Narrative** for identity synthesis and context injection

### API Endpoints
- **Chat**: `/agent/chat` - Conversational interface with memory
- **Memory**: `/agent/memory/*` - STM/LTM storage and retrieval
- **Reminders**: `/agent/reminders/*` - Prospective memory management
- **Executive**: `/executive/*` - Goal management and execution pipeline

## Current Architecture

The canonical runtime path is now:

- `main.py` for user-facing entrypoints
- `src/orchestration/runtime/bootstrap.py` for FastAPI app creation
- `src/orchestration/runtime/app_container.py` for shared runtime composition
- `src/orchestration/cognitive_agent.py` as the public cognitive facade
- `src/orchestration/chat/chat_service.py` as the public chat facade
- `src/memory/memory_system.py` as the public memory facade
- `src/executive/integration.py` as the public executive orchestration surface

The current architectural refactor plan and status are tracked in `phase3.md`.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/human_ai_local.git
cd human_ai_local
python -m venv venv
source venv/Scripts/activate  # Windows (use venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

### Start the System

**Option 1: Chainlit Chat UI** (recommended)
```bash
# Start the backend API first
python main.py api

# In a second terminal, start the Chainlit UI
python main.py chainlit
```
Access at http://localhost:8501

Or start both together:
```bash
python main.py chainlit --with-backend
```

**Option 2: API Server only**
```bash
python main.py api
```
Access at http://127.0.0.1:8000 (docs at /docs)

**Option 3: Streamlit UI** (legacy)
```bash
python main.py ui
```
Access at http://localhost:8501

### API compatibility

`python main.py api` is the canonical API startup path.

Notes:

- Prefer unprefixed routes such as `/agent/chat`, `/memory/...`, and `/executive/...`.
- Legacy `/api/*` aliases are removed.
- If you're pointing a UI at the backend, set `GEORGE_API_BASE_URL` to the server root, for example `http://127.0.0.1:8000`.

For a quick endpoint smoke check, run:
```bash
python scripts/smoke_api_compat.py --base http://localhost:8000
```

---

## 📡 API Reference

### Chat Endpoints
```
POST   /agent/chat                    # Send message, get response
GET    /agent/chat/preview            # Preview context items
GET    /agent/chat/performance        # Performance metrics
GET    /agent/chat/metacog/status     # Metacognition status
```

### Memory Endpoints
```
POST   /memory/{system}/store         # Store memory in STM/LTM/Episodic
GET    /memory/{system}/retrieve/{id} # Retrieve by ID
POST   /memory/{system}/search        # Search by query
GET    /status                        # Memory system status
```

### Reminder Endpoints (Prospective Memory)
```
POST   /agent/reminders               # Create reminder
GET    /agent/reminders               # List all reminders
GET    /agent/reminders/due           # Get due reminders
DELETE /agent/reminders/{id}          # Delete reminder
POST   /agent/reminders/{id}/complete # Mark complete
```

### Executive Endpoints (NEW)
```
# Goal Management
POST   /executive/goals                      # Create goal
GET    /executive/goals                      # List goals
GET    /executive/goals/{goal_id}            # Get goal details

# Execution Pipeline
POST   /executive/goals/{goal_id}/execute    # Run Decision→Plan→Schedule pipeline
GET    /executive/goals/{goal_id}/status     # Get execution context
GET    /executive/goals/{goal_id}/plan       # Get GOAP plan (action sequence)
GET    /executive/goals/{goal_id}/schedule   # Get CP-SAT schedule (Gantt data)

# System Monitoring
GET    /executive/status                     # Basic status
GET    /executive/system/health              # Full health metrics
```

**Full API docs**: http://localhost:8000/docs (when server running)

---

## 💡 Usage Examples

### Example 1: Chat with Memory
```python
from src.orchestration.chat.factory import build_chat_service

service = build_chat_service()

result = service.process_user_message("My favorite color is blue", session_id="user1")
response = service.process_user_message("What's my favorite color?", session_id="user1")
```

### Example 2: Execute a Goal
```python
import requests

# Create goal
response = requests.post("http://localhost:8000/executive/goals", json={
    "title": "Analyze sales data",
    "description": "Generate Q4 insights",
    "priority": "HIGH",
    "success_criteria": ["data_analyzed=True", "report_created=True"]
})
goal_id = response.json()["goal"]["id"]

# Execute integrated pipeline (Decision → Plan → Schedule)
response = requests.post(f"http://localhost:8000/executive/goals/{goal_id}/execute")
context = response.json()["execution_context"]

print(f"Status: {context['status']}")
print(f"Actions: {context['total_actions']}, Tasks: {context['scheduled_tasks']}")
print(f"Planning: {context['planning_time_ms']}ms, Scheduling: {context['scheduling_time_ms']}ms")
```

### Example 3: View GOAP Plan
```python
# Get plan from execution context
response = requests.get(f"http://localhost:8000/executive/goals/{goal_id}/plan")
plan = response.json()["plan"]

for i, step in enumerate(plan["steps"], 1):
    print(f"Step {i}: {step['name']}")
    print(f"  Preconditions: {step['preconditions']}")
    print(f"  Effects: {step['effects']}")
    print(f"  Cost: {step['cost']}")
```

---

## ⚙️ Configuration

Create `.env` file (see `.env.example`):

```bash
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4

# Memory Storage
CHROMA_PERSIST_DIR=./data/memory_stores
STM_CAPACITY=7

# Cognitive Limits
MAX_ATTENTION_CAPACITY=100
COGNITIVE_LOAD_THRESHOLD=0.75

# Feature Flags
GOAP_ENABLED=1
USE_VECTOR_PROSPECTIVE=0  # Set to 1 for semantic reminders
```

---

## 🧪 Testing

```bash
# Default fast contract lane
python -m pytest -q

# Explicit contract lane
python -m pytest tests/contracts -q

# Non-default broader suite
python -m pytest tests -q -c /dev/null

# Lint source and tests
python -m ruff check src tests


The default `pytest -q` target is intentionally scoped to `tests/contracts` via `pytest.ini`.
# Non-default persistence coverage
pytest tests/persistence -q

# Focused specialized memory validation
pytest tests/test_enhanced_ltm_comprehensive.py -q
pytest tests/test_episodic_memory_integration.py -q

# Archived manual-only legacy tests, only when intentionally validating older paths
pytest archived_tests/manual_legacy/integration -q
pytest archived_tests/manual_legacy/scenarios -q
```

Current test tiers:

- `tests/contracts/`: default fast suite for core contracts. This is what `pytest -q` runs.
- `tests/smoke/`: broader orchestration checks that should stay out of the default fast loop.
- `tests/persistence/`: storage and reload behavior checks.
- `tests/test_enhanced_ltm_comprehensive.py` and `tests/test_episodic_memory_integration.py`: focused specialized memory validation outside the default loop.
- `archived_tests/manual_legacy/`: historical manual-only suites retained for migration or archaeology, but removed from the active `tests/` tree.

The default suite is intentionally small and fast. It covers intent classification, context assembly, chat service behavior, memory routing, and canonical API contracts without pulling in the full legacy test surface.

---

## 📖 Documentation

### Key Documentation
- **System Integration**: `docs/archive/WEEK_15_COMPLETION_SUMMARY.md`
- **GOAP Planning**: `docs/archive/PHASE_2_FINAL_COMPLETE.md`
- **Scheduling**: `docs/archive/WEEK_12_COMPLETION_SUMMARY.md`, `docs/archive/WEEK_14_COMPLETION_SUMMARY.md`
- **Learning & A/B Testing**: `docs/archive/WEEK_16_PHASE_4_AB_TESTING.md`
- **Memory Systems**: `docs/archive/enhanced_ltm_summary.md`, `docs/archive/vector_stm_integration_complete.md`
- **Executive API**: `docs/archive/BACKEND_API_COMPLETION_SUMMARY.md`
- **UI Development**: `docs/UI_DEVELOPER_API_QUICKSTART.md`

### Quick References
- **AI Instructions**: `.github/copilot-instructions.md` - Development patterns
- **Roadmap**: `docs/archive/planning/roadmap.md` - Future development plans

---

## 🛠️ Project Structure

```
human_ai_local/
├── src/
│   ├── chat/              # Chat service, context building
│   ├── memory/            # STM, LTM, episodic, prospective
│   ├── executive/         # Goals, decisions, planning, scheduling, learning
│   ├── attention/         # Attention mechanism
│   ├── interfaces/        # API endpoints
│   └── core/              # Configuration
├── tests/                 # Active contract, smoke, persistence, and focused specialized tests
├── archived_tests/        # Historical manual-only legacy tests
├── scripts/
│   ├── chainlit_app/      # Chainlit chat UI (recommended)
│   └── george_streamlit_chat.py  # Legacy Streamlit UI
├── docs/                  # Documentation
└── data/                  # Persistent storage
```

---

## 📊 Performance Benchmarks

- **Chat Response**: <1s typical, <3s with full retrieval
- **Memory Retrieval**: <100ms STM, <200ms LTM
- **GOAP Planning**: <50ms simple, <500ms complex
- **CP-SAT Scheduling**: <30s for 50 tasks
- **Full Pipeline**: 12-15s end-to-end

---

## 🗺️ Status

### Production Ready ✅
- Memory systems (STM, LTM, episodic, prospective)
- Executive functions (goals, GOAP, CP-SAT, learning)
- Chat interface with full context building
- REST API with 30+ endpoints
- A/B testing and ML training pipeline
- Phase 2 cognitive layers (Drives → Narrative) integrated

### In Progress 🚧
- Enhanced UI visualizations (plan viewer, Gantt charts)
- Production monitoring dashboard

### Planned 📋
- Multi-agent collaboration
- Continuous model retraining
- Enhanced sentiment analysis

See `docs/archive/planning/roadmap.md` for details.

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Built With

- **ChromaDB** - Vector database
- **sentence-transformers** - Embeddings
- **Google OR-Tools** - Constraint solving
- **FastAPI** - REST API
- **Chainlit** - Chat UI framework
- **Streamlit** - Legacy UI
- **scikit-learn** - ML pipeline
- **scipy** - Statistical analysis

---

**Built with 🧠 for human-like AI cognition**
