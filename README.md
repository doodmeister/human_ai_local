# Human-AI Cognition Framework

A production-grade cognitive architecture for building AI systems with human-like memory, attention, reasoning, and executive control. Featuring biologically-inspired memory systems, advanced planning algorithms, and continuous learning capabilities.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What is This?

A complete cognitive framework that gives AI systems:
- **Human-like Memory**: Short-term, long-term, episodic, and prospective memory with biological forgetting curves
- **Executive Intelligence**: Goal management, decision-making (AHP, Pareto), GOAP planning, constraint-based scheduling
- **Attention & Metacognition**: Adaptive attention mechanisms, cognitive load tracking, self-monitoring
- **Continuous Learning**: ML-powered outcome tracking, A/B testing, strategy optimization
- **Emotional Awareness**: Emotion-based memory consolidation and context tracking

Think of it as giving your AI agent a "brain" with working memory, long-term knowledge, planning abilities, and self-improvement capabilities.

---

## ✨ Key Features

### 🧠 Memory Systems
- **Short-Term Memory (STM)**: 7-item capacity, activation-based decay, multiple decay modes (exponential/linear/power/sigmoid)
- **Long-Term Memory (LTM)**: ChromaDB vector store, Ebbinghaus forgetting curves, semantic clustering
- **Episodic Memory**: Rich contextual memories with emotional valence, temporal clustering, autobiographical organization
- **Prospective Memory**: Time-based and event-based reminders, semantic search capabilities

### 🎮 Executive Functions
- **Goal Management**: Hierarchical goals with dependencies, priorities, deadlines
- **Advanced Decision-Making**: 
  - Weighted scoring, AHP (Analytic Hierarchy Process), Pareto optimization
  - Context-aware weight adjustment, ML-powered confidence boosting
  - A/B testing with statistical analysis (Chi-square, t-test, Mann-Whitney, Cohen's d)
- **GOAP Planning**: Goal-Oriented Action Planning with A* search, multiple heuristics, constraint support
- **CP-SAT Scheduling**: Google OR-Tools constraint solver, resource management, cognitive load balancing
- **Dynamic Adaptation**: Real-time schedule monitoring, disruption handling, proactive warnings

### 📊 Learning Infrastructure
- **Outcome Tracking**: Records execution results, accuracy analysis, improvement trends
- **Feature Extraction**: 23-field feature vectors for ML training (CSV/JSON/Parquet export)
- **ML Training Pipeline**: 4 models (strategy classifier, success predictor, time regressor, outcome scorer)
- **A/B Testing**: Randomized experiments with 3 assignment methods (Random, Epsilon-Greedy, Thompson Sampling)
- **Statistical Analysis**: Automated strategy recommendation with confidence intervals

### 🎯 Attention & Metacognition
- **Attention Mechanism**: Fatigue tracking, capacity limits, salience scoring
- **Cognitive Load**: Real-time monitoring, adaptive thresholds, overload detection
- **Self-Monitoring**: Performance metrics, memory health diagnostics, system telemetry

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/human_ai_local.git
cd human_ai_local

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**1. Start the Chat Interface**
```bash
python start_george.py
```
Access at http://localhost:8501

**2. Start the API Server**
```bash
python start_server.py
```
Access at http://localhost:8000 (docs at http://localhost:8000/docs)

**3. Use Programmatically**
```python
from src.chat.factory import build_chat_service

# Initialize chat service with all cognitive systems
service = build_chat_service()

# Send a message
response = service.chat("What can you help me with?", session_id="user123")
print(response["response"])

# Memory and context are automatically managed
# - Recent messages stored in STM
# - Important facts promoted to LTM
# - Attention and cognitive load tracked
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Chat Interface                           │
│            (API, Streamlit, CLI)                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  ChatService                                 │
│         (Orchestrates all cognitive systems)                 │
└──────┬────────────────────────────────────────────┬─────────┘
       │                                             │
┌──────▼──────────┐                    ┌────────────▼─────────┐
│ ContextBuilder  │                    │  Executive System    │
│ - STM retrieval │                    │  - Goal management   │
│ - LTM retrieval │                    │  - Decision engine   │
│ - Episodic      │                    │  - GOAP planner      │
│ - Prospective   │                    │  - CP-SAT scheduler  │
│ - Attention     │                    │  - Learning          │
└─────────────────┘                    └──────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                    Memory Systems                            │
│  ┌──────┐  ┌──────┐  ┌──────────┐  ┌──────────────┐        │
│  │ STM  │  │ LTM  │  │ Episodic │  │ Prospective  │        │
│  │ 7-cap│  │Vector│  │ Context  │  │  Reminders   │        │
│  └──────┘  └──────┘  └──────────┘  └──────────────┘        │
│                                                               │
│  ┌────────────────────────────────────────────────┐         │
│  │        Consolidation Engine                     │         │
│  │  (STM → LTM promotion with decay tracking)     │         │
│  └────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────┘
```

---

## 📚 Core Systems

### Memory Systems

| System | Description | Key Features |
|--------|-------------|--------------|
| **STM** | Short-term working memory | 7-item capacity, activation decay, LRU eviction, 4 decay modes |
| **LTM** | Long-term semantic memory | ChromaDB vectors, Ebbinghaus forgetting, salience weighting |
| **Episodic** | Contextual autobiographical memory | Emotional valence, temporal clustering, rich metadata |
| **Prospective** | Future-oriented reminders | Time-based & semantic triggers, ChromaDB optional |
| **Consolidation** | STM→LTM transfer | Age/rehearsal gating, importance thresholds, provenance tracking |

📖 **Docs**: See `docs/enhanced_ltm_summary.md`, `docs/vector_stm_integration_complete.md`

### Executive Functions

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **GoalManager** | Hierarchical goal tracking | Dependencies, priorities, deadlines, success criteria |
| **DecisionEngine** | Multi-criteria decision-making | Weighted/AHP/Pareto strategies, ML predictions, A/B testing |
| **GOAPPlanner** | Goal-oriented action planning | A* search, 5 heuristics, constraint support, replanning |
| **CPScheduler** | Constraint-based scheduling | Google OR-Tools, resource management, optimization |
| **LearningSystem** | Continuous improvement | Outcome tracking, ML training, A/B experiments, analytics |

📖 **Docs**: See `docs/WEEK_15_COMPLETION_SUMMARY.md`, `docs/WEEK_16_PHASE_4_AB_TESTING.md`

### Attention & Metacognition

- **AttentionMechanism**: Fatigue tracking, capacity limits (0-100), salience scoring
- **Metacognitive Monitoring**: Load thresholds, adaptive retrieval, performance telemetry
- **Metrics Registry**: Prometheus-style counters for all subsystems

📖 **Docs**: See `docs/metacog_features.md`, `docs/executive_telemetry.md`

---

## 💡 Usage Examples

### Example 1: Chat with Memory

```python
from src.chat.factory import build_chat_service

service = build_chat_service()

# First conversation
response = service.chat("My favorite color is blue", session_id="user1")
# → Stored in STM, extracted as preference, may promote to LTM

# Later conversation (different session)
response = service.chat("What's my favorite color?", session_id="user1")
# → Retrieved from LTM: "Your favorite color is blue"
```

### Example 2: Goal-Driven Planning

```python
from src.executive.integration import ExecutiveSystem
from src.executive.planning import WorldState

# Initialize executive system
system = ExecutiveSystem()

# Execute a goal with full pipeline
context = system.execute_goal(
    goal_id="analyze_data",
    goal_description="Analyze user sentiment in recent messages",
    initial_state=WorldState({"data_collected": True}),
    success_criteria=["data_analyzed=True", "report_generated=True"]
)

# Pipeline automatically runs:
# 1. Decision Engine chooses strategy
# 2. GOAP Planner creates action sequence
# 3. CP-SAT Scheduler optimizes timing
# 4. Execution tracking monitors progress

print(f"Plan: {len(context.plan.actions)} actions")
print(f"Schedule: {context.schedule.makespan_minutes:.1f} minutes")
```

### Example 3: A/B Testing Strategies

```python
from src.executive.learning import create_experiment_manager
from src.executive import DecisionEngine

# Create experiment
manager = create_experiment_manager()
exp = manager.create_experiment(
    name="Decision Strategy Test",
    strategies=["weighted_scoring", "ahp", "pareto"],
    assignment_method="epsilon_greedy"  # 90% exploit, 10% explore
)
manager.start_experiment(exp.experiment_id)

# DecisionEngine auto-assigns strategies
engine = DecisionEngine(experiment_manager=manager)
result = engine.make_decision(
    options=["option_a", "option_b"],
    criteria={"speed": 0.7, "quality": 0.3},
    experiment_id=exp.experiment_id
)

# Record outcome
engine.record_experiment_outcome(
    assignment_id=result.metadata['assignment_id'],
    success=True,
    outcome_score=0.85
)

# Analyze after 50+ outcomes
analysis = manager.analyze_experiment(exp.experiment_id)
print(f"Winner: {analysis['recommended_strategy']} (p={analysis['significance_level']:.3f})")
```

---

## ⚙️ Configuration

Key configuration via environment variables (`.env` file):

```bash
# Memory Configuration
CHROMA_PERSIST_DIR=./data/memory_stores
STM_COLLECTION=stm_memories
LTM_COLLECTION=ltm_memories
STM_CAPACITY=7
SALIENCE_DECAY_RATE=0.1

# LLM Provider
LLM_PROVIDER=openai  # or anthropic, google, ollama
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4

# Cognitive Limits
MAX_ATTENTION_CAPACITY=100
FATIGUE_THRESHOLD=80
COGNITIVE_LOAD_THRESHOLD=0.75

# Feature Flags
DISABLE_SEMANTIC_MEMORY=0
GOAP_ENABLED=1
USE_VECTOR_PROSPECTIVE=0  # Set to 1 for semantic reminders
```

See `.env.example` for all options.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_chat_*.py          # Chat system tests
pytest tests/test_executive_*.py     # Executive function tests
pytest tests/test_memory_*.py        # Memory system tests
pytest tests/test_integration_*.py   # Integration tests

# Run with coverage
pytest --cov=src --cov-report=html

# Quick validation
python -m pytest tests/test_chat_factory_integration.py -v
```

**Test Stats**: 200+ tests covering memory, executive, chat, API, and integration scenarios.

---

## 📖 Documentation

### Getting Started
- **Quick Start**: This README
- **Startup Guide**: `STARTUP_GUIDE.md` - Detailed setup instructions
- **API Documentation**: http://localhost:8000/docs (when server running)

### System Documentation
- **Memory Systems**: `docs/enhanced_ltm_summary.md`, `docs/vector_stm_integration_complete.md`
- **Executive Functions**: `docs/WEEK_15_COMPLETION_SUMMARY.md` (System Integration)
- **GOAP Planning**: `docs/PHASE_2_FINAL_COMPLETE.md`
- **Scheduling**: `docs/WEEK_12_COMPLETION_SUMMARY.md`, `docs/WEEK_14_COMPLETION_SUMMARY.md`
- **Learning**: `docs/WEEK_16_PHASE_4_AB_TESTING.md` (A/B Testing), `docs/WEEK_16_PHASE_3_TRAINING_PIPELINE.md` (ML)
- **Metacognition**: `docs/metacog_features.md`, `docs/executive_telemetry.md`

### Quick References
- **Week 16 Phase 4**: `docs/WEEK_16_PHASE_4_QUICK_REF.md` (A/B Testing cheat sheet)
- **Week 15**: `docs/WEEK_15_QUICK_REFERENCE.md` (Integration quick start)
- **AI Instructions**: `.github/copilot-instructions.md` (Development patterns)

### Project Planning
- **Roadmap**: `docs/roadmap.md` - Future development plans
- **Architecture**: `docs/executive_refactoring_plan.md` - System design

---

## 🛠️ Development

### Project Structure

```
human_ai_local/
├── src/
│   ├── chat/              # Chat service, context building
│   ├── memory/            # STM, LTM, episodic, prospective
│   ├── executive/         # Goals, decisions, planning, scheduling, learning
│   ├── attention/         # Attention mechanism, fatigue tracking
│   ├── processing/        # Neural networks, dream processing
│   ├── interfaces/        # API endpoints
│   └── core/              # Configuration, base classes
├── tests/                 # Comprehensive test suite
├── scripts/               # Streamlit UI, CLI tools
├── docs/                  # Detailed documentation
├── data/                  # Persistent storage (memory, experiments, outcomes)
└── .github/               # CI/CD, copilot instructions
```

### Contributing

1. **Code Style**: Ruff for linting, type hints required
2. **Testing**: Add tests for new features, maintain >80% coverage
3. **Documentation**: Update relevant docs/ files for major changes
4. **Commits**: Descriptive messages, reference issues when applicable

---

## 🗺️ Roadmap

### Completed ✅
- ✅ Memory systems (STM, LTM, episodic, prospective) with biological forgetting
- ✅ Executive functions (goals, decisions, GOAP planning, CP-SAT scheduling)
- ✅ Learning infrastructure (outcome tracking, ML training, A/B testing)
- ✅ Attention and metacognition with adaptive thresholds
- ✅ Chat interface with context building and consolidation

### In Progress 🚧
- 🚧 Production monitoring dashboard
- 🚧 Visualization tools for experiments and schedules
- 🚧 Enhanced sentiment analysis and emotion modeling

### Planned 📋
- Multi-agent collaboration (internal specialist agents)
- Proactive memory retrieval and recommendations
- Continuous model retraining pipeline
- Dreaming module (offline insight generation)
- AR/VR interface prototypes

See `docs/roadmap.md` for detailed future plans.

---

## 📊 Performance

- **Chat Response Time**: <1s typical, <3s with full memory retrieval
- **Memory Retrieval**: <100ms STM, <200ms LTM (ChromaDB)
- **GOAP Planning**: <50ms simple plans, <500ms complex (20+ actions)
- **CP-SAT Scheduling**: <30s for 50 tasks with constraints
- **Executive Pipeline**: 12-15s end-to-end (Goal→Decision→Plan→Schedule)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- **ChromaDB** - Vector database for semantic memory
- **sentence-transformers** - Embedding models for similarity search
- **Google OR-Tools** - Constraint satisfaction for scheduling
- **FastAPI** - REST API framework
- **Streamlit** - Chat UI framework
- **scikit-learn** - ML training pipeline
- **scipy** - Statistical analysis

Inspired by cognitive science research on human memory, attention, and executive function.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/human_ai_local/issues)
- **Documentation**: `docs/` directory
- **API Docs**: http://localhost:8000/docs (when server running)

---

**Built with 🧠 by the Human-AI Cognition team**
