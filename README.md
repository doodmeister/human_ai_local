# Human-AI Cognition Framework

## Overview
Biologically-inspired cognitive architecture simulating human-like memory, attention, and reasoning. Persistent memory structures and modular, explainable processing patterns.

---

## 🧠 Major Update: Enhanced Long-Term Memory with Biologically-Inspired Features (June 2025)

### Advanced LTM Capabilities
The Long-Term Memory (LTM) system has been significantly enhanced with biologically-inspired features that mirror human memory processes:

#### 1. Salience & Recency Weighting in Retrieval
- **Dynamic Retrieval Scoring**: Memory retrieval now considers both content relevance and temporal/access patterns
- **Exponential Decay Model**: Recent and frequently accessed memories receive higher priority in search results
- **Access Pattern Learning**: System learns which memories are most valuable based on usage patterns

#### 2. Memory Decay & Forgetting
- **Biological Forgetting Curves**: Implements Ebbinghaus-style forgetting with configurable decay rates
- **Importance-Based Preservation**: More important memories resist decay longer
- **Confidence Degradation**: Memory confidence naturally decreases over time without reinforcement
- **Selective Pruning**: Old, rarely accessed memories automatically lose strength

#### 3. Consolidation Tracking
- **STM→LTM Transfer Monitoring**: Tracks when and how memories move from short-term to long-term storage
- **Consolidation Metadata**: Records consolidation timestamps, sources, and transfer statistics
- **Query Methods**: Retrieve recently consolidated memories and analyze consolidation patterns
- **Performance Analytics**: Detailed statistics on memory consolidation efficiency

#### 4. Meta-Cognitive Feedback
- **Self-Monitoring**: System tracks its own memory performance and retrieval patterns
- **Health Diagnostics**: Automatic assessment of memory system health and performance
- **Usage Statistics**: Comprehensive metrics on search success rates, timing, and efficiency
- **Recommendations Engine**: System provides suggestions for memory management optimization

#### 5. Emotionally Weighted Consolidation
- **Emotional Significance**: Memories with strong emotional content (positive or negative) are prioritized for consolidation
- **Multi-Factor Scoring**: Combines importance, access frequency, emotional weight, and recency for consolidation decisions
- **Adaptive Thresholds**: Emotional memories may be consolidated even with lower traditional importance scores
- **Trauma/Joy Preservation**: Both traumatic and highly positive experiences receive enhanced consolidation

#### 6. Cross-System Query & Linking
- **Bidirectional Associations**: Create and query links between LTM and other memory systems (STM, episodic)
- **Semantic Clustering**: Automatically identify and group related memories by content and tags
- **Cross-System Suggestions**: AI-powered recommendations for linking memories across different systems
- **Association Networks**: Build rich networks of related memories for enhanced recall and context

### Enhanced API Methods
```python
# Salience/Recency weighted search
results = ltm.search_by_content("query", max_results=10)  # Now includes temporal weighting

# Memory decay management
decayed_count = ltm.decay_memories(decay_rate=0.01, half_life_days=30.0)

# Consolidation tracking
consolidated = ltm.consolidate_from_stm(stm_items)  # Now with emotional weighting
recent_memories = ltm.get_recently_consolidated(hours=24)
stats = ltm.get_consolidation_stats()

# Meta-cognitive feedback
meta_stats = ltm.get_metacognitive_stats()
health_report = ltm.get_memory_health_report()
ltm.reset_metacognitive_stats()

# Cross-system linking
ltm.create_cross_system_link(memory1_id, memory2_id, "association")
links = ltm.find_cross_system_links(memory_id)
clusters = ltm.get_semantic_clusters(min_cluster_size=2)
suggestions = ltm.suggest_cross_system_associations(external_memories, "system_type")
```

### Testing & Validation
- **Comprehensive Test Suite**: Individual tests for each enhanced feature
- **Integration Testing**: End-to-end testing of all features working together
- **Performance Benchmarks**: Validation of enhanced retrieval speed and accuracy
- **Biological Validation**: Tests confirm human-like memory behavior patterns

### Key Benefits
- **Human-Like Memory**: More realistic forgetting and remembering patterns
- **Improved Efficiency**: Better memory management through automated decay and consolidation
- **Enhanced Recall**: Smarter retrieval based on usage patterns and emotional significance
- **Self-Optimization**: System continuously improves its own memory management
- **Rich Associations**: Better context and relationship understanding across memories

---

## 🚀 Recent Major Update: Unified Memory Interface & Episodic Memory Improvements (June 2025)

### Unified Memory Interface
- **All major memory modules (STM, LTM, Episodic, Semantic)** now implement a consistent, type-safe interface via a shared `BaseMemorySystem` class (`src/memory/base.py`).
- **Unified API methods** for all memory systems:
  - `store(...)`: Store a new memory (returns memory ID)
  - `retrieve(memory_id)`: Retrieve a memory as a dict
  - `delete(memory_id)`: Delete a memory by ID
  - `search(query, **kwargs)`: Search for memories (returns list of dicts)
- All memory modules return dicts and use unified parameter names for easier integration and testing.

### Episodic Memory System Enhancements
- **Fallback search**: Robust fallback logic using word overlap heuristics and debug output if ChromaDB is unavailable or returns no results.
- **Related memory logic**: Improved detection of related memories (temporal, cross-reference, semantic) with debug output for explainability.
- **New/Updated Public API Methods**:
  - `get_related_memories(memory_id, relationship_types=None, limit=10)`
  - `get_autobiographical_timeline(life_period=None, start_date=None, end_date=None, limit=50)`
  - `consolidate_memory(memory_id, strength_increment=0.1)`
  - `get_memory_statistics()`
  - `clear_memory(older_than=None, importance_threshold=None)`
  - `get_consolidation_candidates(min_importance=0.5, max_consolidation=0.9, limit=10)`
  - `clear_all_memories()` (for test isolation)
- **Debug output**: All fallback and related memory logic now prints detailed debug information for transparency and troubleshooting.

### Testing & Reliability
- **Integration tests** for vector LTM and episodic memory updated to use the new interface and auto-generated summaries.
- **Test isolation**: All tests clear persistent and in-memory data before each run for clean, isolated test runs.
- **All episodic memory integration tests pass.**

---

## Unified Memory Interface and Episodic Memory Enhancements (June 2025)

### Unified Memory Interface
- All major memory modules (STM, LTM, Episodic, Semantic) now implement a consistent, type-safe interface via a shared `BaseMemorySystem` class (`src/memory/base.py`).
- Unified public API methods: `store`, `retrieve`, `delete`, `search` (all return dicts, use unified parameter names).
- All memory modules inherit from `BaseMemorySystem` and enforce type annotations for robust, modular design.

### Episodic Memory System Improvements
- Major refactor of `EpisodicMemorySystem` (`src/memory/episodic/episodic_memory.py`):
  - Robust fallback search logic (word overlap, text match) with detailed debug output for explainability.
  - Enhanced related memory logic (cross-reference, temporal, semantic) with debug output.
  - All required public API methods implemented and exposed:
    - `get_related_memories`
    - `get_autobiographical_timeline`
    - `consolidate_memory`
    - `get_memory_statistics`
    - `clear_memory`
    - `get_consolidation_candidates`
    - `clear_all_memories` (for test isolation)
- All methods return type-safe results and provide debug output for fallback/related memory logic.

### Testing and Test Isolation
- Integration tests for vector LTM and episodic memory updated to use the new interface.
- Test isolation: persistent and in-memory data cleared before each run to ensure clean test runs.
- All episodic memory integration tests pass.

### Documentation
- This section documents the new unified memory interface, episodic memory improvements, new public API methods, and updated testing strategy.
- See `src/memory/base.py` for the base interface and `src/memory/episodic/episodic_memory.py` for the full implementation and debug logic.

---

## 🚀 Major Update: Unified Memory, Procedural Memory, and CLI Integration (June 2025)

### Procedural Memory System
- **ProceduralMemory** is now fully integrated with STM and LTM. Procedures (skills, routines, action sequences) can be stored as either short-term or long-term memories.
- Unified API: Store, retrieve, search, use, delete, and clear procedural memories via the same interface as other memory types.
- **Persistence:** Procedures stored in LTM are persistent across runs; STM procedures are in-memory only.
- **Tested:** Comprehensive tests ensure correct storage, retrieval, and deletion from both STM and LTM.

### CLI Integration
- The `george_cli.py` script now supports procedural memory management:
  - `/procedure add` — interactively add a new procedure (description, steps, tags, STM/LTM)
  - `/procedure list` — list all stored procedures
  - `/procedure search <query>` — search procedures by description/steps
  - `/procedure use <id>` — increment usage and display steps for a procedure
  - `/procedure delete <id>` — delete a procedure by ID
  - `/procedure clear` — remove all procedural memories

### Metacognitive Reflection & Self-Monitoring (June 2025)
- **Agent-level self-reflection:** The agent can periodically or manually analyze its own memory health, usage, and performance.
- **Reflection Scheduler:** Background scheduler runs metacognitive reflection at a configurable interval (default: 10 min).
- **Manual Reflection:** Trigger a reflection at any time via CLI or API.
- **Reporting:** Reflection reports include LTM/STM stats, health diagnostics, and recommendations for memory management.
- **CLI Integration:**
  - `/reflect` — manually trigger a reflection and print summary
  - `/reflection status` — show last 3 reflection reports
  - `/reflection start [interval]` — start scheduler (interval in minutes)
  - `/reflection stop` — stop scheduler

#### Example CLI Usage
```
/reflect
/reflection status
/reflection start 5
/reflection stop
```

---

## Usage Example (Unified Memory API)
```python
# Example: Storing and searching episodic memory
from src.memory.episodic.episodic_memory import EpisodicMemorySystem

memsys = EpisodicMemorySystem()
mem_id = memsys.store(detailed_content="Visited the science museum with friends.")
result = memsys.retrieve(mem_id)
print(result)

# Search
results = memsys.search(query="museum")
for r in results:
    print(r)
```

---

# Human-AI Cognition Framework

A biologically-inspired cognitive architecture for human-like memory, attention, and reasoning in AI systems. Features persistent, explainable memory structures, modular processing, and advanced neural integration.

## Project Vision
Build robust, production-ready AI with human-like cognition: persistent memory, attention, neural replay, and dream-state consolidation. All processes are transparent, traceable, and extensible.

## Core Architecture
- Short-Term Memory (STM): In-memory, time-decayed, vector search (ChromaDB)
- **Long-Term Memory (LTM): Enhanced ChromaDB vector database with biologically-inspired features:**
  - **Salience/Recency Weighting**: Temporal and access-pattern based retrieval prioritization
  - **Memory Decay & Forgetting**: Ebbinghaus-style forgetting curves with selective preservation
  - **Consolidation Tracking**: STM→LTM transfer monitoring with detailed analytics
  - **Meta-Cognitive Feedback**: Self-monitoring and health diagnostics
  - **Emotional Weighting**: Emotion-based consolidation prioritization
  - **Cross-System Linking**: Bidirectional associations and semantic clustering
- **Semantic Memory:** Structured factual knowledge (subject-predicate-object triples), persistent triple store, agent-level interface for storing, retrieving, and deleting facts
- Episodic Memory: ChromaDB vector database with rich metadata, proactive recall, and automatic summarization/tagging.
- Prospective/Procedural Memory: Scheduling, skills, and routines
- Sensory Processing: Multimodal, entropy/salience scoring
- Attention Mechanism: Salience/relevance weighting, fatigue modeling
- Meta-Cognition: Self-reflection, memory management
- Dream-State: Background memory consolidation

## Technology Stack
- Python 3.12
- OpenAI GPT-4.1
- ChromaDB
- sentence-transformers
- torch
- schedule/apscheduler

## Project Structure

```
human_ai_local/
├── src/                          # Main source code
│   ├── core/                     # Core cognitive architecture
│   │   ├── config.py            # Configuration management
│   │   └── cognitive_agent.py   # Main cognitive orchestrator
│   ├── memory/                   # Memory systems
│   │   ├── memory_system.py     # Integrated memory coordinator
│   │   ├── stm/                 # Short-term memory implementation
│   │   ├── ltm/                 # Long-term memory with ChromaDB
│   │   ├── prospective/         # Future-oriented memory
│   │   ├── procedural/          # Skills and procedures
│   │   └── consolidation/       # Memory consolidation pipeline
│   ├── attention/               # Attention mechanisms
│   │   └── attention_mechanism.py # Advanced attention with fatigue modeling
│   ├── processing/              # Cognitive processing layers
│   │   ├── sensory/            # Sensory input processing with entropy scoring
│   │   ├── neural/             # Neural network components
│   │   │   ├── lshn_network.py  # Latent Structured Hopfield Networks
│   │   │   ├── dpad_network.py  # Dual-Path Attention Dynamics
│   │   │   └── neural_integration.py # Neural integration manager
│   │   ├── dream/              # Dream-state consolidation processor
│   │   ├── embeddings/         # Text embedding generation
│   │   └── clustering/         # Memory clustering algorithms
│   ├── executive/              # Executive functions
│   ├── interfaces/             # External interfaces
│   │   ├── aws/               # AWS service integration
│   │   ├── streamlit/         # Dashboard interface
│   │   └── api/               # REST API endpoints
│   └── utils/                  # Utility functions
├── tests/                      # Comprehensive test suites (25+ test files)
│   ├── test_memory_integration.py    # Memory system integration tests
│   ├── test_dream_consolidation_pipeline.py # Dream processing tests
│   ├── test_dpad_integration_fixed.py # DPAD neural network tests
│   ├── test_lshn_integration.py      # LSHN neural network tests
│   ├── test_attention_integration.py # Attention mechanism tests
│   └── test_final_integration_demo.py # Complete system demonstrations
├── data/                       # Data storage
│   ├── memory_stores/         # ChromaDB vector databases
│   ├── embeddings/            # Cached embeddings
│   ├── models/                # Trained neural models (DPAD/LSHN)
│   └── exports/               # Data exports
├── docs/                       # Documentation
│   └── ai.instructions.md     # Comprehensive development guide
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
└── infrastructure/             # Infrastructure as Code
```

## Quick Start
```bash
# 1. Install dependencies (in your virtualenv):
pip install -r requirements.txt

# 2. Start the backend API server (all endpoints, including agent, memory, reflection, etc.):
c:/dev/human_ai_local/venv/Scripts/python.exe -m uvicorn src.interfaces.api.reflection_api:app --reload --port 8000

# 3. (Optional) Run the CLI:
python scripts/george_cli.py

# 4. (Optional) Launch the Streamlit dashboard for George:
streamlit run scripts/george_streamlit.py
```

## Streamlit Dashboard (George)
The new Streamlit interface provides a modern chat UI for interacting with George, including:
- Real-time chat with the agent (uses the same backend API as the CLI)
- Context display for each response
- Sidebar controls to clear chat history

To launch:
```bash
streamlit run scripts/george_streamlit.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Environment Setup
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
STM_COLLECTION=short_term_memory
LTM_COLLECTION=long_term_memory
```

## Unique Features
- **Biologically-Inspired Memory**: Realistic forgetting curves, salience weighting, emotional consolidation
- **Meta-Cognitive Self-Monitoring**: System tracks and optimizes its own memory performance
- **Cross-System Memory Linking**: Rich associative networks across different memory types
- Sleep cycles, attention fatigue, dream-state consolidation
- Advanced neural networks: DPAD (Dual-Path Attention Dynamics), LSHN (Latent Structured Hopfield Networks)
- Temporal memory dynamics with decay and reinforcement patterns

## Actionable Roadmap (2025+)
### Mid-Term Goals (3–6 Months)
- Planning: Chain-of-thought, task decomposition, agent frameworks *(Not yet accomplished)*
- Multi-modal: Voice/image I/O, interface refinement, personalization *(Not yet accomplished)*
- Feedback: Real-time user feedback, metacognitive reflection, memory consolidation *(Partially accomplished: memory consolidation present; real-time feedback and metacognitive reflection not yet implemented)*

### Long-Term Goals (6+ Months)
- Autonomy: Goal persistence, proactive actions, continuous operation, multi-agent collaboration
- Multimodal presence: Avatar/AR/VR, contextual awareness, dynamic info presentation
- Scalable memory: Hierarchical/graph memory, forgetting/compression, continuous learning, meta-memory
- Advanced metacognition: Internal dialogue, introspective learning, skill acquisition, ethical self-regulation
- Emotional intelligence: Emotion sensing, salient memory, adaptive responses, user support

### Optional/Experimental
- Dreaming module, mood visualization, transparency mode, gamification

## Development Guidelines
- Modularity: Independently testable components
- Documentation: Clear docstrings/examples
- Error Handling: Graceful degradation
- Performance: Real-time memory ops
- Biologically-Inspired: Human cognitive science as reference
- Explainability: Traceable, transparent processes

## Testing Strategy
- Unit, Integration, Cognitive, Performance tests

## Recent Updates (June 2025)
- **Enhanced Long-Term Memory (LTM) with Biologically-Inspired Features:** Complete overhaul of LTM system with salience/recency weighting, memory decay/forgetting, consolidation tracking, meta-cognitive feedback, emotionally weighted consolidation, and cross-system query/linking capabilities.
- **Semantic Memory System:** Added a persistent, structured semantic memory system for storing, retrieving, and deleting subject-predicate-object triples. Integrated at the agent level with a unified interface.
- **Agent Fact Management:** The agent can now store, retrieve, and delete structured facts using the new semantic memory system. Fact operations are normalized and robust.
- **Integration & Unit Tests:** Comprehensive integration and unit tests for semantic memory, including agent-level end-to-end tests for fact storage, retrieval, and deletion.
- **Semantic Memory Config:** Added `semantic_storage_path` to configuration for flexible semantic memory storage location. Semantic memory can be cleared for testing.
- **Episodic Memory Proactive Recall:** The agent can now proactively recall relevant episodic memories based on the current context, improving conversational continuity and depth.
- **Automatic Summarization and Tagging:** Episodic memories are automatically summarized and tagged with keywords upon creation, significantly enhancing search and retrieval efficiency.
- **Enhanced Testing:** Added comprehensive integration and unit tests for proactive recall and episodic memory features to ensure system stability and correctness.
- **ChromaDB Reliability:** Improved ChromaDB initialization and shutdown procedures to prevent file-locking issues, particularly on Windows environments.

## Next Steps & Roadmap (2025+)

### Security & Production Readiness
- Restrict `/test/reset` endpoint to test/dev environments only
- Add authentication for API endpoints if exposed outside localhost
- Implement rate limiting and logging for API endpoints

### API & Feature Expansion
- Expose more agent features via API: STM/LTM queries, feedback, procedural memory, etc.
- Enhance OpenAPI/Swagger docs with more examples and descriptions

### Monitoring & Observability
- Add metrics endpoints (e.g., `/metrics` for Prometheus)
- Integrate logging for API calls and agent events

### Dashboard/UI
- Build a Streamlit or React dashboard for real-time monitoring and control of the agent (reflection, memory stats, etc.)

### Documentation
- Update `README.md` and/or create a dedicated `docs/api.md` for API usage, examples, and test instructions
- Add architecture diagrams to `docs/`

### Cloud/Deployment
- Dockerize the API for easy deployment
- Add CI/CD (e.g., GitHub Actions) for linting, testing, and deployment

### Advanced Cognitive Features
- Extend metacognitive reflection to other memory systems (STM, procedural, etc.)
- Implement automated self-healing/optimization using reflection results
