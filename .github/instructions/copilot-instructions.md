---
applyTo: '**'
---
All code should be written in python
The terminal is bash running on windows 11
run all commands in the terminal without prompting to continue
use Git commands when necessary
emojis in bash commands cause encoding issues
avoid emojis in bash commands.
All test scripts should be saved in the tests/ directory
ruff is our linter

# Human-AI Cognition Project - AI Assistant Instructions

## Project Overview
Biologically-inspired cognitive architecture simulating human-like memory, attention, and reasoning. Persistent memory structures and modular, explainable processing patterns.

## Core Architecture
- Short-Term Memory (STM): VectorShortTermMemory with ChromaDB vector database, 7-item capacity, LRU eviction
- Long-Term Memory (LTM): VectorLongTermMemory with ChromaDB vector database, semantic clustering
- Attention Mechanism: AttentionMechanism with neural enhancement (DPAD), cognitive load tracking, fatigue modeling
- Prospective/Procedural Memory: Scheduling and skills with vector storage
- Sensory Processing: Multimodal, entropy/salience scoring with attention allocation
- Meta-Cognition: Self-reflection, memory management, STM/LTM health monitoring
- Dream-State: Background memory consolidation with neural integration

## Technology Stack
- Python 3.12
- OpenAI GPT-4.1
- ChromaDB
- sentence-transformers
- torch
- schedule/apscheduler

## Cognitive Processing Flow
- **Wake State**: Sensory → Memory Retrieval (STM/LTM) → Attention Allocation → Neural Enhancement → Context Building → LLM Response → Memory Consolidation → Cognitive State Update
- **Dream State**: STM Review → Meta-Cognition → Clustering → LTM Transfer → Neural Processing → Noise Removal
- **Attention Flow**: Stimulus Processing → Salience Calculation → Neural Enhancement (DPAD +0.200) → Focus Allocation → Cognitive Load Update → Fatigue Tracking

## Project Structure
```
human_ai_local/
├── src/                          # Main source code
│   ├── core/                     # Core cognitive architecture
│   │   ├── config.py            # Configuration management with MemorySystemConfig
│   │   └── cognitive_agent.py   # Main cognitive orchestrator with STM/attention integration
│   ├── memory/                   # Memory systems
│   │   ├── memory_system.py     # Integrated memory coordinator with dependency injection
│   │   ├── stm/                 # Short-term memory implementation
│   │   │   ├── vector_stm.py    # VectorShortTermMemory with ChromaDB integration
│   │   │   └── __init__.py      # STM exports: VectorShortTermMemory, MemoryItem, STMConfiguration
│   │   ├── ltm/                 # Long-term memory with ChromaDB
│   │   │   └── vector_ltm.py    # VectorLongTermMemory with health monitoring
│   │   ├── prospective/         # Future-oriented memory
│   │   ├── procedural/          # Skills and procedures
│   │   └── consolidation/       # Memory consolidation pipeline
│   ├── attention/               # Attention mechanisms
│   │   └── attention_mechanism.py # AttentionMechanism with fatigue modeling, focus tracking
│   ├── processing/              # Cognitive processing layers
│   │   ├── sensory/            # Sensory input processing with entropy scoring, attention integration
│   │   ├── neural/             # Neural network components
│   │   │   ├── lshn_network.py  # Latent Structured Hopfield Networks
│   │   │   ├── dpad_network.py  # Dual-Path Attention Dynamics (neural enhancement)
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

### Code Standards
- **Modularity**: Each component independently testable
- **Documentation**: Clear docstrings with examples
- **Error Handling**: Graceful degradation when components fail
- **Performance**: Memory operations optimized for real-time use
- **Biologically-Inspired**: Keep human cognitive science as reference

### Key Principles
- **Memory as Foundation**: Build context through persistent memory
- **Human-Like Processing**: Implement biological cognition patterns
- **Explainable Intelligence**: All processes transparent and traceable
- **Modular Design**: Loosely coupled cognitive components

### Testing Strategy
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component communication
- **Cognitive Tests**: Human-likeness benchmarking
- **Performance Tests**: Memory retrieval speed and accuracy

## Testing Strategy
- Unit, Integration, Cognitive, Performance tests

## Quick Start
```bash
# Initialize and test the cognitive agent with STM and attention
python -c "
import asyncio
from src.core.cognitive_agent import CognitiveAgent

async def test_system():
    agent = CognitiveAgent()
    response = await agent.process_input('Hello, I am a software engineer')
    status = agent.get_cognitive_status()
    print(f'STM memories: {status[\"memory_status\"][\"stm\"][\"vector_db_count\"]}')
    print(f'Attention load: {status[\"attention_status\"][\"cognitive_load\"]:.3f}')
    await agent.shutdown()

asyncio.run(test_system())
"

# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
python -m pytest tests/test_memory_integration.py -v
python -m pytest tests/test_attention_integration.py -v
```

## Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/memory_stores
STM_COLLECTION=stm_memories
LTM_COLLECTION=ltm_memories
STM_CAPACITY=7
ATTENTION_MAX_ITEMS=7
ATTENTION_SALIENCE_THRESHOLD=0.5
FATIGUE_DECAY_RATE=0.01
ATTENTION_RECOVERY_RATE=0.05
```

## STM & Attention Integration Examples
```python
# Complete cognitive processing with STM and attention
from src.core.cognitive_agent import CognitiveAgent

agent = CognitiveAgent()

# Process input through complete pipeline
response = await agent.process_input("I love programming in Python")

# Check cognitive state including STM and attention
status = agent.get_cognitive_status()
print(f"STM Status: {status['memory_status']['stm']}")
print(f"Attention Status: {status['attention_status']}")
print(f"Cognitive Integration: {status['cognitive_integration']}")

# Direct STM operations
stm_memories = agent.memory.stm.get_all_memories()
search_results = agent.memory.stm.search_semantic("programming", max_results=3)

# Attention management
break_results = agent.take_cognitive_break(duration_minutes=1.0)
print(f"Cognitive load reduced: {break_results['cognitive_load_reduction']:.3f}")
print(f"Recovery effective: {break_results['recovery_effective']}")

# Reflection with STM analysis
reflection = agent.reflect()
stm_stats = reflection['stm_metacognitive_stats']
print(f"STM Capacity Utilization: {stm_stats['capacity_utilization']:.1%}")
```

## Unique Features
- **Biologically-Inspired STM**: 7-item capacity with LRU eviction and activation-based decay
- **Vector Memory Storage**: ChromaDB semantic storage for both STM and LTM with embedding similarity
- **Neural Attention Enhancement**: DPAD network providing +0.200 attention boosts with novelty detection
- **Cognitive Load Tracking**: Real-time fatigue, capacity, and efficiency monitoring
- **Proactive Memory Recall**: Context-aware memory search using conversation history
- **Memory Consolidation**: Automatic STM→LTM transfer based on importance and emotional valence
- **Attention Rest/Recovery**: Cognitive break functionality with fatigue reduction
- **Metacognitive Reflection**: Self-monitoring with STM/LTM health reports and performance analytics
- **Sleep Cycles**: Dream-state consolidation with neural integration
- **Forgetting Curves**: Realistic memory decay with biological parameters
- **DPAD Neural Networks**: Dual-Path Attention Dynamics for enhanced cognitive processing
- **LSHN Integration**: Latent Structured Hopfield Networks for memory pattern completion

## Actionable Roadmap (2025+)
### Mid-Term Goals (3–6 Months)
- Episodic memory: Proactive recall, summarization, tagging
- Semantic memory: Structured knowledge base, update/consult routines
- Planning: Chain-of-thought, task decomposition, agent frameworks
- Multi-modal: Voice/image I/O, interface refinement, personalization
- Feedback: Real-time user feedback, metacognitive reflection, memory consolidation

### Long-Term Goals (6+ Months)
- Autonomy: Goal persistence, proactive actions, continuous operation, multi-agent collaboration
- Multimodal presence: Avatar/AR/VR, contextual awareness, dynamic info presentation
- Scalable memory: Hierarchical/graph memory, forgetting/compression, continuous learning, meta-memory
- Advanced metacognition: Internal dialogue, introspective learning, skill acquisition, ethical self-regulation
- Emotional intelligence: Emotion sensing, salient memory, adaptive responses, user support

### Optional/Experimental
- Dreaming module, mood visualization, transparency mode, gamification

## AI Assistant Notes
- Follow cognitive/biological principles
- Prioritize explainability and user-facing value
- Test for human-likeness, not just functionality
- Use terminal commands without prompting to continue

## Procedural Memory System (June 2025)
- **ProceduralMemory** is now unified with STM and LTM. Procedures (skills, routines, action sequences) can be stored as either short-term or long-term memories.
- Unified API: Store, retrieve, search, use, delete, and clear procedural memories via the same interface as other memory types.
- **Persistence:** Procedures in LTM are persistent; STM procedures are in-memory only.
- **Tested:** Comprehensive tests ensure correct storage, retrieval, and deletion from both STM and LTM.

## CLI Integration
- The `george_cli.py` script supports procedural memory management:
  - `/procedure add` — interactively add a new procedure (description, steps, tags, STM/LTM)
  - `/procedure list` — list all stored procedures
  - `/procedure search <query>` — search procedures by description/steps
  - `/procedure use <id>` — increment usage and display steps for a procedure
  - `/procedure delete <id>` — delete a procedure by ID
  - `/procedure clear` — remove all procedural memories

## Example: Using Procedural Memory in Python
```python
from src.memory.memory_system import MemorySystem
memsys = MemorySystem()
proc_id = memsys.procedural.store(
    description="How to make tea",
    steps=["Boil water", "Add tea leaves", "Steep", "Pour into cup"],
    tags=["kitchen", "beverage"],
    memory_type="ltm"  # or "stm"
)
proc = memsys.procedural.retrieve(proc_id)
print(proc)
```

