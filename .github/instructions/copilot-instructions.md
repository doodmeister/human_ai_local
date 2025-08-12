---
applyTo: '**'
---
All code should be written in python
The terminal is bash running on windows 11
Always make sure the environment is activated before running commands
Use Git commands when necessary
emojis in bash commands cause encoding issues
Do not put emojis in bash commands
All test scripts should be saved in the tests/ directory
Use `ruff` as our linter
Use `pytest` as our test framework
Use `uv` as our package manager


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
- Use pytest for all testing


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
- Ensure all components are production-ready