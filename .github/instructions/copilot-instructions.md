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
- Short-Term Memory (STM): In-memory, time-decayed
- Long-Term Memory (LTM): ChromaDB vector database
- Prospective/Procedural Memory: Scheduling and skills
- Sensory Processing: Multimodal, entropy/salience scoring
- Attention Mechanism: Salience/relevance weighting
- Meta-Cognition: Self-reflection, memory management
- Dream-State: Background memory consolidation

## Technology Stack
- Python 3.12
- OpenAI GPT-4.1
- ChromaDB
- sentence-transformers
- torch
- schedule/apscheduler

## Cognitive Processing Flow
- Wake: Sensory → Memory → Attention → Context → LLM → STM Write
- Dream: STM Review → Meta-Cognition → Clustering → LTM Transfer → Noise Removal

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
python src/core/cognitive_agent.py
pip install -r requirements.txt
```

## Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
STM_COLLECTION=short_term_memory
LTM_COLLECTION=long_term_memory
```

## Unique Features
- Sleep Cycles, Forgetting Curves, Attention Fatigue, Meta-Cognition, DPAD, LSHN

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

