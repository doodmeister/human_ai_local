# Human-AI Cognition Framework

A biologically-inspired cognitive architecture for human-like memory, attention, and reasoning in AI systems. Features persistent, explainable memory structures, modular processing, and advanced neural integration.

## Project Vision
Build robust, production-ready AI with human-like cognition: persistent memory, attention, neural replay, and dream-state consolidation. All processes are transparent, traceable, and extensible.

## Core Architecture
- Short-Term Memory (STM): In-memory, time-decayed, vector search (ChromaDB)
- Long-Term Memory (LTM): ChromaDB vector database, semantic retrieval
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
- Sleep cycles, forgetting curves, attention fatigue, meta-cognition, DPAD, LSHN

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

## Development Guidelines
- Modularity: Independently testable components
- Documentation: Clear docstrings/examples
- Error Handling: Graceful degradation
- Performance: Real-time memory ops
- Biologically-Inspired: Human cognitive science as reference
- Explainability: Traceable, transparent processes

## Testing Strategy
- Unit, Integration, Cognitive, Performance tests
