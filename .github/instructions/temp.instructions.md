---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.
All code written in python, terminal is bash running on windows 11
run all commands in the terminal without prompting to continue
use Git commands to checkout and run in a dev branch


# Human-AI Cognition Project - AI Assistant Instructions

## 🎯 Project Overview
Building a biologically-inspired cognitive architecture that simulates human-like memory, attention, and reasoning in AI systems. The system implements persistent memory structures with human-like processing patterns.

## 🧠 Core Architecture

### Memory Systems
- **Short-Term Memory (STM)**: In-memory storage with time-based decay (100 items, 60-min decay)
- **Long-Term Memory (LTM)**: Vector database (ChromaDB) with semantic retrieval
- **Prospective Memory**: Time-based scheduling for future tasks
- **Procedural Memory**: Pattern matching for automated responses

### Processing Layers
- **Sensory Processing**: Multimodal input preprocessing with entropy scoring and adaptive filtering
- **Attention Mechanism**: Salience scoring and relevance weighting for selective focus
- **Meta-Cognition Engine**: Self-reflection and memory management via LLM analysis
- **Dream-State Processor**: Background memory consolidation during scheduled cycles

## 🛠 Technology Stack
- **Python 3.12** - Primary language
- **OpenAI GPT-4.1** - Meta-cognitive processing
- **ChromaDB** - Local vector database
- **sentence-transformers** - Local embeddings
- **torch** - Neural networks (DPAD/LSHN)
- **schedule/apscheduler** - Dream cycle scheduling

## 🔄 Cognitive Processing Flow

### Primary Loop (Wake State)
1. **Sensory Processing** → entropy/salience scoring → embedding generation
2. **Memory Retrieval** → query STM/LTM for relevant context
3. **Attention Allocation** → use sensory scores for focus distribution
4. **Context Building** → assemble comprehensive prompt with memory
5. **LLM Processing** → generate response with enriched context
6. **Memory Writing** → store conversation in STM for future reference

### Consolidation Loop (Dream State)
1. **Memory Analysis** → review STM for consolidation candidates
2. **Meta-Cognitive Evaluation** → assess importance and relevance
3. **Clustering** → group related memories using HDBSCAN
4. **Selective Transfer** → move high-value memories to LTM
5. **Noise Removal** → clear low-value STM entries

## 📁 Key Project Structure
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
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
└── infrastructure/             # Infrastructure as Code
```

## ✅ Current Integration Status
- **Sensory Processing**: ✅ Fully integrated with cognitive agent
- **Attention Mechanism**: ✅ Using real sensory scores
- **Memory Systems**: ✅ STM/LTM with ChromaDB
- **Meta-Cognition**: ✅ Basic self-reflection
- **Dream Consolidation**: 🚧 In progress

## 🎯 Development Guidelines

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

## 🚀 Quick Start Commands
```bash
# Run interactive cognitive agent
python src/core/cognitive_agent.py

# Run sensory processing tests
python test_sensory_processing.py

# Run integration tests
python test_sensory_integration.py

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
STM_COLLECTION=short_term_memory
LTM_COLLECTION=long_term_memory
```

## 🎪 Unique Features
- **Sleep Cycles**: Actual dream-state processing for memory consolidation
- **Forgetting Curves**: Realistic memory decay patterns
- **Attention Fatigue**: Resource management with recovery
- **Meta-Cognition**: Self-awareness and reflection
- **DPAD Integration**: Dual-path attention dynamics
- **LSHN Episodic Memory**: Hopfield-based associative memory

## 🎯 Current Development Focus
1. **Performance Optimization**: Batch processing and memory usage
2. **Enhanced Multimodal**: Audio, visual input support
3. **Advanced Filtering**: Sophisticated adaptive algorithms
4. **Dream Consolidation**: Complete automated pipeline
5. **Neural Integration**: DPAD/LSHN implementation

## 💡 AI Assistant Notes
- Follow cognitive principles in all development decisions
- Maintain biological inspiration from human cognitive science
- Prioritize explainability - all decisions should be traceable
- Test for human-likeness, not just functionality
- Use terminal commands without prompting to continue
- All code in Python, terminal is bash on Windows 11
