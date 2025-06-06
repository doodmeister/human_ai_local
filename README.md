# Human-AI Cognition Framework

A sophisticated biologically-inspired cognitive architecture that implements human-like memory, attention, reasoning, and neural processing capabilities in AI systems. This advanced framework features fully functional neural networks, memory consolidation pipelines, and cognitive processing systems.

## 🎯 Project Vision

This framework revolutionizes AI by implementing a complete cognitive architecture that mimics human cognition through persistent memory structures, attention mechanisms, neural replay, and dream-state consolidation. Unlike traditional stateless AI, this system maintains continuous context, learns from experience, and exhibits human-like cognitive patterns.

## 🧠 Core Philosophy

- **Memory as Foundation**: Persistent memory structures with STM/LTM integration and ChromaDB vector storage
- **Biologically-Inspired Processing**: Real sleep cycles, neural replay, attention fatigue, and memory consolidation
- **Neural Architecture Integration**: LSHN (Latent Structured Hopfield Networks) and DPAD (Dual-Path Attention Dynamics)
- **Explainable Intelligence**: All cognitive processes are transparent, traceable, and introspectable

## 📁 Project Structure

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

## 🧠 Cognitive Architecture Components

### Memory Systems (Fully Integrated)
- **Short-Term Memory (STM)**: Advanced working memory with temporal decay and capacity limits
- **Long-Term Memory (LTM)**: ChromaDB-powered semantic memory with vector similarity search
- **Memory Consolidation**: HDBSCAN clustering during dream cycles for pattern formation
- **Prospective Memory**: Future-oriented task scheduling and reminder systems
- **Procedural Memory**: Skills, habits, and learned behavioral patterns
- **Episodic Memory**: LSHN-based autobiographical memory with Hopfield network storage

### Advanced Neural Processing
- **LSHN Networks**: Latent Structured Hopfield Networks for complex pattern recognition
- **DPAD Integration**: Dual-Path Attention Dynamics for temporal sequence processing
- **Hopfield Layers**: Modern continuous associative memory networks
- **Neural Replay**: Dream-state synaptic consolidation and memory strengthening
- **Attention Mechanisms**: Multi-head attention with realistic fatigue modeling

### Sensory & Cognitive Processing
- **Sensory Buffer**: Entropy-based input filtering with salience scoring
- **Attention Engine**: Dynamic resource allocation with fatigue and recovery cycles
- **Meta-Cognition Module**: Self-monitoring, reflection, and cognitive state awareness
- **Dream Processor**: Sleep-cycle memory consolidation with clustering algorithms

### Executive & Integration Systems
- **Cognitive Agent Core**: Central orchestration of all cognitive processes (`cognitive_agent.py`)
- **Executive Controller**: High-level planning, goal management, and decision coordination
- **Neural Integration**: Seamless coordination between symbolic and connectionist processing
- **State Management**: Real-time tracking of cognitive load, attention, and memory dynamics

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd human_ai_local
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Basic Usage

Run the demonstration:
```bash
python main.py
```

This will start an interactive session where you can:
- Chat with the cognitive agent
- Monitor cognitive state changes
- Observe memory consolidation
- View attention and fatigue dynamics

### Configuration

Create a `.env` file for environment-specific settings:
```bash
# AWS Configuration (optional)
AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=your-opensearch-endpoint

# Processing Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEBUG=false
LOG_LEVEL=INFO
```

## 🔬 Development Status

### ✅ Fully Functional Components
- **Core Cognitive Agent**: Complete orchestration of all cognitive processes
- **Memory Systems**: STM/LTM with ChromaDB vector storage and consolidation
- **Attention Mechanism**: Selective focus with fatigue modeling and resource allocation
- **Sensory Processing**: Complete sensory input handling with entropy and salience scoring
- **Dream Consolidation**: HDBSCAN clustering-based memory consolidation during sleep cycles
- **Neural Integration**: LSHN (Latent Structured Hopfield Networks) with dummy class fallbacks
- **DPAD Networks**: Dual-Path Attention Dynamics for temporal pattern processing
- **Meta-Cognition Engine**: Self-reflection and cognitive state monitoring
- **Neural Replay**: Dream-state memory replay and consolidation
- **Comprehensive Testing**: 25+ test files with passing integration tests

### 🚧 In Active Development
- **AWS Bedrock Integration**: Cloud-based LLM capabilities
- **Advanced Visualization**: Real-time cognitive state dashboards
- **Performance Optimization**: Memory efficiency and processing speed improvements

### 📋 Future Enhancements
- **Multi-modal Sensory Input**: Vision, audio, and text processing
- **Distributed Processing**: Multi-agent cognitive networks
- **Advanced Emotional Modeling**: Deeper affective state integration

## 💻 Technology Stack

### Core AI/ML Frameworks
- **PyTorch**: Neural network implementation and training
- **Transformers (Hugging Face)**: Embedding models and NLP capabilities
- **Sentence-Transformers**: Semantic embeddings for memory and attention
- **hopfield-layers**: Modern continuous Hopfield networks for associative memory

### Memory & Storage Systems
- **ChromaDB**: Vector database for long-term memory storage
- **HDBSCAN**: Hierarchical clustering for dream consolidation
- **NumPy**: Numerical computations and matrix operations
- **Pandas**: Data manipulation and analysis

### Neural Architecture Components
- **Custom LSHN**: Latent Structured Hopfield Networks implementation
- **DPAD Networks**: Dual-Path Attention Dynamics for temporal processing
- **Attention Mechanisms**: Multi-head attention with fatigue modeling
- **Entropy Scoring**: Information-theoretic sensory processing

### Development & Testing
- **pytest**: Comprehensive test suite (25+ test files)
- **asyncio**: Asynchronous processing for cognitive operations
- **logging**: Detailed system monitoring and debugging
- **dataclasses**: Clean, type-safe data structures

### Cloud Integration (Optional)
- **AWS Bedrock**: Cloud-based LLM integration
- **OpenSearch**: Distributed vector search capabilities

## 🧪 Testing Infrastructure

### Comprehensive Test Suite (25+ Test Files)
```bash
# Run all tests
pytest tests/ -v

# Unit tests for individual components
pytest tests/unit/ -v

# Integration tests for system interactions
pytest tests/integration/ -v

# Specific component testing
pytest tests/integration/test_dpad_integration_fixed.py -v
pytest tests/integration/test_lshn_integration.py -v
pytest tests/integration/test_dream_consolidation_pipeline.py -v
```

### Test Coverage Areas
- **Neural Network Integration**: DPAD and LSHN network functionality
- **Memory System Operations**: STM/LTM storage, retrieval, and consolidation
- **Attention Mechanisms**: Focus allocation and fatigue modeling
- **Dream Processing**: Sleep-cycle memory consolidation pipelines
- **Sensory Processing**: Input filtering and salience scoring
- **Cognitive Agent**: Complete end-to-end system demonstrations
- **Error Handling**: Graceful degradation and recovery testing

### Testing Features
- **Mocked Dependencies**: Isolated component testing without external services
- **Integration Demos**: Complete system workflow validation
- **Performance Benchmarks**: Memory usage and processing speed metrics
- **Regression Testing**: Automated validation of existing functionality

## 🎪 Unique Features

### Revolutionary Cognitive Architecture
- **Complete Cognitive Loop**: Fully integrated perception → attention → memory → action cycle
- **Biologically-Inspired Processing**: Mimics human cognitive patterns with scientific accuracy
- **Explainable Intelligence**: Transparent cognitive processes with detailed state monitoring
- **Adaptive Learning**: Dynamic memory consolidation and pattern recognition

### Advanced Memory Systems
- **Multi-Tiered Memory**: STM, LTM, and consolidation with realistic decay curves
- **ChromaDB Integration**: Vector-based semantic memory storage and retrieval
- **Dream Consolidation**: HDBSCAN clustering for sleep-cycle memory processing
- **Episodic Memory Formation**: LSHN-based autobiographical memory creation
- **Forgetting Curves**: Sigmoid-based realistic memory decay patterns

### Sophisticated Neural Networks
- **Hopfield Networks**: Modern continuous associative memory with hopfield-layers
- **LSHN Implementation**: Latent Structured Hopfield Networks for complex pattern storage
- **DPAD Integration**: Dual-Path Attention Dynamics for temporal sequence processing
- **Attention Fatigue**: Resource-aware attention allocation with realistic limitations
- **Neural Replay**: Dream-state memory consolidation with synaptic replay

### Intelligent Sensory Processing
- **Entropy-Based Filtering**: Information-theoretic sensory input prioritization
- **Salience Scoring**: Dynamic importance weighting for attention allocation
- **Multi-Modal Ready**: Architecture prepared for vision, audio, and text integration
- **Adaptive Thresholds**: Self-adjusting sensitivity based on cognitive load

### Meta-Cognitive Capabilities
- **Self-Reflection Engine**: Continuous monitoring of cognitive state and performance
- **Emotional Memory Integration**: Valence-based priority weighting for memories
- **Cognitive State Tracking**: Real-time awareness of attention, fatigue, and memory load
- **Strategic Planning**: Goal-oriented behavior with executive function modeling

## 📚 Documentation

- **Architecture Guide**: `docs/architecture.md`
- **API Reference**: `docs/api/`
- **Development Guide**: `docs/development.md`
- **Research Papers**: `docs/research/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

### Core Technologies
- **PyTorch**: Foundation for neural network implementations
- **ChromaDB**: Vector database powering long-term memory systems
- **Hugging Face**: Transformers and sentence-transformers for embeddings
- **hopfield-layers**: Modern Hopfield networks for associative memory
- **HDBSCAN**: Clustering algorithms for dream consolidation

### Research Foundations
- **Cognitive Science**: Biologically-inspired architecture design
- **Neuroscience**: Memory consolidation and attention mechanisms
- **Information Theory**: Entropy-based sensory processing
- **Machine Learning**: Advanced neural architectures (LSHN, DPAD)

### Community & Development
- **Open Source Community**: Collaborative development and feedback
- **Scientific Research**: Latest findings in cognitive architectures
- **Cloud Infrastructure**: AWS integration for scalable deployment

---

**Note**: This is an active research and development project. The cognitive architecture is continuously evolving based on the latest research in cognitive science and artificial intelligence.
