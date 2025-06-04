# Human-AI Cognition Framework

A biologically-inspired cognitive architecture that simulates human-like memory, attention, and reasoning capabilities in AI systems.

## 🎯 Project Vision

This framework bridges the gap between traditional stateless AI systems and human-like cognitive processing by implementing a modular architecture that mimics essential aspects of human cognition including perception, attention, memory formation, consolidation, and meta-cognitive reflection.

## 🧠 Core Philosophy

- **Memory as Foundation**: Unlike traditional AI that processes inputs in isolation, this system builds context through persistent memory structures
- **Human-Like Processing**: Implements biological cognition patterns including attention mechanisms, memory decay, and sleep-like consolidation  
- **Explainable Intelligence**: All cognitive processes are transparent and traceable through the system

## 📁 Project Structure

```
human_ai_local/
├── src/                          # Main source code
│   ├── core/                     # Core cognitive architecture
│   │   ├── config.py            # Configuration management
│   │   └── cognitive_agent.py   # Main cognitive orchestrator
│   ├── memory/                   # Memory systems
│   │   ├── stm/                 # Short-term memory
│   │   ├── ltm/                 # Long-term memory  
│   │   ├── prospective/         # Future-oriented memory
│   │   ├── procedural/          # Skills and procedures
│   │   └── consolidation/       # Memory consolidation
│   ├── attention/               # Attention mechanisms
│   ├── processing/              # Cognitive processing layers
│   │   ├── sensory/            # Sensory input processing
│   │   ├── neural/             # Neural network components
│   │   ├── embeddings/         # Text embedding generation
│   │   └── clustering/         # Memory clustering algorithms
│   ├── executive/              # Executive functions
│   ├── interfaces/             # External interfaces
│   │   ├── aws/               # AWS service integration
│   │   ├── streamlit/         # Dashboard interface
│   │   └── api/               # REST API endpoints
│   └── utils/                  # Utility functions
├── tests/                      # Test suites
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── cognitive/             # Cognitive behavior tests
├── data/                       # Data storage
│   ├── memory_stores/         # Memory databases
│   ├── embeddings/            # Cached embeddings
│   ├── models/                # Trained models
│   └── exports/               # Data exports
├── docs/                       # Documentation
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
└── infrastructure/             # Infrastructure as Code
```

## 🧠 Cognitive Architecture Components

### Memory Systems
- **Short-Term Memory (STM)**: Volatile working memory with time-based decay
- **Long-Term Memory (LTM)**: Persistent vector database storage with semantic retrieval
- **Prospective Memory**: Future-oriented tasks and reminders
- **Procedural Memory**: Skills and learned procedures

### Cognitive Processing Layers  
- **Sensory Buffer**: Input preprocessing and filtering
- **Attention Mechanism**: Selective focus and resource allocation
- **Meta-Cognition Engine**: Self-reflection and memory management
- **Dream-State Processor**: Memory consolidation during "sleep" cycles

### Executive Functions
- **Cognitive Agent Orchestrator**: Central coordination of all cognitive processes
- **Executive Planner Module**: High-level goal setting and task decomposition

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

- ✅ **Functional**: Basic cognitive loop with configuration system
- ✅ **Functional**: Core cognitive agent orchestrator
- ✅ **Functional**: Attention and fatigue modeling
- ✅ **Functional**: Memory ID generation and basic utilities
- ✅ **Functional**: Hopfield Networks integration (hopfield-layers)
- 🚧 **In Progress**: Memory systems implementation (STM/LTM)
- 🚧 **In Progress**: Sensory processing and embedding generation
- 🚧 **In Progress**: Dream-state consolidation pipeline
- 📋 **Planned**: AWS Bedrock LLM integration
- 📋 **Planned**: Vector database memory storage
- 📋 **Planned**: Advanced meta-cognition features

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## 🎪 Unique Features

### Biologically-Inspired Design
- **Sleep Cycles**: Actual dream-state processing for memory consolidation
- **Forgetting Curves**: Realistic memory decay patterns with sigmoid-based modeling
- **Attention Fatigue**: Resource management mimicking human cognitive limitations
- **Meta-Cognition**: Self-awareness and reflection capabilities
- **Emotional Memory**: Valence-based priority weighting
- **Hippocampal Dynamics**: LSHN-based episodic memory formation with Hopfield Networks

### Advanced Neural Architecture
- **Hopfield Networks**: Continuous modern Hopfield layers for associative memory
- **DPAD Integration**: Dynamic Pattern-Attentive Dynamics for temporal processing
- **LSHN Memory**: Latent Structured Hopfield Networks for episodic memory formation

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

- Inspired by biological cognitive science research
- Built on modern ML/AI frameworks (PyTorch, Transformers, etc.)
- Leverages cloud infrastructure (AWS Bedrock, OpenSearch)
- Community contributions and feedback

---

**Note**: This is an active research and development project. The cognitive architecture is continuously evolving based on the latest research in cognitive science and artificial intelligence.
