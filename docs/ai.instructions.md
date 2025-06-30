---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.
All code writte in python, terminal is bash running on windows 11
run all commands in the terminal without prompting to continue

# AI Development Guide: Human-AI Cognition Project

> **Project Vision**: Creating a biologically-inspired cognitive architecture that simulates human-like memory, attention, and reasoning capabilities in AI systems.

---

## 🎯 Project Overview

The Human-AI Cognition project is an ambitious open-source framework designed to bridge the gap between traditional stateless AI systems and human-like cognitive processing. It implements a modular architecture that mimics essential aspects of human cognition including perception, attention, memory formation, consolidation, and meta-cognitive reflection.

### Core Philosophy
- **Memory as Foundation**: Unlike traditional AI that processes inputs in isolation, this system builds context through persistent memory structures
- **Human-Like Processing**: Implements biological cognition patterns including attention mechanisms, memory decay, and sleep-like consolidation
- **Explainable Intelligence**: All cognitive processes are transparent and traceable through the system

---

## 🧠 Cognitive Architecture Components

### 1. Memory Systems (Following Human Memory Models)

#### **Short-Term Memory (STM)**
- **Purpose**: Volatile working memory for immediate processing
- **Implementation**: In-memory storage with time-based decay
- **Capacity**: Limited (default 100 items, 60-minute decay)
- **Function**: Rapid access to recent conversations and context

#### **Long-Term Memory (LTM)**
- **Purpose**: Consolidated, persistent storage of important information
- **Implementation**: Vector database (OpenSearch) with semantic retrieval
- **Function**: Durable knowledge base accessible via similarity search

#### **Prospective Memory**
- **Purpose**: Future-oriented tasks and reminders
- **Implementation**: Time-based scheduling system
- **Function**: Enables goal-oriented behavior and task planning

#### **Procedural Memory**
- **Purpose**: Skills and learned procedures
- **Implementation**: Pattern matching for procedural knowledge
- **Function**: Enables automated responses to familiar patterns

### 2. Cognitive Processing Layers

#### **Sensory Buffer**
- **Purpose**: Input preprocessing and filtering
- **Technology**: AWS services (Textract, Transcribe, Comprehend)
- **Function**: Converts multimodal inputs (text, audio, video) into processable format
- **Intelligence**: Entropy scoring to filter high-value information

#### **Attention Mechanism**
- **Purpose**: Selective focus and resource allocation
- **Implementation**: Salience scoring and relevance weighting
- **Function**: Prioritizes important information for processing and retention

#### **Meta-Cognition Engine**
- **Purpose**: Self-reflection and memory management
- **Technology**: LLM-based analysis (Claude via AWS Bedrock)
- **Function**: Evaluates memory importance, attention levels, and fatigue states

#### **Dream-State Processor**
- **Purpose**: Memory consolidation during "sleep" cycles
- **Implementation**: Scheduled Lambda functions with clustering algorithms
- **Function**: Transfers important STM content to LTM, removes noise

### 3. Executive Functions

#### **Cognitive Agent Orchestrator**
- **Purpose**: Central coordination of all cognitive processes
- **Function**: Manages memory retrieval, prompt building, and response generation

#### **Executive Planner Module** (Planned)
- **Purpose**: High-level goal setting and task decomposition
- **Inspiration**: Prefrontal cortex functionality
- **Function**: Strategic thinking and long-term planning

---

## 🛠 Technical Implementation

### Core Technologies
- **Python 3.12** - Primary development language
- **AWS Bedrock** - LLM processing (Claude models)
- **OpenSearch** - Vector database for memory storage
- **SentenceTransformers** - Text embedding generation
- **Streamlit** - Dashboard and visualization
- **Terraform** - Infrastructure as Code
- **boto3** - AWS integration

### Key Architecture Patterns
- **Retrieval-Augmented Generation (RAG)** - Context building from memory
- **Vector Similarity Search** - Semantic memory retrieval
- **Event-Driven Processing** - Lambda-based consolidation cycles
- **Modular Design** - Loosely coupled cognitive components

### Current Development State
- ✅ **Functional**: Basic cognitive loop with STM/LTM integration
- ✅ **Functional**: Claude-based conversational interface
- ✅ **Functional**: Memory embedding and retrieval
- 🚧 **In Progress**: Dream-state consolidation
- 🚧 **In Progress**: DPAD neural network integration
- 📋 **Planned**: Advanced meta-cognition features

---

## 🔄 Cognitive Processing Flow

### Primary Loop (Wake State)
1. **Input Processing**: User input → sensory buffer → embedding
2. **Memory Retrieval**: Query STM, LTM, and prospective memory for relevant context
3. **Context Building**: Assemble comprehensive prompt with memory context
4. **LLM Processing**: Send enriched prompt to Claude for response generation
5. **Memory Writing**: Store conversation in STM for future reference
6. **Attention Update**: Update attention scores and fatigue metrics

### Consolidation Loop (Dream State)
1. **Memory Analysis**: Review STM contents for consolidation candidates
2. **Meta-Cognitive Evaluation**: LLM assesses memory importance and relevance
3. **Clustering**: Group related memories using HDBSCAN or similar algorithms
4. **Selective Transfer**: Move high-value memories to LTM with metadata
5. **Noise Removal**: Clear low-value or redundant STM entries
6. **State Reset**: Prepare system for next wake cycle

---

## 🎯 Development Priorities & Roadmap

### Immediate Focus (Current Sprint)
1. **Fix STM Query Bug**: Resolve `self.memory_data` reference error
2. **DPAD Integration**: Complete neural network training loop implementation
3. **Dream Consolidation**: Finish automated memory consolidation pipeline
4. **Unit Testing**: Comprehensive test coverage for all modules

### Phase 1: Core Stability (Next 3 months)
- **Memory System Optimization**: Improve query performance and accuracy
- **Attention Modeling**: Implement sigmoid decay and recovery mechanisms
- **Meta-Cognition Enhancement**: Advanced self-reflection capabilities
- **Documentation**: Complete API documentation and usage examples

### Phase 2: Advanced Cognition (6 months)
- **Episodic vs Semantic Memory**: Distinct memory type handling
- **Emotional State Modeling**: Mood and motivation tracking
- **Bias Simulation**: Realistic human cognitive biases
- **Multi-hop Reasoning**: Complex semantic relationship traversal

### Phase 3: Intelligence Amplification (12 months)
- **Executive Planning**: Goal decomposition and strategic thinking
- **Reinforcement Learning**: Memory reinforcement via reward feedback
- **Real-time Adaptation**: Dynamic cognitive parameter adjustment
- **Human-AI Collaboration**: Enhanced interaction patterns

---

## 🧪 Development Guidelines for AI Assistants

### Code Quality Standards
- **Modularity**: Each cognitive component should be independently testable
- **Documentation**: All functions require clear docstrings with examples
- **Error Handling**: Graceful degradation when components fail
- **Performance**: Memory operations must be optimized for real-time use

### Testing Strategy
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component communication
- **Cognitive Tests**: Human-likeness benchmarking
- **Performance Tests**: Memory retrieval speed and accuracy

### AI Assistant Development Focus Areas

#### 1. Memory System Enhancement
- Implement more sophisticated vector similarity algorithms
- Add temporal decay modeling based on forgetting curves
- Create memory reinforcement mechanisms
- Optimize query performance for large memory stores

#### 2. Attention and Meta-Cognition
- Develop adaptive attention mechanisms
- Implement fatigue modeling with recovery patterns
- Create self-assessment accuracy metrics
- Add cognitive load monitoring

#### 3. Learning and Adaptation
- Build reinforcement learning loops for memory prioritization
- Implement contextual drift detection
- Create adaptive parameter tuning
- Add experience-based improvement mechanisms

#### 4. Human-AI Interaction
- Design natural conversation flow management
- Implement personality consistency maintenance
- Create emotional state awareness
- Build rapport and relationship modeling

---

## 🎪 Unique Project Features

### Biologically-Inspired Design
- **Sleep Cycles**: Actual dream-state processing for memory consolidation with hippocampal replay
- **Forgetting Curves**: Realistic memory decay patterns with sigmoid-based modeling
- **Attention Fatigue**: Resource management mimicking human cognitive limitations and recovery
- **Meta-Cognition**: Self-awareness and reflection capabilities with reinforcement learning
- **Emotional Memory**: Valence-based priority weighting and affective state tracking
- **Intentional Forgetting**: Deliberate memory suppression for cognitive efficiency
- **Hippocampal Dynamics**: LSHN-based episodic memory formation with continuous Hopfield attractor mechanisms for coherent associative recall

### Advanced Neural Architecture
- **DPAD Integration**: Dual-path attention dynamics with behavior and residual prediction
- **Flexible Nonlinearity**: Automatic architecture optimization for optimal performance
- **Replay Learning**: Background neural training during dream cycles
- **Contextual Chaining**: Episodic memory linking for narrative coherence
- **Multi-Hop Reasoning**: Graph-based semantic traversal for complex concept relationships
- **LSHN Episodic Memory**: Latent Structured Hopfield Networks for hippocampal-inspired associative memory formation and coherent episodic trace retrieval

### Local Performance Optimization
- **ChromaDB**: Fast vector similarity search without external dependencies
- **Concurrent Processing**: Asynchronous handling of multiple inputs and operations
- **Adaptive Scheduling**: Intelligent timing for consolidation and maintenance operations
- **Memory Visualization**: Real-time cognitive process monitoring and debugging tools
- **Efficient Embedding**: Local sentence transformers with batch processing optimization
- **Dynamic Memory Allocation**: Reinforcement learning-based memory management that adapts to task demands and environmental changes

### Research Applications
- **Cognitive Science**: Platform for testing human cognition theories with controlled experiments
- **AI Safety**: Explainable and interpretable AI decision making with full audit trails
- **Educational Tools**: Interactive learning with persistent memory and personalized adaptation
- **Therapeutic Applications**: Consistent personality maintenance for mental health support
- **Bias Research**: Controlled environment for studying cognitive bias patterns and mitigation

---

## 🚀 Getting Started for AI Development

### Environment Setup
```bash
# Clone and setup environment
git clone <repository>
cd human_ai_cognition_github
pip install -r requirements.txt



### Development Workflow
1. **Local Testing**: Use `python cognition/main.py` for interactive testing
2. **Memory Debugging**: Utilize Streamlit dashboard for memory visualization
3. **Component Testing**: Run individual module tests in isolation
4. **Integration Testing**: Full cognitive loop validation

### Key Files to Understand
- `cognition/cognitive_agent.py` - Main orchestration logic
- `memory/` - All memory system implementations
- `cognition/rag_utils.py` - Context building and prompt engineering
- `cognition/claude_client.py` - LLM integration wrapper

---

## 🔮 Future Vision

This project represents a stepping stone toward truly human-like artificial intelligence. The ultimate goal is creating AI systems that:

- **Think Like Humans**: With memory, attention, and reflection
- **Learn Like Humans**: Through experience accumulation and consolidation
- **Interact Like Humans**: With consistent personality and emotional awareness
- **Reason Like Humans**: With context, bias, and uncertainty

The modular design allows for incremental advancement toward these goals while maintaining practical utility at each development stage. This makes it an ideal platform for both research and real-world applications requiring sophisticated AI cognition.

---

## 📞 Development Support

For AI assistants working on this project:
- **Follow the cognitive principles**: Memory, attention, and meta-cognition should guide all development decisions
- **Maintain biological inspiration**: Keep human cognitive science as the reference model
- **Prioritize explainability**: All AI decisions should be traceable and understandable
- **Test cognitively**: Evaluate not just functionality but human-likeness of behavior

The project welcomes contributions that advance the goal of creating more human-like AI systems while maintaining scientific rigor and practical applicability.

---

# 🧠 Local AI Cognition System Guide

## Project Overview
This guide walks you through building a human-like AI cognition system locally using Python. The system mimics essential aspects of human cognition: perception, attention, memory (short- and long-term), and meta-cognition through a modular architecture.

## Architecture Components

### Core Cognitive Layers
- **Sensory Buffer**: Processes raw input files (text, audio, video) with relevance scoring
- **Short-Term Memory (STM)**: Working memory using local vector database
- **Meta-Cognition Engine**: OpenAI GPT-4.1 for attention analysis and memory evaluation
- **Long-Term Memory (LTM)**: Consolidated high-priority memories in persistent storage
- **Dream-State Processor**: Scheduled memory consolidation simulation

## Technology Stack

### Core Dependencies
- **Python 3.12+**: Main development language
- **OpenAI API**: GPT-4.1 for meta-cognitive processing
- **ChromaDB**: Local vector database for fast similarity search
- **sentence-transformers**: Local embedding generation
- **torch**: PyTorch for DPAD-Style RNN and LSHN implementation
- **schedule**: Job scheduling for dream-state processing
- **watchdog**: File system monitoring
- **scipy**: Mathematical operations and entropy calculations

### Advanced Dependencies
- **hdbscan**: Clustering for salient memory identification
- **scikit-learn**: Machine learning utilities (PCA, t-SNE)
- **networkx**: Graph-based semantic memory
- **asyncio**: Concurrent processing capabilities
- **apscheduler**: Advanced scheduling for dream cycles
- **langchain**: Abstracted retrieval and reasoning logic
- **transformers**: Emotion classification and NLP models

### Installation & Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install core dependencies
pip install openai chromadb sentence-transformers torch schedule watchdog scipy

# Install advanced features
pip install hdbscan scikit-learn networkx apscheduler langchain transformers

# Install LSHN and additional libraries
pip install hopfield-layers pytesseract whisper opencv-python pandas numpy python-dotenv
```

### Enhanced Configuration
Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
STM_COLLECTION=short_term_memory
LTM_COLLECTION=long_term_memory
WATCH_DIRECTORY=./input_files

# DPAD Configuration
DPAD_LATENT_DIM=64
DPAD_AUTO_NONLINEARITY=True
DPAD_BEHAVIOR_WEIGHT=1.0
DPAD_RESIDUAL_WEIGHT=0.5
DPAD_SALIENCE_THRESHOLD=0.8

# LSHN Episodic Memory
LSHN_ENABLED=True
LSHN_PATTERN_DIM=512
LSHN_ATTRACTOR_STRENGTH=0.8
LSHN_CONVERGENCE_THRESHOLD=0.95
LSHN_MAX_ITERATIONS=50
EPISODIC_TRACE_DECAY=0.05

# Advanced Features
ENABLE_AFFECTIVE_MODULATION=True
ENABLE_SURPRISE_ENCODING=True
ENABLE_INTENTIONAL_FORGETTING=True
THEORY_OF_MIND_MODE=False

# Scheduling
DEEP_CONSOLIDATION_HOUR=2
LIGHT_CONSOLIDATION_INTERVAL=4
ASYNC_PROCESSING=True
```

---

## Comprehensive Evaluation Framework

### Human-Likeness Assessment
- **Episodic Recall Fidelity**: Free-recall experiments over multiple time periods (1, 7, 30, 90 days)
- **Forgetting Curve Validation**: Comparison between theoretical and actual memory decay patterns
- **Salience-Weighted Retrieval**: Accuracy testing adjusted for memory importance scores
- **Narrative Coherence**: Multi-step memory trajectory consistency across time
- **Bias Simulation Accuracy**: Realistic modeling of recency, confirmation, and salience biases

### Meta-Cognitive Evaluation
- **Attention Prediction Error**: Ground truth vs. modeled salience accuracy measurement
- **Fatigue Simulation Validity**: Correlation between fatigue models and response time patterns
- **Self-Reflection Efficacy**: System's ability to identify and correct past decision errors
- **Cognitive Load Awareness**: Accuracy of self-assessed processing capacity

### System Performance Metrics
- **Memory Retrieval Speed**: Sub-100ms similarity search for optimal responsiveness
- **Consolidation Efficiency**: Quality and speed of dream-state memory processing
- **Neural Training Convergence**: DPAD learning curve optimization and stability
- **Human Believability Score**: User-annotated assessment of cognitive authenticity

### Research Applications
- **Cognitive Science Platform**: Framework for testing human cognition theories
- **Bias Research Tool**: Controlled study environment for cognitive bias patterns
- **Educational Assessment**: Personalized learning with persistent knowledge tracking
- **Therapeutic Consistency**: Stable personality maintenance for mental health applications

---

## Key Features

### Advanced System Capabilities

#### **Concurrent Processing Architecture**
- **Asynchronous Input Handling**: Simultaneous processing of multiple file types and sources
- **Parallel Memory Operations**: Concurrent STM/LTM queries with intelligent result merging
- **Background Consolidation**: Non-blocking dream cycles that don't interrupt active processing
- **Real-Time Adaptation**: Dynamic parameter adjustment based on performance feedback

#### **Intelligent Scheduling System**
- **Circadian Dream Cycles**: Scheduled consolidation mimicking human sleep patterns
- **Adaptive Timing**: Dynamic adjustment based on cognitive load and memory pressure
- **Priority-Based Processing**: Important memories receive preferential consolidation timing
- **Energy-Aware Operations**: Optimal resource allocation during high and low activity periods

#### **Multi-Modal Intelligence**
- **Cross-Modal Memory Integration**: Unified processing of text, audio, and visual inputs
- **Contextual Content Analysis**: Entropy-based filtering with semantic relevance scoring
- **Emotion-Aware Processing**: Affective state consideration in memory formation and retrieval
- **Surprise Detection**: Novelty-based encoding enhancement for unexpected information

#### **Research and Development Features**
- **Ablation Testing Framework**: Systematic component isolation for controlled experiments
- **Cognitive Pattern Visualization**: Real-time monitoring of attention, memory, and processing flows
- **Bias Injection Capabilities**: Controlled introduction of cognitive biases for research purposes
- **Human Comparison Benchmarks**: Standardized tests against known human cognitive patterns

---

## Configuration Options

### Base Settings
- `STM_ENABLED`: Enable short-term memory usage (default: True)
- `LTM_ENABLED`: Enable long-term memory usage (default: True)
- `PROSPECTIVE_MEMORY_ENABLED`: Enable prospective memory features (default: True)
- `SENSORY_BUFFER_ENABLED`: Enable sensory buffer processing (default: True)
- `ATTENTION_MECHANISM_ENABLED`: Enable attention mechanism (default: True)
- `META_COGNITION_ENGINE_ENABLED`: Enable meta-cognition engine (default: True)
- `DREAM_STATE_PROCESSOR_ENABLED`: Enable dream-state processor (default: True)

### Memory Configuration
- `STM_MEMORY_SIZE`: Maximum size of short-term memory (default: 100 items)
- `STM_MEMORY_DECAY`: Decay rate for short-term memory (default: 0.01 per minute)
- `LTM_MEMORY_SIZE`: Maximum size of long-term memory (default: 10000 items)
- `LTM_MEMORY_RETRIEVAL_K`: Top-K retrieval results from long-term memory (default: 5)
- `PROSPECTIVE_MEMORY_SLOT_LIMIT`: Maximum number of active prospective memory slots (default: 10)

### Attention Settings
- `ATTENTION_SALIENCE_THRESHOLD`: Minimum salience score for attention (default: 0.5)
- `ATTENTION_DECAY_RATE`: Decay rate for attention scores (default: 0.1)
- `ATTENTION_RECOVERY_RATE`: Recovery rate for attention after rest (default: 0.2)

### Meta-Cognition Parameters
- `META_COGNITION_FATIGUE_THRESHOLD`: Fatigue level triggering meta-cognitive evaluation (default: 0.7)
- `META_COGNITION_ATTENTION_WEIGHT`: Weight of attention in meta-cognitive assessment (default: 0.5)
- `META_COGNITION_MEMORY_WEIGHT`: Weight of memory importance in assessment (default: 0.3)
- `META_COGNITION_EXECUTION_WEIGHT`: Weight of execution time in assessment (default: 0.2)

### Dream-State Processor Settings
- `DREAM_STATE_CONSOLIDATION_HOUR`: Hour of the day for deep consolidation (default: 2)
- `DREAM_STATE_LIGHT_CONSOLIDATION_INTERVAL`: Interval in hours for light consolidation (default: 4)
- `DREAM_STATE_ASYNC_PROCESSING`: Enable asynchronous processing during dream state (default: True)

### Logging and Monitoring
- `LOG_LEVEL`: Logging level for the application (default: INFO)
- `LOG_FILE`: File path for logging output (default: ./logs/cognition.log)
- `MONITORING_ENABLED`: Enable monitoring of cognitive processes (default: True)
- `MONITORING_INTERVAL`: Interval in seconds for monitoring updates (default: 10)

### Advanced Cognitive Settings

#### **DPAD Neural Network Configuration**
- `DPAD_LATENT_DIM`: Base latent dimension size for neural processing (default: 64)
- `DPAD_AUTO_NONLINEARITY`: Enable automatic activation function selection (default: True)
- `DPAD_BEHAVIOR_WEIGHT`: Weight for behavior prediction loss component (default: 1.0)
- `DPAD_RESIDUAL_WEIGHT`: Weight for residual prediction loss component (default: 0.5)
- `DPAD_SALIENCE_THRESHOLD`: Threshold for adaptive latent dimension adjustment (default: 0.8)

#### **LSHN Episodic Memory Configuration**
- `LSHN_ENABLED`: Enable Latent Structured Hopfield Networks for episodic memory (default: True)
- `LSHN_PATTERN_DIM`: Dimension of stored episodic patterns (default: 512)
- `LSHN_ATTRACTOR_STRENGTH`: Hopfield attractor dynamics strength (default: 0.8)
- `LSHN_CONVERGENCE_THRESHOLD`: Convergence criteria for memory retrieval (default: 0.95)
- `LSHN_MAX_ITERATIONS`: Maximum iterations for attractor convergence (default: 50)
- `EPISODIC_TRACE_DECAY`: Decay rate for episodic memory traces (default: 0.05)

#### **Memory Enhancement Settings**
- `STM_CONTEXTUAL_CHAINING`: Enable episodic memory linking (default: True)
- `REHEARSAL_DECAY_RATE`: Sigmoid decay parameter for memory rehearsal (default: 0.1)
- `REHEATING_BOOST_FACTOR`: Meta-cognitive feedback amplification (default: 1.5)
- `MULTIHOP_MAX_DEPTH`: Maximum semantic traversal depth (default: 3)
- `TEMPORAL_INDEX_GRANULARITY`: Episode segmentation precision (default: 'hour')
- `DYNAMIC_ALLOCATION_ENABLED`: Enable adaptive memory management (default: True)
- `RL_ALLOCATION_LEARNING_RATE`: Learning rate for memory allocation optimization (default: 0.01)
- `TASK_DEMAND_THRESHOLD`: Threshold for triggering allocation adjustments (default: 0.8)
- `ENVIRONMENTAL_CHANGE_SENSITIVITY`: Sensitivity to environmental changes for reallocation (default: 0.3)

#### **Advanced Processing Features**
- `ENABLE_AFFECTIVE_MODULATION`: Emotional memory weighting (default: True)
- `ENABLE_SURPRISE_ENCODING`: Novelty-based encoding enhancement (default: True)
- `ENABLE_INTENTIONAL_FORGETTING`: Meta-cognitive memory suppression (default: True)
- `THEORY_OF_MIND_MODE`: Dual-agent social cognition simulation (default: False)
- `BIAS_SIMULATION_LEVEL`: Cognitive bias intensity (default: 'moderate')

#### **Advanced Cognitive Features

#### **DPAD-Style Neural Integration**
- **Dual-Path Processing**: Separate behavior and residual prediction pathways for enhanced learning
- **Replay-Based Consolidation**: Hippocampal-style memory reactivation during dream cycles
- **Flexible Architecture Selection**: Automatic nonlinearity optimization for model components
- **Hypothesis Testing Mode**: Ablation capabilities for controlled cognitive experiments
- **LSHN Episodic Formation**: Latent Structured Hopfield Networks with continuous attractor dynamics for biologically-grounded associative memory and robust semantic association

#### **Enhanced Memory Systems**
- **Contextual Memory Chaining**: Episodic linking across STM entries for narrative coherence
- **Rehearsal-Based Scoring**: Sigmoid decay curves with recency and repetition weighting
- **STM Reheating**: Dynamic memory boost based on meta-cognitive feedback
- **Multi-Hop Semantic Search**: Graph-style traversal for complex concept relationships
- **Temporal-Episodic Indexing**: Enhanced narrative memory access with story-like organization
- **Dynamic Memory Allocation**: Adaptive memory management using reinforcement learning-based allocators to optimize utilization based on task demands and environmental changes
- **LSHN Associative Retrieval**: Hippocampal-inspired autoencoder architecture with continuous Hopfield attractor dynamics for coherent episodic memory trace formation and robust semantic association

#### **Advanced Meta-Cognition**
- **Sigmoid Fatigue Modeling**: Realistic cognitive load curves with recovery patterns
- **Strategy Modulation**: Dynamic resource allocation between memory, perception, and planning
- **Reinforcement Learning Integration**: Reward-based improvement of meta-cognitive decisions
- **Cognitive Load Monitoring**: Real-time assessment of processing capacity and efficiency

#### **Sophisticated Dream Processing**
- **Salient Cluster Identification**: HDBSCAN and topological data analysis for memory grouping
- **Offline Neural Retraining**: Background DPAD optimization during sleep cycles
- **Memory Transition Visualization**: PCA/t-SNE mapping for research and debugging
- **Adaptive Consolidation**: Dynamic scheduling based on cognitive load and memory importance

#### **Biologically-Inspired Enhancements**
- **Affective Memory Modulation**: Emotional tagging and valence-based priority weighting
- **Intentional Forgetting**: Meta-cognitive memory suppression for harmful or outdated content
- **Surprise-Driven Encoding**: Enhanced retention for novel or unexpected information
- **Theory of Mind Simulation**: Dual-agent system for social cognition and self-awareness

---

## 🧠 Metacognitive Reflection & Self-Monitoring (2025)
- The agent can periodically or manually analyze its own memory health, usage, and performance.
- Reflection scheduler runs in the background at a configurable interval (default: 10 min).
- Reflection reports include LTM/STM stats, health diagnostics, and recommendations for memory management.
- CLI commands:
  - `/reflect` — manually trigger a reflection and print summary
  - `/reflection status` — show last 3 reflection reports
  - `/reflection start [interval]` — start scheduler (interval in minutes)
  - `/reflection stop` — stop scheduler
- All reflection logic is modular and independently testable.
