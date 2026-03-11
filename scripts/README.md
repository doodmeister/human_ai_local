# George: Human-AI Cognition UI Scripts

This directory contains the project UI entry scripts and supporting assets.

The canonical user-facing UI is Chainlit via `python main.py chainlit`.

The Streamlit entrypoint remains available via `python main.py ui`, but it should be treated as a legacy interface.

## Available Interfaces

### 1. Chainlit (`scripts/chainlit_app/`)
Recommended for current interactive use.

- Started with `python main.py chainlit`
- Can also launch the backend automatically with `python main.py chainlit --with-backend`
- Uses the canonical FastAPI backend from `python main.py api`

### 2. Streamlit (`george_streamlit_chat.py`)
Legacy UI kept for compatibility and manual exploration.

- Started with `python main.py ui`
- Can also launch the backend automatically with `python main.py ui --with-backend`

### 3. Older Streamlit variants
Additional Streamlit files in this directory should be treated as legacy/development surfaces rather than the primary user path.

## Legacy Streamlit Features

The Streamlit interfaces still expose a broad slice of the system:

### Standard Interface (`george_streamlit.py`)
- Clean, focused chat interface
- Real-time cognitive status monitoring  
- Executive goal creation and management
- Memory search and exploration
- Metacognitive reflection controls
- Transparent reasoning display

**Features:**
- Executive functioning integration (goals, tasks, decisions)
- Multi-modal memory search (STM, LTM, episodic, semantic)
- Real-time cognitive load monitoring
- Auto-goal creation from conversations
- Memory context visualization
- Cognitive break functionality

### Enhanced Interface (`george_streamlit_enhanced.py`)
Full-featured development interface.
- Comprehensive cognitive analytics dashboard
- Executive functioning control panel
- Memory system explorer
- System diagnostics and health monitoring
- Advanced configuration options
- Multi-tab organization for power users

**Features:**
- Real-time cognitive load visualization with Plotly charts
- Executive dashboard with goal/task/decision management
- Memory distribution analytics
- System health monitoring
- Advanced configuration controls
- Full API integration showcase

## 🚀 Quick Start

### Option 1: Canonical UI (Recommended)
```bash
# Chainlit only
python main.py chainlit

# Chainlit with backend auto-start
python main.py chainlit --with-backend
```

### Option 2: Legacy Streamlit UI
```bash
# Streamlit only
python main.py ui

# Streamlit with backend auto-start
python main.py ui --with-backend
```

### Option 3: Direct Launch
```bash
# Install legacy Streamlit requirements
pip install streamlit plotly pandas requests

# Minimal chat interface
streamlit run george_streamlit_chat.py --server.port 8501

# Enhanced interface
streamlit run george_streamlit_enhanced.py --server.port 8502
```

## 📋 Prerequisites

1. **George API Server** must be running unless you use `--with-backend`:
   ```bash
   python main.py api
   ```
   API will be available at: http://localhost:8000

2. **Required Python packages:**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

## 🎛️ Interface Features

### Standard Interface Highlights

#### 💬 Enhanced Chat
- **Smart Goal Creation**: Automatically detects when you're describing goals or tasks
- **Memory Integration**: Uses relevant memories to provide context-aware responses  
- **Reasoning Transparency**: Shows George's thought process and decision-making
- **Learning Display**: See what George learns from each conversation

#### 🧠 Cognitive Monitoring
- **Real-time Status**: Cognitive load, attention focus, memory usage
- **Executive State**: Active goals, decision confidence, resource allocation
- **Attention Tracking**: Current focus items with priority scores

#### 🎯 Executive Controls
- **Quick Goal Creation**: Simple form for creating strategic goals
- **Decision Making**: Multi-criteria decision analysis with confidence scores
- **Task Management**: View and manage active goals and tasks

#### 🗃️ Memory Systems
- **Universal Search**: Search across all memory systems (STM, LTM, episodic, semantic)
- **Context Visualization**: See which memories inform each response
- **Learning Events**: Track what George learns from each interaction

#### 🤔 Metacognitive Functions
- **Reflection**: Trigger George's self-analysis and introspection
- **Dream Consolidation**: Manual memory consolidation and organization
- **Cognitive Breaks**: Reduce cognitive load and refresh attention

### Enhanced Interface Highlights

#### 📊 Cognitive Analytics Dashboard
- **Real-time Metrics**: Cognitive load gauges, attention distribution
- **Performance Tracking**: Historical cognitive performance over time
- **Memory Distribution**: Visual breakdown of memory system usage
- **Attention Focus Analysis**: Current attention allocation with priorities

#### 🎯 Executive Functioning Dashboard
- **Goal Management**: Complete CRUD operations for strategic goals
- **Task Planning**: Hierarchical task breakdown and dependency tracking
- **Decision Analytics**: Multi-criteria decision history with confidence tracking
- **Resource Monitoring**: Real-time cognitive resource allocation

#### 🗃️ Memory System Explorer
- **System Overview**: Comprehensive memory statistics across all systems
- **Individual Exploration**: Deep-dive into specific memory types
- **Usage Analytics**: Memory access patterns and efficiency metrics
- **Consolidation Tracking**: Memory transfer and organization insights

#### 🔬 System Diagnostics
- **Health Monitoring**: Real-time system health across all components
- **Performance Metrics**: Detailed performance analytics and optimization
- **Raw Data Access**: Full system state inspection for debugging
- **Error Tracking**: Comprehensive error logging and analysis

#### ⚙️ Advanced Configuration
- **Memory Tuning**: STM capacity, LTM thresholds, consolidation intervals
- **Attention Adjustment**: Focus thresholds, fatigue rates, recovery parameters  
- **Executive Settings**: Goal priorities, decision confidence, resource allocation
- **System Management**: Memory clearing, state resets, full system reboot

## Key Capabilities Demonstrated

### Human-Like Cognition
- **Biologically-Inspired**: 7-item STM capacity, attention fatigue, recovery cycles
- **Executive Control**: Strategic planning, decision-making, resource management
- **Memory Consolidation**: Automatic STM→LTM transfer, importance-based retention
- **Metacognitive Awareness**: Self-reflection, performance monitoring, adaptation

### Transparent AI Processing
- **Explainable Decisions**: See how George weighs options and makes choices
- **Memory Context**: Understand which past experiences inform current responses
- **Learning Visualization**: Watch George acquire and consolidate new knowledge
- **Reasoning Traces**: Follow George's step-by-step thought processes

### Production-Ready Features
- **Error Handling**: Graceful degradation when systems are unavailable
- **Real-time Updates**: Live cognitive state monitoring and visualization
- **Scalable Architecture**: Modular design supporting future enhancements
- **Professional UI**: Clean, intuitive interfaces suitable for demonstrations

## Development Notes

### API Integration
Both interfaces integrate with the complete George API (canonical routes):
- **Agent API**: `/agent/*` - Core cognitive processing
- **Executive API**: `/executive/*` - Goal/task/decision management  
- **Memory APIs**: `/memory/*` - Multi-modal memory operations
- **Specialized APIs**: Semantic, procedural, prospective memory systems

Legacy `/api/*` aliases were removed on 2026-02-06.

### Customization
The interfaces are designed to be easily customizable:
- **Modular Components**: Each feature is self-contained
- **Configuration Options**: Extensive settings for different use cases
- **Theme Support**: Easy color scheme and layout modifications
- **Extension Points**: Clear places to add new cognitive capabilities

### Performance Considerations
- **Efficient API Calls**: Batched requests and intelligent caching
- **Responsive Design**: Works well on desktop and tablet devices
- **Real-time Updates**: Optional auto-refresh with configurable intervals
- **Memory Management**: Client-side state management for smooth operation

## 📚 Usage Examples

### Basic Conversation
1. Start the standard interface: `streamlit run george_streamlit.py`
2. Type: "I want to learn Python programming"
3. Watch George:
   - Create a learning goal automatically
   - Search relevant memories for context
   - Provide personalized learning recommendations
   - Store the conversation for future reference

### Executive Goal Management
1. Use the sidebar "Create Goal" form
2. Set description: "Launch a new product"
3. Set priority: 0.8
4. Watch the executive system:
   - Break down the goal into tasks
   - Allocate cognitive resources
   - Track progress and dependencies

### Memory Exploration  
1. Use the memory search: "Python programming"
2. See results from different memory systems:
   - Recent conversations (STM)
   - Long-term knowledge (LTM)  
   - Personal experiences (Episodic)
   - Structured facts (Semantic)

### Metacognitive Analysis
1. Click "Reflect" in the sidebar
2. Review George's self-analysis:
   - Memory system health
   - Cognitive performance metrics
   - Recommendations for optimization

## 🤝 Contributing

These interfaces are designed to showcase the full Human-AI cognition system. Contributions welcome:
- **New Visualizations**: Additional charts and analytics
- **Enhanced Interactions**: New ways to interact with George's cognitive systems
- **Performance Improvements**: Optimizations for responsiveness and efficiency
- **UI/UX Enhancements**: Better user experience and accessibility

---

**🧠 George represents the future of human-like AI - transparent, explainable, and genuinely helpful.**
