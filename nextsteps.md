# Human-AI Cognition Framework: Current State & Implementation Plan

**Date:** December 12, 2025

---

## 📊 Project Health Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 1,030 |
| **Passing** | 945 (91.7%) |
| **Failing** | 74 |
| **Skipped** | 11 |
| **Chat Tests** | 59/59 ✅ |

---

## 🏗️ Architecture Overview

### Backend Layers (Fully Implemented)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GEORGE UI (Streamlit)                        │
│  Chat • Reminders • Goals • LLM Config • Dream/Reflection Triggers  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼────────────────────────────────────┐
│                         FastAPI Server                              │
│  /agent/chat • /executive/* • /agent/reminders • /reflect           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────────┐
    │                            │                                │
    ▼                            ▼                                ▼
┌────────────┐          ┌────────────────┐          ┌─────────────────┐
│   Chat     │          │   Executive    │          │     Memory      │
│  Service   │◄────────►│    System      │◄────────►│     System      │
│            │          │                │          │                 │
│ • Intent   │          │ • GoalManager  │          │ • STM (Vector)  │
│ • GoalDet. │          │ • GOAP Planner │          │ • LTM (Vector)  │
│ • Context  │          │ • CP-SAT Sched │          │ • Episodic      │
│ • Capture  │          │ • ML Learning  │          │ • Prospective   │
│ • Orchest. │          │ • A/B Testing  │          │ • Procedural    │
└────────────┘          └────────────────┘          │ • Semantic      │
                                                    └─────────────────┘
```

---

## ✅ What's Complete

### 1. Memory Systems (100%)
| Component | Location | Features |
|-----------|----------|----------|
| **VectorSTM** | `src/memory/stm/` | ChromaDB, 7-item capacity, activation/decay, LRU |
| **VectorLTM** | `src/memory/ltm/` | Semantic clusters, decay, health reports |
| **Episodic** | `src/memory/episodic/` | Session-based recall, temporal ordering |
| **Prospective** | `src/memory/prospective/` | Time-based reminders, vector search option |
| **Procedural** | `src/memory/procedural/` | Skills/routines storage |
| **Semantic** | `src/memory/semantic/` | Fact storage |
| **Consolidation** | `src/memory/consolidation/` | STM→LTM promotion with thresholds |

### 2. Executive Functions (100%)
| Component | Location | Features |
|-----------|----------|----------|
| **GoalManager** | `src/executive/goal_manager.py` | Hierarchical goals, priorities, status tracking |
| **GOAP Planner** | `src/executive/planning/` | A* search, 10 predefined actions, constraints, heuristics |
| **CP-SAT Scheduler** | `src/executive/scheduling/` | Precedence, resources, deadlines, cognitive load |
| **Dynamic Scheduler** | `src/executive/scheduling/` | Real-time adaptation, quality metrics, visualization |
| **ExecutiveSystem** | `src/executive/integration.py` | Goal→Decision→Plan→Schedule pipeline |
| **ML Learning** | `src/executive/learning/` | Outcome tracking, feature extraction, model training |
| **A/B Testing** | `src/executive/learning/` | Experiments, statistical analysis, strategy comparison |

### 3. Chat Integration (100%)
| Component | Location | Features |
|-----------|----------|----------|
| **ChatService** | `src/chat/chat_service.py` | Turn processing, context building, consolidation |
| **IntentClassifierV2** | `src/chat/intent_classifier_v2.py` | Multi-intent, context-aware, 7 intent types |
| **GoalDetector** | `src/chat/goal_detector.py` | NL→Goal extraction, deadline parsing |
| **ExecutiveOrchestrator** | `src/chat/executive_orchestrator.py` | Async goal execution, progress tracking |
| **MemoryCapture** | `src/chat/memory_capture.py` | Auto-extract facts/preferences from chat |

### 4. API Layer (90% - Some Not Mounted)
| Router | Status | Endpoints |
|--------|--------|-----------|
| `chat_endpoints.py` | ✅ Mounted | `/agent/chat`, `/agent/reminders/*`, `/agent/dream/*` |
| `executive_api.py` | ✅ Mounted | `/executive/goals/*`, `/executive/system/health` |
| `memory_api.py` | ❌ Not Mounted | `/memory/{system}/*` |
| `semantic_api.py` | ❌ Not Mounted | `/semantic/fact/*` |
| `prospective_api.py` | ❌ Not Mounted | `/prospective/*` |
| `procedural_api.py` | ❌ Not Mounted | `/procedure/*` |

### 5. George UI (85%)
| Feature | Status |
|---------|--------|
| Chat with memory retrieval | ✅ |
| Reminder CRUD | ✅ |
| Goal creation (sidebar) | ✅ |
| Goal execution (▶️ button) | ✅ |
| GOAP plan viewer | ✅ |
| Schedule viewer | ✅ |
| NL goal detection display | ✅ |
| LLM provider switching | ✅ |
| Dream cycle trigger | ✅ |
| Metacog reflection | ✅ |
| Memory browser | ❌ |
| Learning dashboard | ❌ |
| A/B testing UI | ❌ |

---

## ❌ Gaps & Issues

### 1. Test Failures (74)
Most failures are in:
- `test_stm_similarity.py` - VectorSTM signature issues
- `test_episodic_memory_features.py` - Missing summarization
- Various integration tests with async/import issues

### 2. API Routes Not Mounted
These routers exist but aren't included in `start_server.py`:
- Memory API (browse/search memories)
- Semantic API (fact management)
- Prospective API (advanced reminder features)
- Procedural API (skill management)

### 3. Missing UI Features
- **Memory Browser**: Can't view/search/delete individual memories
- **Learning Dashboard**: Can't see ML predictions, accuracy trends
- **A/B Testing Panel**: Can't create/view experiments
- **System Health Dashboard**: No real-time metrics visualization

### 4. Natural Language Gaps
- Chat-created goals don't have proper `success_criteria` for GOAP
- No way to say "check my goals" and get a list in chat
- No voice input support

---

## 🚀 Implementation Plan

### Phase 1: Stabilization (1-2 weeks)
**Goal:** Get to 95%+ test pass rate, mount all APIs

| Task | Effort | Priority |
|------|--------|----------|
| Fix VectorSTM test failures | 4h | P0 |
| Fix episodic memory test failures | 4h | P0 |
| Mount remaining API routers | 2h | P0 |
| Add API endpoint documentation | 2h | P1 |

### Phase 2: Memory Browser UI (1 week)
**Goal:** Let users browse and manage memories

| Task | Effort | Priority |
|------|--------|----------|
| Create memory browser page in Streamlit | 8h | P0 |
| Add search/filter capabilities | 4h | P1 |
| Add memory deletion | 2h | P1 |
| Add memory promotion/demotion | 4h | P2 |

### Phase 3: Enhanced NL Goal Creation (1 week)
**Goal:** Better goal extraction from natural language

| Task | Effort | Priority |
|------|--------|----------|
| Extract `success_criteria` from NL | 4h | P0 |
| Add "check my goals" chat response | 2h | P1 |
| Add goal modification via chat | 4h | P2 |
| Show execution progress in chat | 4h | P1 |

### Phase 4: Learning Dashboard (2 weeks)
**Goal:** Expose ML learning to users

| Task | Effort | Priority |
|------|--------|----------|
| Create outcome history viewer | 4h | P1 |
| Show accuracy trends chart | 6h | P1 |
| Display ML predictions in goal view | 4h | P2 |
| Add A/B experiment creation UI | 8h | P2 |
| Show experiment results/recommendations | 6h | P2 |

### Phase 5: Production Hardening (2 weeks)
**Goal:** Ready for real users

| Task | Effort | Priority |
|------|--------|----------|
| Add authentication | 8h | P0 |
| Add rate limiting | 4h | P1 |
| Add request validation | 4h | P1 |
| Add error tracking (Sentry) | 4h | P1 |
| Create deployment scripts | 8h | P0 |
| Performance optimization | 8h | P2 |

---

## 📋 Quick Reference: Key Files

### Start Points
- `start_server.py` - FastAPI server entry point
- `start_george.py` - Launches backend + Streamlit UI
- `scripts/george_streamlit_chat.py` - Main UI (1,211 lines)

### Core Systems
- `src/chat/chat_service.py` - Chat orchestration (1,760 lines)
- `src/executive/integration.py` - Executive pipeline
- `src/memory/memory_system.py` - Memory system facade

### Configuration
- `.github/copilot-instructions.md` - AI assistant instructions
- `src/core/config.py` - Runtime configuration
- `.env` - Environment variables (ChromaDB paths, etc.)

---

## 🔧 Quick Commands

```bash
# Start everything
python start_george.py

# Start just the API server
python start_server.py

# Run all tests
pytest -q

# Run chat tests only
pytest tests/test_chat_*.py -q

# Check test failures
pytest -q --tb=short 2>&1 | grep FAILED

# Verify syntax
python -m py_compile scripts/george_streamlit_chat.py
```

---

## 📝 Recent Changes (December 12, 2025)

1. **Added NL Goal Detection Display** - George UI now shows when a goal is auto-created from conversation
2. **Added Intent Classification Display** - Shows detected intent type and confidence
3. **Added Goal Creation Tips** - Collapsible tip section explaining natural language goal creation

---

## 🎯 Recommended Next Session

Start with **Phase 1: Stabilization** - fixing the 74 failing tests and mounting the remaining API routers. This provides a solid foundation for all subsequent work.

Command to see what's failing:
```bash
pytest -q --tb=no 2>&1 | grep FAILED | head -20
```
