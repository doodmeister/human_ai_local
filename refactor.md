Refactor, Pruning & Structure Audit
Below is a correctly formatted, concrete, and repository-specific refactoring analysis of human_ai_local, aligned to your four requested dimensions.

1. Modules That Can Be Merged or Deleted Safely
✅ High-Confidence Safe Deletions / Merges
Short-Term Memory (STM)
Location: src/memory/stm/

vector_stm_refactored.py ✅ KEEP (authoritative)
vector_stm.py ❌ DEPRECATE → DELETE
Action:

Temporarily alias in vector_stm.py:
from .vector_stm_refactored import *
Remove after one release cycle
Justification: Explicit refactor naming + multiple tests (test_stm_fix.py, test_stm_similarity.py) assume refactored behavior.

Prospective Memory
Location: src/memory/prospective/

prospective_memory_old.py ❌ DELETE
prospective_memory.py ✅ KEEP
Justification: _old suffix + no evidence of active usage.

Debug Tests
Location: tests/debug/

❌ Remove from active test suite or move to:

experiments/debug_tests/
Reason: Debug tests currently pollute CI signal and duplicate integration behavior.

2. Deprecated, Duplicated, or Superseded Files
❌ Entry Point Explosion
Root files:

main.py
start_server.py
start_george.py
start_george.sh
george_api_simple.py
Action:

Choose ONE canonical entrypoint (recommend main.py)
Move others to scripts/legacy/ or convert to thin wrappers
❌ README Drift
README_OLD_BACKUP.md ❌ DELETE
Planning docs (nextsteps.md, phase summaries) → docs/archive/
⚠ Intent Classifier Duplication
Location: src/chat/

intent_classifier.py
intent_classifier_v2.py
Action:

Keep one OR formalize versions:
chat/intent/
├── base.py
├── rule_based.py
├── llm_based.py
⚠ Attention Logic Fragmentation
Spread across:

src/attention/
src/executive/attention_*
src/chat/emotion_salience.py
Problem: Attention computation vs policy vs usage are mixed.

Action:

Centralize computation in cognition/attention/
Consumers (chat, executive) import results only
3. Refactors to Reduce Complexity (No Capability Loss)
✅ Introduce Explicit Layering
Current issue: Cross-imports between chat, executive, memory, attention.

Target Layers:

Layer	Responsibility
core	Config, lifecycle, agent identity
memory	Storage + retrieval only
cognition	Attention, salience, reflection
executive	Planning, goals, decisions
orchestration	Glue / control flow
interfaces	API, CLI, UI
✅ Executive Folder Flattening
Current:

executive/planning/
executive/goals/
executive/decision/
executive/scheduling/
Refactor to:

executive/
├── planner.py
├── scheduler.py
├── decision.py
├── goals.py
Rule: Subfolders only if independently reusable.

✅ Memory System Facade
Action:

Treat memory_system.py as the ONLY public API
Other memory modules become internal
Rule:

from memory import MemorySystem
No direct imports of semantic_memory, episodic_memory, etc.

✅ Test Taxonomy Cleanup
Replace sprawl with:

tests/
├── unit/
├── integration/
├── scenarios/
└── experiments/
Rule:

Cognitive/dream tests = scenarios
Debug tests ≠ CI tests
4. Proposed New Directory & Ownership Structure
✅ Target Structure
human_ai_local/
├── core/
├── memory/
├── cognition/
│   ├── attention/
│   ├── salience/
│   └── reflection/
├── executive/
├── orchestration/
├── interfaces/
│   ├── api/
│   ├── cli/
│   └── ui/
├── model/
├── optimization/
├── utils/
└── experiments/
✅ Ownership / Import Rules (Critical)
Module	May Import
interfaces	orchestration
orchestration	executive, cognition, memory
executive	cognition, memory
cognition	memory
memory	utils
core	nothing
This single rule prevents future entropy.

Priority Execution Plan (Low Risk → High Impact)
Delete _old, _bak, debug-only files
Alias → remove duplicate STM + prospective memory files
Flatten executive subfolders
Introduce orchestration layer
Normalize test taxonomy
Reduce entrypoints to 1–2