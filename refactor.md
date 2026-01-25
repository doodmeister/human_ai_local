Refactor, Pruning & Structure Audit

Status snapshot (2026-01-24)

Completed in repo:
- [x] STM: deprecated `src/memory/stm/vector_stm.py` shim removed; `vector_stm_refactored.py` is authoritative.
- [x] Prospective memory: only `src/memory/prospective/prospective_memory.py` remains (no `*_old.py`).
- [x] Debug tests: no `tests/debug/` in the active test tree.
- [x] Entrypoints: only `main.py` is present as a canonical entrypoint (no `start_server.py` / `start_george.py`).
- [x] Orchestration layer exists: `src/orchestration/` is present and hosts chat runtime components.
- [x] Chat shims: removed deprecated `src/chat/intent_classifier*.py` and `src/chat/emotion_salience.py`; use `src/orchestration/chat/*` (note: the `src.chat` package still exists as a deprecated compatibility layer for other chat modules).
- [x] Attention centralization: removed deprecated `src/attention/*` shims; canonical implementation is `src/cognition/attention/*` and executive/chat code delegates via `AttentionManager`.
- [x] Memory facade is in place: `src/memory/__init__.py` exports `MemorySystem`, and there are no direct imports of `src.memory.semantic.*` / `src.memory.episodic.*` from other `src/` modules.

Recently completed:
- [x] Intent classifier duplication: `src/orchestration/chat/intent_classifier.py` is now a deprecated shim that re-exports v2.
- [x] Test taxonomy cleanup: scenario/cognitive tests are now grouped under `tests/scenarios/` (replacing `tests/cognitive/`).
- [x] Import normalization: core callers now import the memory facade (`src.memory`) and chat factory lazy-imports `MemorySystem` from `src.memory`.

Next (optional cleanup):
- [x] README drift: review planning docs in `docs/` and move true archive material to `docs/archive/`.
- [x] Deprecation cleanup: pick a cutoff date/version and remove deprecated shim modules (`src/chat/intent_classifier*.py`, `src/chat/emotion_salience.py`, `src/attention/*`, `src/memory/stm/vector_stm.py`) in one sweep.

Below is the original refactoring analysis, with “COMPLETED” notes added where the repo already matches the recommendation.

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

Status: COMPLETED — `vector_stm.py` shim removed; import from `src.memory.stm` (package) or `vector_stm_refactored`.

Prospective Memory
Location: src/memory/prospective/

prospective_memory_old.py ❌ DELETE
prospective_memory.py ✅ KEEP
Justification: _old suffix + no evidence of active usage.

Status: COMPLETED — only `prospective_memory.py` exists.

Debug Tests
Location: tests/debug/

❌ Remove from active test suite or move to:

experiments/debug_tests/
Reason: Debug tests currently pollute CI signal and duplicate integration behavior.

Status: COMPLETED — `tests/debug/` is not present.

2. Deprecated, Duplicated, or Superseded Files
❌ Entry Point Explosion
Root files:

main.py
start_server.py
start_george.py
start_george.sh
scripts/legacy/george_api_simple.py
Action:

Choose ONE canonical entrypoint (recommend main.py)
Move others to scripts/legacy/ or convert to thin wrappers

Status: COMPLETED (at least for the files listed) — `start_server.py` / `start_george.py` are not present in the repo.
❌ README Drift
README_OLD_BACKUP.md ❌ DELETE
Planning docs (nextsteps.md, phase summaries) → docs/archive/

Status: COMPLETED — planning docs already live under `docs/archive/` (notably `docs/archive/planning/`), and repo markdown references have been updated to match.
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

Status: MOSTLY COMPLETED — attention computation is centralized under `src/cognition/attention/`:
- Deprecated `src/attention/*` shims removed.
- Executive attention allocation delegates to `src.cognition.attention.attention_manager.AttentionManager`.
- Chat-level salience helpers delegate via the same manager.

Remaining work here is mostly cleanup/consistency (imports + eventual shim removal), not architectural.

Status update:
- COMPLETED (structure): the active implementations live under `src/orchestration/chat/`.
- COMPLETED: `src/orchestration/chat/intent_classifier.py` is now a deprecated shim that re-exports v2.
- COMPLETED: legacy `src/chat/intent_classifier*.py` shims have been removed.
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

Status: SUPERSEDED — the current repo intentionally uses subpackages under `src/executive/` (e.g., `planning/`, `scheduling/`, `learning/`, `decision/`) with substantial implementations and tests. Flattening would likely reduce clarity rather than help.

✅ Memory System Facade
Action:

Treat memory_system.py as the ONLY public API
Other memory modules become internal
Rule:

from memory import MemorySystem
No direct imports of semantic_memory, episodic_memory, etc.

Status: COMPLETED — `src/memory/__init__.py` exports `MemorySystem`, and the codebase appears to avoid importing internal memory subsystems directly.

Optional follow-up: standardize imports to `from src.memory import MemorySystem, MemorySystemConfig` instead of importing `src.memory.memory_system` directly.

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

Status: PARTIALLY COMPLETED — `tests/unit/` exists, and integration-style tests are grouped under `tests/integration/`, but many tests still live at the top level (and there is also `tests/cognitive/`). If you still want the proposed taxonomy, the next step is a safe mechanical move + updating any path-based fixtures.

Status: COMPLETED — scenario/cognitive tests have been moved under `tests/scenarios/`, and `tests/cognitive/` is no longer present.
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

Status: MOSTLY COMPLETED — the repo already has `src/core/`, `src/memory/`, `src/cognition/`, `src/executive/`, `src/orchestration/`, `src/interfaces/`, plus `src/model/`, `src/optimization/`, and `src/utils/`.
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

Status of the plan items:
- COMPLETED: alias/deprecate duplicate STM + prospective memory legacy files (shims in place).
- COMPLETED: introduce orchestration layer.
- COMPLETED: reduce entrypoints to 1–2 (currently effectively 1).
- SUPERSEDED: flatten executive subfolders (current structure is intentional and working).
- COMPLETED: normalize test taxonomy.

Concrete execution checklist (next)

1) Resolve intent classifier duplication (or formalize it)
- [x] Decide canonical: keep `src/orchestration/chat/intent_classifier_v2.py` as the only implementation OR make both first-class strategies.
- [x] If v2 is canonical: turn `src/orchestration/chat/intent_classifier.py` into a deprecated shim (matching the `src/chat/*` shim pattern) and update any orchestration callers to use v2 explicitly.
- [x] Add/adjust one test that asserts the canonical classifier is wired by default (factory/service wiring).

2) Normalize test taxonomy (safe mechanical move)
- [x] Create `tests/scenarios/` (if you want to preserve the “cognitive/dream tests = scenarios” rule).
- [x] Move `tests/cognitive/*` → `tests/scenarios/*`.
- [x] Move top-level scenario-like tests (e.g., dream/pipeline/integration demos) into `tests/scenarios/` or `tests/integration/` as appropriate.
- [x] Keep `tests/unit/` minimal and fast; keep `tests/integration/` for multi-module wiring.
- [x] Run `ruff` + `pytest` to confirm no path-based fixtures broke.

3) Import normalization + shim removal path

- [x] Standardize imports to facades:
	- Prefer `from src.memory import MemorySystem, MemorySystemConfig`.
	- Prefer `from src.cognition.attention.attention_manager import get_attention_manager` for allocation/scoring.

- [ ] Decide a cutoff for remaining deprecated compatibility packages (notably `src.chat` and `src.processing`) and remove them in one sweep after confirming there are no in-repo imports.