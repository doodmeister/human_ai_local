---
name: phase-implementation-audit
description: 'Audit a phase plan, roadmap, or checklist against the actual repository. Use for examining whether phase docs were really implemented, mapping claims to code/tests, identifying drift between docs and runtime, and recommending precise follow-up fixes.'
argument-hint: 'Audit a phase doc or implementation checklist against the current codebase and report what is true, false, stale, or incomplete.'
user-invocable: true
---

# Phase Implementation Audit

Use this skill when the task is to verify that a plan document or milestone summary matches the actual repository state.

## When To Use

- audit `phase*.md` files against code
- verify that roadmap claims were really implemented
- compare docs to current runtime behavior
- find drift between tests, docs, and source layout
- produce a truthfulness review before cleanup or refactoring

## Primary Sources Of Truth

Prefer these first:

1. `README.md`
2. `phase3.md`
3. `.github/copilot-instructions.md`
4. current files under `src/`
5. current files under `tests/`

Treat `docs/archive/` as historical evidence, not the current contract.

## Rules

- Do not assume a checklist item is implemented because the doc says so.
- Map every major claim to concrete files, code paths, or tests.
- Distinguish clearly between implemented, partially implemented, stale, and contradicted claims.
- If runtime behavior and docs disagree, trust the code and report the mismatch.
- Recommend the smallest corrective action set that restores truthfulness.

## Procedure

1. Read the target phase or roadmap document.
2. Extract concrete claims, checklists, and architecture statements.
3. Locate the corresponding implementation in `src/`, entrypoints, tests, and docs.
4. Verify whether the behavior exists and whether it is actually wired into the current runtime.
5. Check related docs and test lanes for drift.
6. Summarize findings by severity and truth status.
7. If asked to fix, prioritize source-of-truth alignment over cosmetic updates.

## Output Expectations

- list verified items
- list stale or contradicted items
- call out missing tests or runtime-path mismatches
- recommend concrete follow-up fixes in priority order
