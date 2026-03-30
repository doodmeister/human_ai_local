---
name: docs-archive-reconciliation
description: 'Reconcile active docs versus historical docs in this repo. Use for docs cleanup, moving superseded files into docs/archive, fixing stale links, simplifying docs/README, and keeping current documentation aligned with README.md and phase3.md.'
argument-hint: 'Clean up docs, archive stale material, and reconcile the docs index with the current architecture contract.'
user-invocable: true
---

# Docs Archive Reconciliation

Use this skill when the task is to clean up documentation structure rather than to change runtime behavior.

## When To Use

- archive old or superseded docs
- determine whether a doc is current or historical
- simplify `docs/README.md`
- fix stale links after moving docs
- reconcile docs with `README.md` and `phase3.md`
- reduce ambiguity between current guidance and implementation history

## Current Sources Of Truth

Prefer these files first:

1. `README.md`
2. `phase3.md`
3. `docs/README.md`
4. `docs/memory_personality_roadmap.md`
5. `docs/memory_personality_architecture.md`

Treat `docs/archive/` as historical unless the task is explicitly about archaeology or migration.

## Rules

- Move stale material to `docs/archive/`; do not delete it unless explicitly requested.
- Keep the active docs set small and clearly current.
- Preserve docs that are still operationally useful even if they mention legacy systems.
- Update links and indexes in the same task when files move.
- Do not promote historical docs back into the active set without evidence they are current.

## Procedure

1. Inventory root-level docs and `docs/` files.
2. Classify each candidate as current, historical, or ambiguous.
3. Cross-check references from `README.md`, `phase3.md`, and `docs/README.md`.
4. Move only clearly historical or superseded docs into `docs/archive/`.
5. Update `docs/README.md` so it reflects the active set and archive structure.
6. Fix stale links created by the move.
7. Validate edited markdown files and confirm moved paths exist.

## Classification Heuristics

Strong archive signals:

- implementation snapshot tied to a specific week or phase
- document describes an original slice rather than current behavior
- file explicitly says legacy, historical, superseded, or reference-only
- file conflicts with `README.md` or `phase3.md`

Strong keep-current signals:

- operational startup, runtime, testing, or API guidance
- current architecture contracts or repo conventions
- active roadmap or target architecture docs
- still referenced as current by `README.md` or `phase3.md`

## Output Expectations

- summarize what moved and why
- call out any ambiguous files left in place
- mention any references intentionally left untouched because they are historical artifacts