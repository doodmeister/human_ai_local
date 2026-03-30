---
name: memory-personality-slice-planning
description: 'Turn the memory and personality roadmap into concrete implementation slices. Use for planning phased work from docs/memory_personality_roadmap.md and docs/memory_personality_architecture.md, defining sequence, boundaries, tests, and migration steps for memory, retrieval, self-model, and response-policy changes.'
argument-hint: 'Plan a concrete implementation slice for memory, retrieval, relationship memory, self-model, or personality-to-behavior wiring.'
user-invocable: true
---

# Memory Personality Slice Planning

Use this skill when the task is to turn the repo's forward-looking architecture into a buildable implementation plan.

## When To Use

- break the roadmap into implementable slices
- define the next phase for memory or personality work
- plan retrieval-planner, reranker, relationship-memory, or response-policy work
- turn architecture docs into tickets, migrations, and tests
- choose a low-risk next slice that improves the current runtime

## Primary Sources Of Truth

Prefer these first:

1. `docs/memory_personality_roadmap.md`
2. `docs/memory_personality_architecture.md`
3. `README.md`
4. `phase3.md`
5. current implementation under `src/memory/`, `src/orchestration/`, and `src/executive/`

## Rules

- plan from the current codebase, not from an imagined greenfield architecture
- prefer thin vertical slices over broad framework rewrites
- keep runtime composition stable while introducing new subsystems
- include validation, migration, and rollback considerations
- identify which docs must change with the code

## Procedure

1. Identify the target capability gap.
2. Find the nearest existing code seams that can absorb the change.
3. Define the smallest end-to-end slice that produces user-visible or architecture-visible progress.
4. Specify affected modules, new files, public interfaces, and migration constraints.
5. Define the tests and telemetry needed to prove the slice works.
6. Sequence immediate follow-up slices so the path remains coherent.

## Slice Template

For each proposed slice, include:

- goal
- why this slice now
- existing code to reuse
- new modules or interfaces
- code changes by area
- test plan
- docs updates
- risks and deferrals

## Output Expectations

- produce a sequenced slice plan, not just ideas
- keep each slice small enough to implement safely in the current repo
- call out dependencies on current runtime, memory facades, and active docs