# Documentation Index

This index separates active documentation from archived historical material.

Use the files in this directory for the current architecture contract. Use `archive/` only for implementation history, older plans, and legacy reference notes.

For the fastest current picture of the repo, start with the root runtime docs and then drill into the focused design docs in `docs/`.

## Current Docs

- `../README.md`: primary project overview, current runtime surface, API endpoints, and test lanes
- `../STARTUP_GUIDE.md`: canonical startup, ports, and runtime troubleshooting
- `../cognition.md`: metacognition architecture, rollout phases, and completion status
- `../phase3.md`: current runtime refactor plan and architecture status
- `UI_DEVELOPER_API_QUICKSTART.md`: current frontend and API integration reference
- `metacog_features.md`: chat-layer metacognitive snapshots, adaptive retrieval, and thresholds
- `executive_telemetry.md`: executive metrics and event wiring
- `memory_personality_roadmap.md`: current roadmap for memory and personality work
- `memory_personality_architecture.md`: target architecture for memory, identity, and personality
- `memory_personality_implementation_tickets.md`: phased implementation tickets for building the roadmap
- `memory_personality_issue_backlog.md`: issue-style backlog for tracking the implementation work
- `ai.instructions.md`: supplementary background for assistants working in the repo
- `cognitive_theory.md`: current cognition model and canonical turn loop
- `memory_taxonomy.md`: current memory ownership rules and constraints
- `goap_architecture.md`: primary GOAP design guide
- `goap_quick_reference.md`: GOAP API and usage cheat sheet
- `stm_decay_adaptive_activation.md`: current STM decay modes and activation-weight behavior
- `../scripts/generate_memory_scorecard.py`: deterministic memory and behavior scorecard generator; pass `--fail-on-gate` to use it as a review gate

## Current Runtime Docs

- `../README.md`: top-level overview of the live runtime and mounted API surface
- `../STARTUP_GUIDE.md`: operational startup and health-check steps
- `UI_DEVELOPER_API_QUICKSTART.md`: frontend-facing API usage and integration notes
- `metacog_features.md`: chat-specific metacognitive behavior and adaptive thresholds
- `executive_telemetry.md`: event bus and telemetry wiring for executive flows

## Recommended Reading Order

### For developers
1. `../README.md`
2. `../STARTUP_GUIDE.md`
3. `../cognition.md`
4. `../phase3.md`
5. `memory_personality_roadmap.md`
6. `memory_personality_architecture.md`

### For assistants
1. `ai.instructions.md`
2. `../README.md`
3. `../cognition.md`
4. `../phase3.md`
5. `memory_personality_architecture.md`
6. `memory_personality_implementation_tickets.md`

### For GOAP work
1. `goap_architecture.md`
2. `goap_quick_reference.md`

## Archive

`archive/` contains:

- week-by-week completion summaries
- phase snapshots and historical plans
- superseded root-level planning documents
- slice-era feature notes and implementation summaries
- legacy UI and local setup reference material

Representative archive files:

- `archive/executive_refactoring_plan.md`
- `archive/planning/roadmap.md`
- `archive/WEEK_15_COMPLETION_SUMMARY.md`
- `archive/PHASE_2_FINAL_COMPLETE.md`
- `archive/PHASE2_ARCHITECTURE.md`
- `archive/testplan.md`
- `archive/llm_provider_feature.md`
- `archive/goap_usage_examples.md`
- `archive/ui_showcase.md`

## Current Layout

```text
docs/
  README.md
  ai.instructions.md
  cognitive_theory.md
  executive_telemetry.md
  goap_architecture.md
  goap_quick_reference.md
  memory_personality_implementation_tickets.md
  memory_personality_issue_backlog.md
  memory_personality_architecture.md
  memory_personality_roadmap.md
  memory_taxonomy.md
  metacog_features.md
  stm_decay_adaptive_activation.md
  UI_DEVELOPER_API_QUICKSTART.md
  archive/
```

## Status

- The files listed under `Current Docs` are the active documentation set.
- Root runtime docs now live primarily in `../README.md`, `../STARTUP_GUIDE.md`, `../cognition.md`, and `../phase3.md`.
- Most other material has been moved under `archive/` to reduce ambiguity.
- If a document describes an older implementation slice or historical milestone, it belongs in `archive/` unless it is still referenced as current contract.

Last updated: April 11, 2026
