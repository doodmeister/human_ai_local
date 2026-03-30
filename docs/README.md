# Documentation Index

This index separates active documentation from archived historical material.

Use the files in this directory for the current architecture contract. Use `archive/` only for implementation history, older plans, and legacy reference notes.

## Current Docs

- `../README.md`: primary project overview, startup paths, and test lanes
- `../phase3.md`: current architecture refactor plan and execution status
- `memory_personality_roadmap.md`: current roadmap for memory and personality work
- `memory_personality_architecture.md`: target architecture for memory, identity, and personality
- `memory_personality_implementation_tickets.md`: phased implementation tickets for building the roadmap
- `memory_personality_issue_backlog.md`: issue-style backlog for tracking the implementation work
- `UI_DEVELOPER_API_QUICKSTART.md`: current UI-facing executive API quick reference
- `ai.instructions.md`: supplementary background for assistants working in the repo
- `cognitive_theory.md`: current cognition model and canonical turn loop
- `memory_taxonomy.md`: current memory ownership rules and constraints
- `metacog_features.md`: metacognitive snapshots, adaptive retrieval, and thresholds
- `executive_telemetry.md`: executive metrics and event wiring
- `goap_architecture.md`: primary GOAP design guide
- `goap_quick_reference.md`: GOAP API and usage cheat sheet
- `stm_decay_adaptive_activation.md`: current STM decay modes and activation-weight behavior
- `../scripts/generate_memory_scorecard.py`: deterministic memory quality scorecard generator for retrieval and longitudinal eval fixtures

## Recommended Reading Order

### For developers
1. `../README.md`
2. `../phase3.md`
3. `memory_personality_roadmap.md`
4. `memory_personality_architecture.md`
5. `memory_personality_implementation_tickets.md`
6. `memory_personality_issue_backlog.md`

### For assistants
1. `ai.instructions.md`
2. `../README.md`
3. `../phase3.md`
4. `memory_personality_architecture.md`
5. `memory_personality_implementation_tickets.md`
6. `memory_personality_issue_backlog.md`

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
- Most other material has been moved under `archive/` to reduce ambiguity.
- If a document describes an older implementation slice or historical milestone, it belongs in `archive/` unless it is still referenced as current contract.

Last updated: March 28, 2026
