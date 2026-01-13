# P1 Action Plan (Jan 2026) — Observability + Reliability

**Date:** January 13, 2026  
**Scope:** P1 items from [nextsteps.md](../nextsteps.md), plus doc consolidation to keep the repo navigable.

## Canonical Status Docs

- **Current state / what’s deployed:** [nextsteps.md](../nextsteps.md)
- **P1 execution plan (this doc):** `docs/P1_ACTION_PLAN.md`
- **Long-term product direction:** [docs/roadmap.md](roadmap.md)

> Note: Older UI/executive integration audit/plans are kept for history but may be out of date.

---

## P1 Outcomes (What “done” looks like)

### 1) Observability (1 week)

**Goal:** We can detect regressions quickly, measure latency, and capture exceptions in production-like runs.

Deliverables:
- [ ] **Sentry integration** behind env var `SENTRY_DSN`
- [ ] **Prometheus metrics endpoint** at `GET /metrics` behind `PROMETHEUS_ENABLED=1`
- [ ] Basic dashboards/playbook:
  - [ ] p50/p95 request latency
  - [ ] error rate by endpoint
  - [ ] top slow endpoints

Implementation notes:
- Metrics labels must stay low-cardinality (use route templates when possible).

### 2) Skipped Tests Reduction (1–2 weeks)

**Goal:** Reduce skipped tests from 19 → as close to 0 as possible, without flakiness.

Buckets (from [nextsteps.md](../nextsteps.md)):
- **OpenMemory MCP tests:** require Docker container running
- **ChromaDB PyO3 reinit issues:** Python 3.12+ limitation
- **Singleton shared state:** test isolation

Deliverables:
- [ ] Make OpenMemory tests auto-skip with a clear message and a one-command enable path
- [ ] Isolate singleton state in tests (fixtures reset/clear) where feasible
- [ ] Document known ChromaDB/PyO3 constraints and test strategy

### 3) Documentation Consolidation (0.5–1 day)

**Goal:** No contradictions in “what’s next” docs.

Deliverables:
- [ ] Mark older audit/plan docs as historical and point to [nextsteps.md](../nextsteps.md) + this plan
- [ ] Ensure README / startup guide mention observability toggles

---

## Execution Checklist (Start Work)

### Week 1 — Observability

1. Add optional deps:
   - `sentry-sdk`
   - `prometheus-client`
2. Add env toggles:
   - `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_TRACES_SAMPLE_RATE`
   - `PROMETHEUS_ENABLED`
3. Validate locally:
   - `PROMETHEUS_ENABLED=1 python start_server.py` then hit `GET /metrics`
   - Force an exception and confirm Sentry capture (if configured)

### Week 2 — Skipped Tests

1. Identify skipped tests and categorize by bucket
2. Fix easy isolation issues (singleton resets)
3. Add “how to run OpenMemory tests” doc snippet

---

## Work Started (Jan 13, 2026)

- Implemented optional Prometheus `/metrics` and optional Sentry exception capture in `george_api_simple.py`.
- Added `sentry-sdk` and `prometheus-client` to `requirements.txt`.
