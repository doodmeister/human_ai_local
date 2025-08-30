# STM Decay Modes & Adaptive Activation Weights

Date: 2025-08-30

## Overview
Short-Term Memory (STM) now supports configurable decay functions and dynamically adjustable activation component weights. This improves biological fidelity and allows metacognitive modules to modulate retrieval emphasis based on system state (e.g., cognitive load, degradation signals).

## Decay Modes
Configured via `STMConfiguration.decay_mode`:
- `linear`: `recency = max(0, 1 - age_hours * decay_rate)` (original behavior)
- `exponential`: `exp(-λ * age)` with λ scaled by `decay_lambda * decay_rate`
- `power`: `(age + 1)^(-α)` with α scaled by `decay_power_alpha * (decay_rate + ε)`
- `sigmoid`: `1 / (1 + exp(k * (age - midpoint)))` using `decay_sigmoid_k` and `decay_sigmoid_midpoint_hours`

All modes clamp output to `[0,1]` and are monotonic w.r.t. age.

## Adaptive Activation Weights
Activation combines three components:
| Component | Source | Description |
|-----------|--------|-------------|
| Recency   | Decay function | Time-based freshness |
| Frequency | `access_count` | Usage reinforcement (diminishing to 1.0 at ~10 uses) |
| Salience  | Mean(importance, attention_score) | Combined subjective value |

Weights are normalized each call and can be updated at runtime:
```python
stm.set_activation_weights(recency=0.6, frequency=0.2, salience=0.2)
current = stm.get_activation_weights()
```

Chat/metacog layers can opportunistically rebalance weights (e.g., elevate recency under overload or boost salience during focus tasks).

## Test Mode (Storage Disabled)
`VectorShortTermMemory(..., disable_storage=True)` provides a no-IO mode with null vector/DB backends enabling fast unit testing of activation logic without ChromaDB or embedding model availability.

## Added Public Helper
`compute_activation_for_metadata(metadata: dict) -> float` exposes activation computation for diagnostics/tests.

## Validation
Test file: `tests/test_stm_decay_modes.py`
Coverage:
- Monotonic decay across all modes.
- Comparative behavior linear vs exponential early retention.
- Sigmoid shape sanity.
- Weight adjustment increases separation between recent and old items.

## Future Enhancements
- Meta-driven automatic weight annealing based on retrieval precision/latency.
- Per-item adaptive decay rates (e.g., emotion/importance modulation).
- Logging hooks for activation component contributions (explainability surface).

---
Maintainer: Cognitive Architecture Team