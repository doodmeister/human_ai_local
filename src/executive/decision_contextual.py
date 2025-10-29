from __future__ import annotations
from typing import Dict, List, Optional
import random
from .executive_core import DecisionEngine, Decision, Option, ExecutiveMode

"""Contextual (mode-aware) decision engine.

Adapts weighting based on current ExecutiveMode to encourage different
behaviors (focus vs exploration vs recovery). Inject into ExecutiveController
by passing as decision_engine.
"""


DEFAULT_MODE_WEIGHTS: Dict[ExecutiveMode, Dict[str, float]] = {
    ExecutiveMode.FOCUSED: {"relevance": 0.55, "urgency": 0.25, "cost": 0.15, "latency": 0.05},
    ExecutiveMode.MULTI_TASK: {"relevance": 0.45, "urgency": 0.25, "cost": 0.20, "latency": 0.10},
    ExecutiveMode.EXPLORATION: {"relevance": 0.40, "urgency": 0.15, "cost": 0.15, "latency": 0.10},  # leave room for randomness
    ExecutiveMode.REFLECTION: {"relevance": 0.50, "urgency": 0.10, "cost": 0.20, "latency": 0.20},
    ExecutiveMode.RECOVERY: {"relevance": 0.35, "urgency": 0.10, "cost": 0.30, "latency": 0.25},
}


class ContextualDecisionEngine(DecisionEngine):
    def __init__(
        self,
        mode_weights: Optional[Dict[ExecutiveMode, Dict[str, float]]] = None,
        rng: Optional[random.Random] = None,
        exploration_scale: float = 0.05,
    ):
        self.mode_weights = mode_weights or DEFAULT_MODE_WEIGHTS
        self.rng = rng or random.Random(17)
        self.exploration_scale = exploration_scale  # adaptive knob

    def adapt(self, reward: float) -> None:
        """Adjust exploration scale based on reward signal.
        Higher reward -> reduce randomness slightly; low reward -> increase within bounds."""
        if reward > 1.2:
            self.exploration_scale = max(0.01, self.exploration_scale - 0.005)
        elif reward < 0.6:
            self.exploration_scale = min(0.10, self.exploration_scale + 0.007)

    async def choose(self, options: List[Option], mode: ExecutiveMode) -> Decision:  # type: ignore[override]
        if not options:
            raise ValueError("No options provided.")
        weights = self.mode_weights.get(mode) or DEFAULT_MODE_WEIGHTS[ExecutiveMode.FOCUSED]
        w_rel = weights.get("relevance", 0.5)
        w_urg = weights.get("urgency", 0.2)
        w_cost = weights.get("cost", 0.2)
        w_lat = weights.get("latency", 0.1)
        best: tuple[Option, float] = (options[0], -1e9)
        rationales: Dict[str, str] = {}
        for opt in options:
            rel = float(opt.metadata.get("relevance", 0.5))
            urg = float(opt.metadata.get("urgency", 0.5))
            cost = float(opt.est_cost_tokens or opt.metadata.get("cost_tokens", 200))
            lat = float(opt.est_latency_ms or opt.metadata.get("latency_ms", 500))
            cost_term = 1.0 - min(cost / 2000.0, 1.0)
            lat_term = 1.0 - min(lat / 5000.0, 1.0)
            score = (w_rel * rel + w_urg * urg + w_cost * cost_term + w_lat * lat_term)
            # Exploration randomness boost
            if mode == ExecutiveMode.EXPLORATION and self.exploration_scale > 0:
                score += self.rng.random() * self.exploration_scale  # adaptive stochasticity
            # Recovery encourages low latency & low cost more strongly (already weighted), add tiny penalty for high urgency
            if mode == ExecutiveMode.RECOVERY:
                score -= urg * 0.05
            rationales[opt.id] = (
                f"mode={mode.name} rel={rel:.2f} urg={urg:.2f} cost_term={cost_term:.2f} lat_term={lat_term:.2f}"
            )
            if score > best[1]:
                best = (opt, score)
        chosen = best[0]
        return Decision(option_id=chosen.id, score=best[1], rationale=rationales[chosen.id], policy="contextual_v1")