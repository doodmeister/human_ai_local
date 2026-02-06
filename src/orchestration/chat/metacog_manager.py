from __future__ import annotations

from typing import Any, Dict, Optional
import time

from pydantic import BaseModel, Field
try:  # Pydantic v2
    from pydantic import ConfigDict
except Exception:  # fallback for older versions
    ConfigDict = None  # type: ignore

from src.learning.learning_law import clamp01, utility_score


class MetacogManager:
    def __init__(self, metrics_registry: Any) -> None:
        self._metrics = metrics_registry

    class SnapshotModel(BaseModel):
        ts: float = Field(..., description="Timestamp (epoch seconds)")
        turn_counter: int
        performance: Optional[Dict[str, Any]] = None
        recent_consolidation_selectivity: Optional[float] = None
        promotion_age_p95_seconds: Optional[float] = None
        stm_utilization: Optional[float] = None
        stm_capacity: Optional[int] = None
        last_user_turn_status: Optional[str] = None
        if ConfigDict is not None:  # pydantic v2
            model_config = ConfigDict(extra="allow")  # type: ignore
        else:  # pragma: no cover - legacy compatibility
            class Config:  # type: ignore
                extra = "allow"

    def snapshot(
        self,
        *,
        turn_counter: int,
        consolidation_log: list[Dict[str, Any]],
        consolidator: Any,
    ) -> Dict[str, Any]:
        """Generate a lightweight metacognitive snapshot."""
        snap: Dict[str, Any] = {
            "ts": time.time(),
            "turn_counter": turn_counter,
        }
        try:
            perf_p95 = self._metrics.get_p95("chat_turn_latency_ms")
            degraded = self._metrics.state.get("performance_degraded")
            snap["performance"] = {"latency_p95_ms": perf_p95, "degraded": degraded}

            stm_total = self._metrics.counters.get("consolidation_stm_store_total")
            ltm_promos = self._metrics.counters.get("consolidation_ltm_promotions_total")
            if isinstance(stm_total, (int, float)) and stm_total > 0 and isinstance(ltm_promos, (int, float)):
                snap["recent_consolidation_selectivity"] = ltm_promos / max(stm_total, 1)

            if "consolidation_promotion_age_seconds" in self._metrics.histograms:
                try:
                    age_p95 = self._metrics.percentile("consolidation_promotion_age_seconds", 95)
                    snap["promotion_age_p95_seconds"] = age_p95
                except Exception:
                    pass

            util, capacity = self._get_stm_utilization(consolidator)
            if util is not None:
                snap["stm_utilization"] = util
            if capacity is not None:
                snap["stm_capacity"] = capacity

            if consolidation_log:
                snap["last_user_turn_status"] = consolidation_log[-1]["status"]
        except Exception:
            pass
        try:
            model = self.SnapshotModel(**snap)
            if hasattr(model, "model_dump"):
                return model.model_dump()
            return model.dict()
        except Exception:
            return snap

    def adjust_interval(
        self,
        *,
        current_interval: int,
        consolidator: Any,
    ) -> int:
        """Adjust metacog interval based on performance and STM utilization."""
        new_interval = current_interval
        try:
            stm_util, _capacity = self._get_stm_utilization(consolidator)
            degraded_flag = self._metrics.state.get("performance_degraded")
            cost = 0.0
            if degraded_flag:
                cost += 0.6
            if stm_util is not None and stm_util >= 0.85:
                cost += 0.6
            u = utility_score(
                benefit=clamp01(self._metrics.state.get("retrieval_benefit_ema", 0.5)),
                cost=clamp01(cost),
            )
            if u < 0.0:
                if current_interval > 2:
                    new_interval = current_interval - 1
            else:
                if (stm_util is None or stm_util < 0.70) and not degraded_flag:
                    if current_interval < 10:
                        new_interval = current_interval + 1
            if new_interval != current_interval:
                self._metrics.state["metacog_interval"] = new_interval
        except Exception:
            return current_interval
        return new_interval

    def adjust_activation_weights(self, *, consolidator: Any) -> None:
        """Apply adaptive activation weighting based on metacog signals."""
        try:
            stm_obj = None
            if consolidator is not None:
                stm_obj = getattr(consolidator, "stm", None)
            if stm_obj is None:
                return
            if not (hasattr(stm_obj, "set_activation_weights") and hasattr(stm_obj, "get_activation_weights")):
                return
            weights = stm_obj.get_activation_weights()
            degraded_flag = self._metrics.state.get("performance_degraded")
            stm_util_ratio, _capacity = self._get_stm_utilization(consolidator)
            rec = weights.get("recency", 0.4)
            freq = weights.get("frequency", 0.3)
            sal = weights.get("salience", 0.3)
            adjusted = False
            cost = 0.0
            if degraded_flag:
                cost += 0.6
            if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                cost += 0.6
            u = utility_score(
                benefit=clamp01(self._metrics.state.get("retrieval_benefit_ema", 0.5)),
                cost=clamp01(cost),
            )
            if u < 0.0:
                if degraded_flag:
                    rec += 0.05
                    sal -= 0.025
                    freq -= 0.025
                    adjusted = True
                if stm_util_ratio is not None and stm_util_ratio >= 0.85:
                    sal += 0.05
                    rec -= 0.025
                    freq -= 0.025
                    adjusted = True
            if adjusted:
                rec = max(0.01, rec)
                freq = max(0.01, freq)
                sal = max(0.01, sal)
                stm_obj.set_activation_weights(recency=rec, frequency=freq, salience=sal)
                self._metrics.inc("metacog_activation_weight_adjustments_total")
        except Exception:
            return

    def _get_stm_utilization(self, consolidator: Any) -> tuple[Optional[float], Optional[int]]:
        if consolidator is None:
            return None, None
        stm_obj = getattr(consolidator, "stm", None)
        if stm_obj is None:
            return None, None
        capacity = getattr(stm_obj, "capacity", None)
        size = None
        try:
            if hasattr(stm_obj, "__len__"):
                size = len(stm_obj)  # type: ignore
        except Exception:
            size = None
        if size is None:
            size = getattr(stm_obj, "size", None)
        if not (isinstance(size, int) and isinstance(capacity, int) and capacity > 0):
            return None, capacity if isinstance(capacity, int) else None
        return min(1.0, size / capacity), capacity
