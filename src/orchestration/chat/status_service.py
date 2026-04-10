from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory

from .metrics import metrics_registry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ChatStatusService:
    """Formatting and status-query helper for chat-facing system responses."""

    def __init__(
        self,
        *,
        get_config: Callable[[], Dict[str, Any]],
        get_stm_usage_snapshot: Callable[[], tuple[Optional[int], Optional[int], Optional[float]]],
        get_active_goal_count: Callable[[str], int],
        get_metacog_interval: Callable[[], int],
        get_consolidation_event_count: Callable[[], int],
    ) -> None:
        self._get_config = get_config
        self._get_stm_usage_snapshot = get_stm_usage_snapshot
        self._get_active_goal_count = get_active_goal_count
        self._get_metacog_interval = get_metacog_interval
        self._get_consolidation_event_count = get_consolidation_event_count

    def build_performance_status(self) -> Dict[str, Any]:
        p95 = metrics_registry.get_p95("chat_turn_latency_ms")
        cfg = self._get_config()
        target = cfg.get("performance_target_p95_ms")
        degraded = False
        if isinstance(target, (int, float)) and target > 0:
            degraded = p95 > target
        throughput = metrics_registry.get_rate(
            "chat_turn",
            window_seconds=float(cfg.get("throughput_window_seconds", 60)),
        )
        ema = metrics_registry.state.get("ema_turn_latency_ms", 0.0)
        cons_counters: Dict[str, Any] = {}
        promotion_age_p95 = 0.0
        if metrics_registry.counters.get("consolidation_stm_store_total") is not None:
            cons_counters["stm_store_total"] = metrics_registry.counters.get("consolidation_stm_store_total", 0)
        if metrics_registry.counters.get("consolidation_ltm_promotions_total") is not None:
            cons_counters["ltm_promotions_total"] = metrics_registry.counters.get("consolidation_ltm_promotions_total", 0)
        if "consolidation_promotion_age_seconds" in metrics_registry.histograms:
            promotion_age_p95 = metrics_registry.percentile("consolidation_promotion_age_seconds", 95)
        base: Dict[str, Any] = {
            "latency_p95_ms": p95,
            "target_p95_ms": target,
            "performance_degraded": degraded,
            "ema_turn_latency_ms": ema,
            "chat_turns_per_sec": throughput,
        }
        try:
            metacog_counters = {}
            for key in (
                "metacog_snapshots_total",
                "metacog_advisory_items_total",
                "metacog_stm_high_util_events_total",
                "metacog_performance_degraded_events_total",
                "adaptive_retrieval_applied_total",
            ):
                if key in metrics_registry.counters:
                    metacog_counters[key] = metrics_registry.counters.get(key, 0)
            if metacog_counters:
                base["metacog"] = {
                    "counters": metacog_counters,
                    "interval": self._get_metacog_interval(),
                }
        except Exception:
            pass
        if cons_counters:
            stm_total = float(cons_counters.get("stm_store_total", 0))
            ltm_total = float(cons_counters.get("ltm_promotions_total", 0))
            selectivity = (ltm_total / stm_total) if stm_total > 0 else 0.0
            ages_hist = metrics_registry.histograms.get("consolidation_promotion_age_seconds", [])
            recent_window = ages_hist[-5:] if ages_hist else []
            recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0.0
            alert_threshold = cfg.get("consolidation_promotion_age_p95_alert_seconds")
            age_alert = False
            if isinstance(alert_threshold, (int, float)) and alert_threshold > 0 and promotion_age_p95 > alert_threshold:
                age_alert = True
                metrics_registry.state["consolidation_age_alert"] = True
            base["consolidation"] = {
                "counters": cons_counters,
                "promotion_age_p95_seconds": promotion_age_p95,
                "selectivity_ratio": selectivity,
                "recent_promotion_age_seconds": {
                    "count": len(recent_window),
                    "avg": recent_avg,
                    "values": recent_window,
                },
                "promotion_age_alert": age_alert,
                "promotion_age_alert_threshold": alert_threshold,
            }
        return base

    def format_latency_ms(self, value: Optional[Any]) -> str:
        try:
            if isinstance(value, (int, float)):
                return f"{value:.0f} ms"
        except Exception:
            pass
        return "unknown"

    def format_percentage(self, value: Optional[Any]) -> str:
        try:
            if isinstance(value, (int, float)):
                return f"{value * 100:.0f}%"
        except Exception:
            pass
        return "unknown"

    def resolve_reminder_due_time(self, intent: Any) -> Optional[datetime]:
        offset = intent.entities.get("reminder_offset_seconds")
        if offset is not None:
            try:
                seconds = float(offset)
                if seconds > 0:
                    return _utc_now() + timedelta(seconds=seconds)
            except (TypeError, ValueError):
                pass
        due_text = intent.entities.get("reminder_due_time")
        if isinstance(due_text, str):
            try:
                return _normalize_datetime(datetime.fromisoformat(due_text))
            except Exception:
                pass
        return None

    def format_due_phrase(self, due_time: Optional[datetime]) -> str:
        if due_time is None:
            return "no specific time"
        due_time = _normalize_datetime(due_time)
        if due_time is None:
            return "no specific time"
        now = _utc_now()
        delta = due_time - now
        total_seconds = delta.total_seconds()
        if total_seconds < -60:
            return "past due"
        if total_seconds < 60:
            return "due now"
        minutes = total_seconds / 60
        if minutes < 60:
            return f"due in {int(round(minutes))} min"
        hours = minutes / 60
        if hours < 24:
            return f"due in {hours:.1f} hr"
        days = hours / 24
        return f"due in {days:.1f} days"

    def format_proactive_reminder_summary(self, due_reminders: list[Any], upcoming_reminders: list[Any]) -> str:
        lines: list[str] = []
        if due_reminders:
            if len(due_reminders) == 1:
                lines.append(f"Reminder now due: {getattr(due_reminders[0], 'content', '')}")
            else:
                lines.append("Reminders now due:")
                for reminder in due_reminders[:3]:
                    lines.append(f" - {getattr(reminder, 'content', '')}")
                if len(due_reminders) > 3:
                    lines.append(f"   (+{len(due_reminders) - 3} more)")
        if upcoming_reminders:
            lines.append("Coming up soon:")
            for reminder in upcoming_reminders[:3]:
                lines.append(
                    f" - {getattr(reminder, 'content', '')} ({self.format_due_phrase(getattr(reminder, 'due_time', None))})"
                )
            if len(upcoming_reminders) > 3:
                lines.append(f"   (+{len(upcoming_reminders) - 3} more)")
        return "\n".join(lines).strip()

    def serialize_reminder_brief(self, reminder: Any) -> Dict[str, Any]:
        due_time = getattr(reminder, "due_time", None)
        return {
            "id": getattr(reminder, "id", None),
            "content": getattr(reminder, "content", ""),
            "due_time": due_time.isoformat() if isinstance(due_time, datetime) else None,
            "due_phrase": self.format_due_phrase(due_time) if due_time else "no specific time",
        }

    def get_stat_value(self, stats: Any, key: str, default: Any) -> float:
        value: Any = default
        try:
            if isinstance(stats, dict):
                value = stats.get(key, default)
            else:
                value = getattr(stats, key, default)
        except Exception:
            value = default

        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(default)
            except (TypeError, ValueError):
                return 0.0

    def handle_performance_query(self, intent: Any, session_id: str) -> Optional[str]:
        del session_id
        try:
            status = self.build_performance_status()
        except Exception:
            return None

        if not isinstance(status, dict):
            return None

        metric_focus = intent.entities.get("metric_type", "general")
        latency_text = self.format_latency_ms(status.get("latency_p95_ms"))
        target_text = self.format_latency_ms(status.get("target_p95_ms"))
        ema_text = self.format_latency_ms(status.get("ema_turn_latency_ms"))
        throughput = status.get("chat_turns_per_sec")
        if throughput is None:
            throughput_text = "unknown throughput"
        else:
            try:
                throughput_text = f"{float(throughput):.2f} turns/sec"
            except (TypeError, ValueError):
                throughput_text = "unknown throughput"
        degraded = bool(status.get("performance_degraded"))
        health_text = "Running slower than target" if degraded else "Meeting latency target"

        lines = [
            f"- Latency p95: {latency_text} (target {target_text})",
            f"- EMA latency: {ema_text}",
            f"- Throughput: {throughput_text}",
            f"- Status: {health_text}",
        ]
        if metric_focus and metric_focus != "general":
            focus_text = metric_focus.replace("_", " ")
            lines.append(f"- Metric focus: watching {focus_text} trends")
        return "\n".join(lines)

    def handle_system_status(self, intent: Any, session_id: str) -> str:
        detail_level = intent.entities.get("detail_level", "normal")
        component_focus = intent.entities.get("component", "general")
        active_goals = self._get_active_goal_count(session_id)
        size, capacity, util = self._get_stm_usage_snapshot()
        if size is None and capacity is None:
            stm_line = "STM load: unavailable (not configured)"
        elif capacity:
            stm_line = f"STM load: {size}/{capacity} items ({self.format_percentage(util)})"
        else:
            stm_line = f"STM load: {size or 0} items"

        perf_snapshot = None
        try:
            perf_snapshot = self.build_performance_status()
        except Exception:
            perf_snapshot = None

        latency_line = None
        if isinstance(perf_snapshot, dict):
            latency_line = f"Latency: {self.format_latency_ms(perf_snapshot.get('latency_p95_ms'))}"
            latency_line += " (above target)" if perf_snapshot.get("performance_degraded") else " (stable)"

        due_count = 0
        upcoming_hour = 0
        try:
            pm = get_inmemory_prospective_memory()
            reminders = pm.list_reminders(include_completed=False)
            now = _utc_now()
            for reminder in reminders:
                due_time = _normalize_datetime(getattr(reminder, "due_time", None))
                if not due_time:
                    continue
                delta = (due_time - now).total_seconds()
                if delta <= 0:
                    due_count += 1
                elif delta <= 3600:
                    upcoming_hour += 1
        except Exception:
            pass

        lines = [
            f"- Active goals: {active_goals}",
            f"- {stm_line}",
            f"- Prospective reminders: {due_count} due, {upcoming_hour} within an hour",
            f"- Metacog cadence: every {max(1, self._get_metacog_interval())} turns",
        ]

        if latency_line and component_focus in ("general", "performance", "cognitive"):
            lines.append(f"- {latency_line}")

        if detail_level == "detailed":
            ema_text = self.format_latency_ms(perf_snapshot.get("ema_turn_latency_ms") if isinstance(perf_snapshot, dict) else None)
            lines.append(f"- EMA latency: {ema_text}")
            lines.append(f"- Consolidation history: {self._get_consolidation_event_count()} recent events")

        return "\n".join(lines)