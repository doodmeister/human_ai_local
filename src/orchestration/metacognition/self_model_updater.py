from __future__ import annotations

from .models import CriticReport, SelfModel


class DefaultSelfModelUpdater:
    """Apply critic evidence to the compact metacognitive self-model."""

    def __init__(self, *, confidence_blend: float = 0.2, max_beliefs: int = 20, max_updates: int = 10) -> None:
        self._confidence_blend = confidence_blend
        self._max_beliefs = max_beliefs
        self._max_updates = max_updates

    def apply(self, current: SelfModel, report: CriticReport) -> SelfModel:
        blended_confidence = current.confidence + self._confidence_blend * (report.success_score - current.confidence)
        new_traits = dict(current.traits)
        new_traits["stability"] = round(blended_confidence, 6)
        new_traits["contradiction_sensitivity"] = round(min(1.0, 0.1 * len(report.contradictions_detected)), 6)

        beliefs = list(current.beliefs)
        if report.success_score >= 0.7:
            beliefs.append(f"successful_cycle:{report.cycle_id}")
        if report.contradictions_detected:
            beliefs.append(f"contradiction_detected:{len(report.contradictions_detected)}")
        beliefs = beliefs[-self._max_beliefs :]

        recent_updates = list(current.recent_updates)
        recent_updates.append(f"cycle={report.cycle_id};score={report.success_score:.2f}")
        recent_updates = recent_updates[-self._max_updates :]

        metadata = dict(current.metadata)
        metadata.update(
            {
                "last_cycle_id": report.cycle_id,
                "last_success_score": report.success_score,
                "last_goal_progress": report.goal_progress,
            }
        )
        return SelfModel(
            session_id=current.session_id,
            confidence=round(max(0.0, min(1.0, blended_confidence)), 6),
            traits=new_traits,
            beliefs=tuple(beliefs),
            recent_updates=tuple(recent_updates),
            metadata=metadata,
        )
