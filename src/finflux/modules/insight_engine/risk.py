"""Risk scoring for aggregated financial commitments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .aggregator import AggregatedCommitment


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: float
    risk_level: str
    risk_reasons: tuple[str, ...]


class RiskScorer:
    """Produces risk score and reasons from commitment aggregates."""

    def score(self, commitments: Sequence[AggregatedCommitment]) -> RiskAssessment:
        if not commitments:
            return RiskAssessment(risk_score=0.1, risk_level="low", risk_reasons=("no_commitments_detected",))

        reasons: list[str] = []
        total = len(commitments)
        unfulfilled = sum(1 for item in commitments if not item.fulfilled)
        low_conf = sum(1 for item in commitments if item.confidence < 0.55)
        ambiguous = sum(1 for item in commitments if item.has_ambiguity)
        conflicts = sum(1 for item in commitments if item.has_conflict)
        missed_deadline = sum(1 for item in commitments if self._is_missed_deadline(item.due_date_iso, item.fulfilled))

        score = 0.18
        score += min(0.25, total * 0.04)
        score += min(0.2, unfulfilled * 0.05)
        score += min(0.15, low_conf * 0.05)
        score += min(0.15, ambiguous * 0.05)
        score += min(0.15, conflicts * 0.05)
        score += min(0.15, missed_deadline * 0.07)

        if unfulfilled:
            reasons.append("unfulfilled_commitments_present")
        if low_conf:
            reasons.append("low_confidence_commitments_present")
        if ambiguous:
            reasons.append("ambiguity_signals_detected")
        if conflicts:
            reasons.append("conflicting_commitments_detected")
        if missed_deadline:
            reasons.append("potential_missed_deadlines")

        score = max(0.0, min(1.0, score))
        if score >= 0.75:
            level = "high"
        elif score >= 0.45:
            level = "medium"
        else:
            level = "low"

        if not reasons:
            reasons.append("normal_commitment_profile")

        return RiskAssessment(risk_score=score, risk_level=level, risk_reasons=tuple(reasons))

    @staticmethod
    def _is_missed_deadline(due_date_iso: str | None, fulfilled: bool) -> bool:
        if due_date_iso is None or fulfilled:
            return False
        try:
            due = datetime.fromisoformat(due_date_iso).date()
        except ValueError:
            return False
        return due < datetime.utcnow().date()


__all__ = ["RiskAssessment", "RiskScorer"]
