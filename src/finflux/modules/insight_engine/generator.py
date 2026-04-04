"""Insight item generation from aggregated commitments and risk assessment."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from finflux.contracts.events import InsightItem
from .aggregator import AggregatedCommitment
from .risk import RiskAssessment


@dataclass(frozen=True)
class ConfidenceInputs:
    entities_avg_confidence: float
    transcript_avg_confidence: float


class InsightGenerator:
    """Builds commitment, risk, summary, and follow-up insights."""

    def generate(
        self,
        commitments: Sequence[AggregatedCommitment],
        risk: RiskAssessment,
        confidence_inputs: Sequence[ConfidenceInputs],
    ) -> tuple[InsightItem, ...]:
        items: list[InsightItem] = []

        blended_confidence = self._blend_confidence(commitments, confidence_inputs)

        for commitment in commitments:
            items.append(
                InsightItem(
                    insight_id=self._id("commitment", commitment.commitment_ids),
                    insight_type="commitment",
                    priority="high" if commitment.commitment_level in {"DECISION", "FULFILLED"} else "medium",
                    summary=self._commitment_summary(commitment),
                    confidence=max(0.0, min(1.0, blended_confidence * commitment.confidence)),
                    supporting_segment_ids=commitment.supporting_segment_ids,
                    commitment_ids=commitment.commitment_ids,
                    entity_ids=commitment.entity_ids,
                    insight_source="aggregation",
                )
            )

        items.append(
            InsightItem(
                insight_id=self._id("risk", tuple(item.commitment_ids for item in commitments)),
                insight_type="risk",
                priority="high" if risk.risk_level == "high" else "medium",
                summary=f"Risk level: {risk.risk_level} ({risk.risk_score:.2f})",
                confidence=max(0.0, min(1.0, 0.55 + (risk.risk_score * 0.35))),
                supporting_segment_ids=tuple(
                    sorted({sid for item in commitments for sid in item.supporting_segment_ids})
                ),
                commitment_ids=tuple(
                    sorted({cid for item in commitments for cid in item.commitment_ids})
                ),
                entity_ids=tuple(
                    sorted({eid for item in commitments for eid in item.entity_ids})
                ),
                risk_score=risk.risk_score,
                risk_level=risk.risk_level,
                risk_reasons=risk.risk_reasons,
                insight_source="risk_engine",
            )
        )

        items.append(
            InsightItem(
                insight_id=self._id("summary", tuple(item.commitment_ids for item in commitments)),
                insight_type="summary",
                priority="medium",
                summary=self._summary_line(commitments),
                confidence=max(0.0, min(1.0, blended_confidence)),
                supporting_segment_ids=tuple(
                    sorted({sid for item in commitments for sid in item.supporting_segment_ids})
                ),
                commitment_ids=tuple(
                    sorted({cid for item in commitments for cid in item.commitment_ids})
                ),
                entity_ids=tuple(
                    sorted({eid for item in commitments for eid in item.entity_ids})
                ),
                insight_source="summary_generator",
            )
        )

        for commitment in commitments:
            needs_follow_up = (
                commitment.due_date_iso is None
                or commitment.has_ambiguity
                or commitment.confidence < 0.6
            )
            if not needs_follow_up:
                continue

            items.append(
                InsightItem(
                    insight_id=self._id("follow_up", commitment.commitment_ids),
                    insight_type="follow_up",
                    priority="high" if commitment.has_ambiguity else "medium",
                    summary=self._follow_up_summary(commitment),
                    confidence=max(0.0, min(1.0, 0.5 + (commitment.confidence * 0.4))),
                    supporting_segment_ids=commitment.supporting_segment_ids,
                    commitment_ids=commitment.commitment_ids,
                    entity_ids=commitment.entity_ids,
                    insight_source="summary_generator",
                )
            )

        return tuple(items)

    @staticmethod
    def _blend_confidence(
        commitments: Sequence[AggregatedCommitment],
        confidence_inputs: Sequence[ConfidenceInputs],
    ) -> float:
        commitment_conf = fmean(item.confidence for item in commitments) if commitments else 0.4
        entities_conf = fmean(item.entities_avg_confidence for item in confidence_inputs) if confidence_inputs else 0.4
        transcript_conf = fmean(item.transcript_avg_confidence for item in confidence_inputs) if confidence_inputs else 0.4
        return max(0.0, min(1.0, (0.5 * commitment_conf) + (0.3 * entities_conf) + (0.2 * transcript_conf)))

    @staticmethod
    def _commitment_summary(item: AggregatedCommitment) -> str:
        if item.commitment_level == "FULFILLED":
            return f"{item.actor} completed {item.target or 'financial obligation'}"
        if item.due_date_iso:
            return f"{item.actor} committed to {item.action} {item.target} by {item.due_date_iso}"
        return f"{item.actor} committed to {item.action} {item.target}".strip()

    @staticmethod
    def _summary_line(commitments: Sequence[AggregatedCommitment]) -> str:
        if not commitments:
            return "No commitments detected"
        total = len(commitments)
        fulfilled = sum(1 for item in commitments if item.fulfilled)
        if total == 1:
            item = commitments[0]
            if item.due_date_iso:
                return f"User committed to {item.action} {item.target} by {item.due_date_iso}".strip()
            return f"User committed to {item.action} {item.target}".strip()
        return f"Multiple financial obligations detected ({total} commitments, {fulfilled} fulfilled)"

    @staticmethod
    def _follow_up_summary(item: AggregatedCommitment) -> str:
        if item.has_ambiguity:
            return "Low confidence commitment needs review"
        if item.due_date_iso is None:
            return "Commitment detected without due date, follow-up required"
        return "Commitment requires follow-up confirmation"

    @staticmethod
    def _id(prefix: str, seed: object) -> str:
        return str(uuid5(NAMESPACE_URL, f"{prefix}:{seed}"))


__all__ = ["ConfidenceInputs", "InsightGenerator"]
