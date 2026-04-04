"""Confidence scoring for commitment extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommitmentConfidenceScorer:
    """Computes precision-oriented confidence for commitments."""

    phrase_weight: float = 0.35
    entity_presence_weight: float = 0.2
    entities_avg_weight: float = 0.2
    transcript_weight: float = 0.15
    segment_weight: float = 0.1

    def score(
        self,
        phrase_strength: float,
        has_entity_support: bool,
        entities_avg_confidence: float,
        transcript_avg_confidence: float,
        segment_quality_score: float,
        conflict_penalty: float = 0.0,
        actor_penalty: float = 0.0,
    ) -> float:
        score = (
            self.phrase_weight * self._clip(phrase_strength)
            + self.entity_presence_weight * (1.0 if has_entity_support else 0.0)
            + self.entities_avg_weight * self._clip(entities_avg_confidence)
            + self.transcript_weight * self._clip(transcript_avg_confidence)
            + self.segment_weight * self._clip(segment_quality_score)
        )
        score -= self._clip(conflict_penalty)
        score -= self._clip(actor_penalty)
        return self._clip(score)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


__all__ = ["CommitmentConfidenceScorer"]
