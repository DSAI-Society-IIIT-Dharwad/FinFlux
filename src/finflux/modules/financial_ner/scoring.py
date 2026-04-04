"""Confidence scoring for extracted financial entities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceScorer:
    """Combines detection confidence with transcript and segment quality signals."""

    detection_weight: float = 0.55
    transcript_weight: float = 0.3
    segment_weight: float = 0.15

    def score(
        self,
        detection_confidence: float,
        transcript_avg_confidence: float,
        segment_quality_score: float,
    ) -> float:
        weighted = (
            self.detection_weight * self._clip(detection_confidence)
            + self.transcript_weight * self._clip(transcript_avg_confidence)
            + self.segment_weight * self._clip(segment_quality_score)
        )
        return self._clip(weighted)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


__all__ = ["ConfidenceScorer"]
