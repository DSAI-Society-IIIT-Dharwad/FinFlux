"""Insight engine adapter that aggregates commitments into final insight events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from finflux.contracts.events import InsightEvent, TranscriptSegmentEvent, EventEnvelope
from finflux.contracts.flow import validate_commitment_to_insight
from finflux.contracts.interfaces import InsightEngine, InsightInput

from .aggregator import CommitmentAggregator
from .generator import ConfidenceInputs, InsightGenerator
from .risk import RiskScorer


@dataclass(frozen=True)
class InsightContext:
    """Lookup hooks for transcript-level confidence propagation."""

    transcript_lookup: Callable[[str], TranscriptSegmentEvent]


class BatchInsightEngine(InsightEngine):
    """Batch-of-segments insight synthesis implementation."""

    def __init__(
        self,
        aggregator: CommitmentAggregator | None = None,
        risk_scorer: RiskScorer | None = None,
        generator: InsightGenerator | None = None,
        context: InsightContext | None = None,
    ) -> None:
        self._aggregator = aggregator or CommitmentAggregator()
        self._risk = risk_scorer or RiskScorer()
        self._generator = generator or InsightGenerator()
        self._context = context

    def synthesize(self, source: Sequence[InsightInput]) -> InsightEvent:
        if not source:
            return InsightEvent(
                envelope=EventEnvelope(trace_id="", call_id=""),
                call_id="",
                items=(),
            )

        for item in source:
            validate_commitment_to_insight(item)

        aggregated = self._aggregator.aggregate(source)
        risk = self._risk.score(aggregated)
        confidence_inputs = self._collect_confidence_inputs(source)
        items = self._generator.generate(aggregated, risk, confidence_inputs)

        first = source[0]
        return InsightEvent(
            envelope=EventEnvelope(
                trace_id=first.commitment_event.envelope.trace_id,
                call_id=first.commitment_event.envelope.call_id,
            ),
            call_id=first.commitment_event.envelope.call_id,
            items=items,
        )

    def _collect_confidence_inputs(
        self, source: Sequence[InsightInput]
    ) -> tuple[ConfidenceInputs, ...]:
        inputs: list[ConfidenceInputs] = []
        for item in source:
            transcript_conf = 1.0
            transcript_id = item.commitment_event.source_transcript_segment_id
            if self._context is not None and transcript_id:
                transcript = self._context.transcript_lookup(transcript_id)
                transcript_conf = transcript.avg_confidence
            inputs.append(
                ConfidenceInputs(
                    entities_avg_confidence=item.entities_event.avg_confidence,
                    transcript_avg_confidence=transcript_conf,
                )
            )
        return tuple(inputs)


__all__ = ["BatchInsightEngine", "InsightContext"]
