"""Financial NER adapter with lineage-safe extraction and confidence propagation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean
from typing import Callable, Sequence
from uuid import NAMESPACE_URL, uuid5

from finflux.contracts.events import (
    FinancialEntitiesEvent,
    FinancialEntity,
    LanguageRoutedEvent,
    SpeechSegmentEvent,
    TranscriptSegmentEvent,
    TranscriptToken,
    EventEnvelope,
)
from finflux.contracts.flow import (
    validate_router_to_ner,
    validate_router_to_ner_lineage,
)
from finflux.contracts.interfaces import FinancialNER

from .detector import EntityDetector
from .normalization import NormalizationLayer
from .scoring import ConfidenceScorer


@dataclass(frozen=True)
class NERContext:
    """Optional lookup hooks that enable full evidence lineage propagation."""

    transcript_lookup: Callable[[str], TranscriptSegmentEvent]
    speech_segment_lookup: Callable[[str], SpeechSegmentEvent]
    transcript_update: Callable[[TranscriptSegmentEvent], None] | None = None


class FinancialNERExtractor(FinancialNER):
    """Extracts finance entities from normalized routed text one segment at a time."""

    def __init__(
        self,
        detector: EntityDetector | None = None,
        normalizer: NormalizationLayer | None = None,
        scorer: ConfidenceScorer | None = None,
        context: NERContext | None = None,
    ) -> None:
        self._detector = detector or EntityDetector()
        self._normalizer = normalizer or NormalizationLayer()
        self._scorer = scorer or ConfidenceScorer()
        self._context = context

    def extract(self, routed: LanguageRoutedEvent) -> FinancialEntitiesEvent:
        normalization = self._normalizer.normalize(routed.normalized_text)
        normalized_text = normalization.normalized_text

        transcript_avg_confidence = 1.0
        segment_quality = 1.0
        source_transcript_segment_id = routed.segment_id
        source_speech_segment_id = ""
        source_chunk_event_id = ""
        transcript_tokens: Sequence[TranscriptToken] = ()

        if self._context is not None:
            transcript = self._context.transcript_lookup(routed.segment_id)
            transcript_avg_confidence = transcript.avg_confidence
            source_transcript_segment_id = transcript.segment_id
            source_speech_segment_id = transcript.source_speech_segment_id
            transcript_tokens = transcript.tokens

            if normalization.normalized_applied and not transcript.is_normalized and self._context.transcript_update:
                self._context.transcript_update(replace(transcript, is_normalized=True))

            speech = self._context.speech_segment_lookup(transcript.source_speech_segment_id)
            segment_quality = speech.quality_score
            source_chunk_event_id = speech.source_chunk_event_id

        detected = self._detector.detect(normalized_text, routed.dominant_language)
        entities: list[FinancialEntity] = []

        for idx, item in enumerate(detected):
            original_start, original_end = normalization.map_normalized_to_original(
                item.start_char, item.end_char
            )
            token_start, token_end = self._map_chars_to_token_span(
                item.start_char,
                item.end_char,
                normalized_text,
                transcript_tokens,
            )

            confidence = self._scorer.score(
                detection_confidence=item.detection_confidence,
                transcript_avg_confidence=transcript_avg_confidence,
                segment_quality_score=segment_quality,
            )
            entities.append(
                FinancialEntity(
                    entity_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{routed.segment_id}:{item.entity_type}:{item.start_char}:{item.end_char}:{idx}",
                        )
                    ),
                    entity_type=item.entity_type,
                    value_text=item.value_text,
                    normalized_value=item.normalized_value,
                    start_char=item.start_char,
                    end_char=item.end_char,
                    confidence=confidence,
                    evidence_text=item.value_text,
                    original_start_char=original_start,
                    original_end_char=original_end,
                    token_start_index=token_start,
                    token_end_index=token_end,
                )
            )

        avg_confidence = fmean(entity.confidence for entity in entities) if entities else 0.0
        event = FinancialEntitiesEvent(
            envelope=EventEnvelope(
                trace_id=routed.envelope.trace_id,
                call_id=routed.envelope.call_id,
            ),
            segment_id=routed.segment_id,
            avg_confidence=max(0.0, min(1.0, avg_confidence)),
            entities=tuple(entities),
            source_transcript_segment_id=source_transcript_segment_id,
            source_speech_segment_id=source_speech_segment_id,
            source_chunk_event_id=source_chunk_event_id,
        )
        validate_router_to_ner(event)
        validate_router_to_ner_lineage(routed, event)
        return event

    def _map_chars_to_token_span(
        self,
        start_char: int,
        end_char: int,
        normalized_text: str,
        tokens: Sequence[TranscriptToken],
    ) -> tuple[int | None, int | None]:
        if not tokens:
            return None, None

        offsets = self._token_char_offsets(normalized_text, tokens)
        start_idx: int | None = None
        end_idx: int | None = None

        for index, (token_start, token_end) in enumerate(offsets):
            if token_end <= start_char or token_start >= end_char:
                continue
            if start_idx is None:
                start_idx = index
            end_idx = index

        return start_idx, end_idx

    @staticmethod
    def _token_char_offsets(
        text: str,
        tokens: Sequence[TranscriptToken],
    ) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        cursor = 0
        lower_text = text.lower()

        for token in tokens:
            token_text = token.token.strip()
            if not token_text:
                offsets.append((cursor, cursor))
                continue
            token_lower = token_text.lower()
            found = lower_text.find(token_lower, cursor)
            if found < 0:
                found = lower_text.find(token_lower)
                if found < 0:
                    offsets.append((cursor, cursor))
                    continue
            start = found
            end = found + len(token_text)
            offsets.append((start, end))
            cursor = end

        return offsets


__all__ = ["FinancialNERExtractor", "NERContext"]
