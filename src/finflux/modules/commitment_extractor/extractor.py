"""Commitment extraction adapter with evidence lineage and precision controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from uuid import NAMESPACE_URL, uuid5

from finflux.contracts.events import (
    CommitmentCandidate,
    CommitmentExtractionEvent,
    EventEnvelope,
    FinancialEntity,
    SpeechSegmentEvent,
    TranscriptSegmentEvent,
)
from finflux.contracts.flow import (
    validate_ner_to_commitment,
    validate_ner_to_commitment_lineage,
)
from finflux.contracts.interfaces import CommitmentExtractor, ExtractionInput

from .detector import CommitmentDetector, DetectedCommitment
from .resolver import ActorResolver, TimeResolver
from .scoring import CommitmentConfidenceScorer


@dataclass(frozen=True)
class CommitmentContext:
    """Optional lookup hooks enabling full confidence and lineage propagation."""

    transcript_lookup: Callable[[str], TranscriptSegmentEvent]
    speech_segment_lookup: Callable[[str], SpeechSegmentEvent]


class PrecisionCommitmentExtractor(CommitmentExtractor):
    """High-precision commitment extractor operating on one segment at a time."""

    def __init__(
        self,
        detector: CommitmentDetector | None = None,
        actor_resolver: ActorResolver | None = None,
        time_resolver: TimeResolver | None = None,
        scorer: CommitmentConfidenceScorer | None = None,
        context: CommitmentContext | None = None,
    ) -> None:
        self._detector = detector or CommitmentDetector()
        self._actor_resolver = actor_resolver or ActorResolver()
        self._time_resolver = time_resolver or TimeResolver()
        self._scorer = scorer or CommitmentConfidenceScorer()
        self._context = context

    def extract(self, source: ExtractionInput) -> CommitmentExtractionEvent:
        validate_ner_to_commitment(source)

        text = source.language_event.normalized_text
        entities = tuple(source.entities_event.entities)
        detections = self._detector.detect(text, entities)

        transcript_avg = 1.0
        segment_quality = 1.0
        source_transcript_segment_id = source.entities_event.source_transcript_segment_id
        source_speech_segment_id = source.entities_event.source_speech_segment_id
        source_chunk_event_id = source.entities_event.source_chunk_event_id

        if self._context is not None:
            transcript = self._context.transcript_lookup(source.entities_event.source_transcript_segment_id)
            speech = self._context.speech_segment_lookup(transcript.source_speech_segment_id)
            transcript_avg = transcript.avg_confidence
            segment_quality = speech.quality_score
            source_transcript_segment_id = transcript.segment_id
            source_speech_segment_id = transcript.source_speech_segment_id
            source_chunk_event_id = speech.source_chunk_event_id

        unresolved_reasons: list[str] = []
        candidates: list[CommitmentCandidate] = []

        for idx, detection in enumerate(detections):
            actor_resolution = self._actor_resolver.resolve(text, entities)
            if actor_resolution.unresolved_reason:
                unresolved_reasons.append(actor_resolution.unresolved_reason)

            due_date_iso = self._time_resolver.resolve_due_date_iso(entities)
            action, target = self._resolve_action_target(detection, entities)

            relevant_entities = self._entities_near_detection(detection, entities)
            if not relevant_entities:
                relevant_entities = entities

            conflict_penalty = 0.15 if self._has_entity_conflict(relevant_entities) else 0.0
            actor_penalty = 0.12 if actor_resolution.unresolved_reason else 0.0

            confidence = self._scorer.score(
                phrase_strength=detection.phrase_strength,
                has_entity_support=bool(relevant_entities),
                entities_avg_confidence=source.entities_event.avg_confidence,
                transcript_avg_confidence=transcript_avg,
                segment_quality_score=segment_quality,
                conflict_penalty=conflict_penalty,
                actor_penalty=actor_penalty,
            )

            token_start, token_end = self._merge_entity_token_span(relevant_entities)
            normalized_start = detection.start_char
            normalized_end = detection.end_char
            original_start, original_end = self._map_original_span(
                relevant_entities,
                normalized_start,
                normalized_end,
            )

            candidates.append(
                CommitmentCandidate(
                    commitment_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{source.language_event.segment_id}:{detection.commitment_level}:{detection.start_char}:{idx}",
                        )
                    ),
                    commitment_level=detection.commitment_level,
                    actor=actor_resolution.actor,
                    action=action,
                    target=target,
                    due_date_iso=due_date_iso,
                    evidence_sentence=self._evidence_sentence(
                        text,
                        detection.start_char,
                        detection.end_char,
                    ),
                    conditions=self._extract_conditions(text),
                    confidence=confidence,
                    evidence_segment_id=source.language_event.segment_id,
                    entity_ids_used=tuple(entity.entity_id for entity in relevant_entities),
                    entity_evidence_texts=tuple(self._entity_evidence_text(entity) for entity in relevant_entities),
                    token_start_index=token_start,
                    token_end_index=token_end,
                    normalized_start_char=normalized_start,
                    normalized_end_char=normalized_end,
                    original_start_char=original_start,
                    original_end_char=original_end,
                )
            )

        event = CommitmentExtractionEvent(
            envelope=EventEnvelope(
                trace_id=source.language_event.envelope.trace_id,
                call_id=source.language_event.envelope.call_id,
            ),
            segment_id=source.language_event.segment_id,
            candidates=tuple(candidates),
            unresolved_reasons=tuple(sorted(set(unresolved_reasons))),
            source_transcript_segment_id=source_transcript_segment_id,
            source_speech_segment_id=source_speech_segment_id,
            source_chunk_event_id=source_chunk_event_id,
        )
        validate_ner_to_commitment_lineage(source, event)
        return event

    @staticmethod
    def _entity_evidence_text(entity: FinancialEntity) -> str:
        return entity.evidence_text or entity.value_text

    @staticmethod
    def _resolve_action_target(
        detection: DetectedCommitment,
        entities: Sequence[FinancialEntity],
    ) -> tuple[str, str]:
        action = detection.phrase.strip().lower()

        product_entities = [entity for entity in entities if entity.entity_type == "product"]
        obligation_entities = [entity for entity in entities if entity.entity_type == "obligation"]
        amount_entities = [entity for entity in entities if entity.entity_type == "amount"]

        if obligation_entities:
            action = obligation_entities[0].normalized_value

        target_parts: list[str] = []
        if product_entities:
            target_parts.append(product_entities[0].normalized_value)
        if amount_entities:
            target_parts.append(amount_entities[0].normalized_value)

        return action or "unknown_action", " ".join(target_parts) if target_parts else "financial_obligation"

    @staticmethod
    def _entities_near_detection(
        detection: DetectedCommitment,
        entities: Sequence[FinancialEntity],
    ) -> tuple[FinancialEntity, ...]:
        relevant: list[FinancialEntity] = []
        for entity in entities:
            if entity.end_char < detection.start_char - 50:
                continue
            if entity.start_char > detection.end_char + 80:
                continue
            relevant.append(entity)
        return tuple(relevant)

    @staticmethod
    def _has_entity_conflict(entities: Sequence[FinancialEntity]) -> bool:
        dates = [entity for entity in entities if entity.entity_type == "date"]
        products = [entity for entity in entities if entity.entity_type == "product"]
        return len(dates) > 1 or len(products) > 1

    @staticmethod
    def _merge_entity_token_span(
        entities: Sequence[FinancialEntity],
    ) -> tuple[int | None, int | None]:
        starts = [entity.token_start_index for entity in entities if entity.token_start_index is not None]
        ends = [entity.token_end_index for entity in entities if entity.token_end_index is not None]
        if not starts or not ends:
            return None, None
        return min(starts), max(ends)

    @staticmethod
    def _map_original_span(
        entities: Sequence[FinancialEntity],
        normalized_start: int,
        normalized_end: int,
    ) -> tuple[int | None, int | None]:
        starts = [entity.original_start_char for entity in entities if entity.original_start_char is not None]
        ends = [entity.original_end_char for entity in entities if entity.original_end_char is not None]
        if not starts or not ends:
            return normalized_start, normalized_end
        return min(starts), max(ends)

    @staticmethod
    def _extract_conditions(text: str) -> tuple[str, ...]:
        lowered = text.lower()
        conditions: list[str] = []
        for marker in ("if", "agar", "provided", "subject to"):
            index = lowered.find(marker)
            if index >= 0:
                conditions.append(text[index:].strip())
        return tuple(conditions)

    @staticmethod
    def _evidence_sentence(text: str, start_char: int, end_char: int) -> str:
        if not text:
            return ""
        left = text.rfind(".", 0, max(0, start_char))
        right = text.find(".", min(len(text), end_char))
        sentence_start = 0 if left < 0 else left + 1
        sentence_end = len(text) if right < 0 else right
        snippet = text[sentence_start:sentence_end].strip()
        return snippet or text[max(0, start_char - 30) : min(len(text), end_char + 30)].strip()


__all__ = ["CommitmentContext", "PrecisionCommitmentExtractor"]
