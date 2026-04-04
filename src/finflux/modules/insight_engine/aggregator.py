"""Commitment aggregation logic for conversation-level intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from finflux.contracts.events import CommitmentCandidate
from finflux.contracts.interfaces import InsightInput


@dataclass(frozen=True)
class AggregatedCommitment:
    key: tuple[str, str, str, str | None]
    commitment_ids: tuple[str, ...]
    actor: str
    action: str
    target: str
    due_date_iso: str | None
    commitment_level: str
    confidence: float
    evidence_sentences: tuple[str, ...]
    supporting_segment_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    has_ambiguity: bool
    fulfilled: bool
    has_conflict: bool


@dataclass
class _AggregationBucket:
    commitment_ids: set[str]
    supporting_segment_ids: set[str]
    entity_ids: set[str]
    evidence_sentences: set[str]
    max_confidence: float
    commitment_level: str
    actor: str
    action: str
    target: str
    due_date_iso: str | None
    has_ambiguity: bool
    fulfilled: bool


class CommitmentAggregator:
    """Groups commitments across segments and resolves duplicates/conflicts."""

    def aggregate(self, inputs: Sequence[InsightInput]) -> tuple[AggregatedCommitment, ...]:
        groups: dict[tuple[str, str, str, str | None], _AggregationBucket] = {}
        conflict_tracker: dict[tuple[str, str, str], set[str | None]] = {}

        for item in inputs:
            unresolved = bool(item.commitment_event.unresolved_reasons)
            for candidate in item.commitment_event.candidates:
                key = self._group_key(candidate)
                base_key = (candidate.actor, candidate.action, candidate.target)
                conflict_tracker.setdefault(base_key, set()).add(candidate.due_date_iso)

                bucket = groups.setdefault(
                    key,
                    _AggregationBucket(
                        commitment_ids=set(),
                        supporting_segment_ids=set(),
                        entity_ids=set(),
                        evidence_sentences=set(),
                        max_confidence=0.0,
                        commitment_level=candidate.commitment_level,
                        actor=candidate.actor,
                        action=candidate.action,
                        target=candidate.target,
                        due_date_iso=candidate.due_date_iso,
                        has_ambiguity=False,
                        fulfilled=False,
                    ),
                )

                bucket.commitment_ids.add(candidate.commitment_id)
                bucket.supporting_segment_ids.add(candidate.evidence_segment_id)
                bucket.entity_ids.update(candidate.entity_ids_used)
                if candidate.evidence_sentence:
                    bucket.evidence_sentences.add(candidate.evidence_sentence)
                bucket.max_confidence = max(bucket.max_confidence, candidate.confidence)

                bucket.commitment_level = self._merge_level(
                    bucket.commitment_level, candidate.commitment_level
                )
                bucket.has_ambiguity = bool(bucket.has_ambiguity or unresolved)
                bucket.fulfilled = bool(
                    bucket.fulfilled or candidate.commitment_level == "FULFILLED"
                )

        aggregated: list[AggregatedCommitment] = []
        for key, bucket in groups.items():
            base_key = (key[0], key[1], key[2])
            has_conflict = len(conflict_tracker.get(base_key, set())) > 1

            confidence = float(bucket.max_confidence)
            if has_conflict:
                confidence = max(0.0, confidence - 0.12)
            if bucket.has_ambiguity:
                confidence = max(0.0, confidence - 0.1)

            aggregated.append(
                AggregatedCommitment(
                    key=key,
                    commitment_ids=tuple(sorted(bucket.commitment_ids)),
                    actor=bucket.actor,
                    action=bucket.action,
                    target=bucket.target,
                    due_date_iso=bucket.due_date_iso,
                    commitment_level=bucket.commitment_level,
                    confidence=confidence,
                    evidence_sentences=tuple(sorted(bucket.evidence_sentences)),
                    supporting_segment_ids=tuple(sorted(bucket.supporting_segment_ids)),
                    entity_ids=tuple(sorted(bucket.entity_ids)),
                    has_ambiguity=bucket.has_ambiguity,
                    fulfilled=bucket.fulfilled,
                    has_conflict=has_conflict,
                )
            )

        return tuple(sorted(aggregated, key=lambda item: (item.actor, item.action, item.target)))

    @staticmethod
    def _group_key(candidate: CommitmentCandidate) -> tuple[str, str, str, str | None]:
        return (candidate.actor, candidate.action, candidate.target, candidate.due_date_iso)

    @staticmethod
    def _merge_level(existing: str, incoming: str) -> str:
        priority = {
            "OBSERVATION": 1,
            "SUGGESTION": 2,
            "INTENTION": 3,
            "DECISION": 4,
            "FULFILLED": 5,
        }
        return incoming if priority.get(incoming, 0) >= priority.get(existing, 0) else existing


__all__ = ["AggregatedCommitment", "CommitmentAggregator"]
