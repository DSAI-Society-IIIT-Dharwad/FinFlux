"""Rule-based commitment detection for high-precision extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from finflux.contracts.events import CommitmentLevel, FinancialEntity


@dataclass(frozen=True)
class DetectedCommitment:
    commitment_level: CommitmentLevel
    phrase: str
    start_char: int
    end_char: int
    phrase_strength: float


class CommitmentDetector:
    """Detects commitment-level signals from normalized text and entity context."""

    _LEVEL_MARKERS: dict[CommitmentLevel, tuple[tuple[str, ...], float]] = {
        "DECISION": (("will", "shall", "must", "karunga", "karenge", "pakka", "confirmed"), 0.9),
        "INTENTION": (("plan to", "thinking of", "soch raha", "intend", "considering"), 0.72),
        "SUGGESTION": (("should", "maybe", "try", "shayad", "recommend"), 0.5),
        "FULFILLED": (("done", "paid", "kar diya", "ho gaya", "cleared"), 0.95),
    }

    _ACTION_TERMS = (
        "pay",
        "repay",
        "transfer",
        "submit",
        "clear",
        "deposit",
        "settle",
        "भुगतान",
        "जमा",
        "दे",
    )

    def detect(self, text: str, entities: Sequence[FinancialEntity]) -> Sequence[DetectedCommitment]:
        lowered = text.lower()
        detections: list[DetectedCommitment] = []

        for level, (markers, strength) in self._LEVEL_MARKERS.items():
            for marker in markers:
                start = lowered.find(marker)
                while start >= 0:
                    end = start + len(marker)
                    if self._is_valid_commitment_context(level, lowered, entities):
                        detections.append(
                            DetectedCommitment(
                                commitment_level=level,
                                phrase=text[start:end],
                                start_char=start,
                                end_char=end,
                                phrase_strength=strength,
                            )
                        )
                    start = lowered.find(marker, start + 1)

        if detections:
            return tuple(sorted(self._dedupe(detections), key=lambda item: item.start_char))

        # High-precision fallback: only observation when explicit financial entities exist.
        if entities:
            return (
                DetectedCommitment(
                    commitment_level="OBSERVATION",
                    phrase=text,
                    start_char=0,
                    end_char=len(text),
                    phrase_strength=0.35,
                ),
            )

        return ()

    def _is_valid_commitment_context(
        self,
        level: CommitmentLevel,
        lowered_text: str,
        entities: Sequence[FinancialEntity],
    ) -> bool:
        if level in ("DECISION", "INTENTION", "FULFILLED"):
            has_entity_support = any(entity.entity_type in {"amount", "date", "tenure", "product", "obligation"} for entity in entities)
            has_action_support = any(term in lowered_text for term in self._ACTION_TERMS)
            return has_entity_support or has_action_support
        if level == "SUGGESTION":
            return True
        return bool(entities)

    @staticmethod
    def _dedupe(detections: Sequence[DetectedCommitment]) -> Sequence[DetectedCommitment]:
        seen: set[tuple[CommitmentLevel, int, int]] = set()
        unique: list[DetectedCommitment] = []
        for detection in detections:
            key = (detection.commitment_level, detection.start_char, detection.end_char)
            if key in seen:
                continue
            seen.add(key)
            unique.append(detection)
        return unique


__all__ = ["CommitmentDetector", "DetectedCommitment"]
