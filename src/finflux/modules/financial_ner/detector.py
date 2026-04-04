"""Rule-based multilingual financial entity detector."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from finflux.contracts.events import EntityType


@dataclass(frozen=True)
class DetectedEntity:
    entity_type: EntityType
    value_text: str
    normalized_value: str
    start_char: int
    end_char: int
    detection_confidence: float


class EntityDetector:
    """Detects finance entities from normalized text using deterministic rules."""

    _AMOUNT_PATTERN = re.compile(
        r"(?P<amount>(?:rs\.?|inr|₹)\s*\d+(?:,\d{2,3})*(?:\.\d+)?|\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:rupees|lakh|lakhs|crore|crores|k))",
        flags=re.IGNORECASE,
    )
    _DATE_PATTERN = re.compile(
        r"(?P<date>\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b)",
        flags=re.IGNORECASE,
    )
    _RATE_PATTERN = re.compile(
        r"(?P<rate>\b\d+(?:\.\d+)?\s*%\b|\b\d+(?:\.\d+)?\s*(?:percent|pa)\b)",
        flags=re.IGNORECASE,
    )
    _TENURE_PATTERN = re.compile(
        r"(?P<tenure>\b\d+\s*(?:day|days|month|months|year|years|mahine|saal)\b)",
        flags=re.IGNORECASE,
    )

    _PRODUCT_TERMS = {
        "loan": "loan",
        "home loan": "home_loan",
        "personal loan": "personal_loan",
        "emi": "emi",
        "sip": "sip",
        "mutual fund": "mutual_fund",
        "credit card": "credit_card",
        "insurance": "insurance",
        "fd": "fixed_deposit",
        "fixed deposit": "fixed_deposit",
        "ऋण": "loan",
        "लोन": "loan",
    }

    _OBLIGATION_TERMS = {
        "pay": "payment_commitment",
        "repay": "payment_commitment",
        "transfer": "transfer_commitment",
        "submit": "document_commitment",
        "clear dues": "payment_commitment",
        "भुगतान": "payment_commitment",
        "जमा": "deposit_commitment",
    }

    _PARTY_PATTERN = re.compile(
        r"\b(?:mr|mrs|ms|shri|smt)\.?\s+[a-z]+\b|\b(?:customer|advisor|manager|borrower|lender|applicant)\b",
        flags=re.IGNORECASE,
    )

    def detect(self, text: str, dominant_language: str) -> Sequence[DetectedEntity]:
        entities: list[DetectedEntity] = []

        entities.extend(self._detect_regex(self._AMOUNT_PATTERN, text, "amount", 0.92))
        entities.extend(self._detect_regex(self._DATE_PATTERN, text, "date", 0.84))
        entities.extend(self._detect_regex(self._RATE_PATTERN, text, "rate", 0.86))
        entities.extend(self._detect_regex(self._TENURE_PATTERN, text, "tenure", 0.82))
        entities.extend(self._detect_lexicon(text, self._PRODUCT_TERMS, "product", 0.88))
        entities.extend(self._detect_lexicon(text, self._OBLIGATION_TERMS, "obligation", 0.8))
        entities.extend(self._detect_regex(self._PARTY_PATTERN, text, "party", 0.75))

        deduped = self._dedupe(entities)
        return tuple(sorted(deduped, key=lambda item: (item.start_char, item.end_char)))

    def _detect_regex(
        self,
        pattern: re.Pattern[str],
        text: str,
        entity_type: EntityType,
        confidence: float,
    ) -> Sequence[DetectedEntity]:
        found: list[DetectedEntity] = []
        for match in pattern.finditer(text):
            value = match.group(0)
            found.append(
                DetectedEntity(
                    entity_type=entity_type,
                    value_text=value,
                    normalized_value=self._normalize_value(entity_type, value),
                    start_char=match.start(),
                    end_char=match.end(),
                    detection_confidence=confidence,
                )
            )
        return found

    def _detect_lexicon(
        self,
        text: str,
        lexicon: dict[str, str],
        entity_type: EntityType,
        confidence: float,
    ) -> Sequence[DetectedEntity]:
        lowered = text.lower()
        found: list[DetectedEntity] = []
        for term, normalized in lexicon.items():
            for start in self._find_all(lowered, term.lower()):
                end = start + len(term)
                found.append(
                    DetectedEntity(
                        entity_type=entity_type,
                        value_text=text[start:end],
                        normalized_value=normalized,
                        start_char=start,
                        end_char=end,
                        detection_confidence=confidence,
                    )
                )
        return found

    @staticmethod
    def _find_all(text: str, pattern: str) -> Iterable[int]:
        start = 0
        while True:
            index = text.find(pattern, start)
            if index < 0:
                return
            yield index
            start = index + 1

    @staticmethod
    def _normalize_value(entity_type: EntityType, value: str) -> str:
        value_clean = value.strip().lower()
        if entity_type == "amount":
            return value_clean.replace("rs.", "inr ").replace("₹", "inr ")
        if entity_type == "rate":
            return value_clean.replace("percent", "%")
        return value_clean

    @staticmethod
    def _dedupe(entities: Sequence[DetectedEntity]) -> Sequence[DetectedEntity]:
        seen: set[tuple[EntityType, int, int, str]] = set()
        result: list[DetectedEntity] = []
        for entity in entities:
            key = (entity.entity_type, entity.start_char, entity.end_char, entity.normalized_value)
            if key in seen:
                continue
            seen.add(key)
            result.append(entity)
        return result


__all__ = ["DetectedEntity", "EntityDetector"]
