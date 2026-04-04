"""Normalization layer for financial NER processing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationResult:
    """Holds normalized text and optional index mapping to original text."""

    normalized_text: str
    normalized_applied: bool

    def map_normalized_to_original(self, start_char: int, end_char: int) -> tuple[int | None, int | None]:
        # Current normalization is length-preserving for supported transforms.
        if start_char < 0 or end_char < start_char:
            return None, None
        return start_char, end_char


class NormalizationLayer:
    """Produces a normalization-safe copy without mutating source transcript text."""

    def normalize(self, text: str) -> NormalizationResult:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return NormalizationResult(
            normalized_text=normalized,
            normalized_applied=(normalized != text),
        )


__all__ = ["NormalizationLayer", "NormalizationResult"]
