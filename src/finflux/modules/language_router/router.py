"""Deterministic language routing for transcript normalization and span tagging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from finflux.contracts.events import LanguageLabel, LanguageRoutedEvent, LanguageSpan, TranscriptSegmentEvent


@dataclass(frozen=True)
class LanguageRoutingResult:
    dominant_language: LanguageLabel
    code_switch_score: float
    normalized_text: str
    spans: tuple[LanguageSpan, ...]


class LanguageRouterImpl:
    """Heuristic language router that keeps pipeline deterministic and testable."""

    _HINDI_MARKERS = (
        "karunga",
        "karenge",
        "soch",
        "shayad",
        "pakka",
        "kar diya",
        "ho gaya",
        "bhugtan",
        "lona",
        "mahine",
        "saal",
        "rupee",
        "karo",
    )
    _ENGLISH_MARKERS = (
        "will",
        "shall",
        "must",
        "should",
        "plan",
        "think",
        "payment",
        "loan",
        "emi",
        "interest",
        "transfer",
    )

    def route(self, transcript: TranscriptSegmentEvent) -> LanguageRoutedEvent:
        if not transcript.text.strip():
            raise ValueError("Transcript text cannot be empty")
        result = self._route_text(transcript.text)
        return LanguageRoutedEvent(
            envelope=transcript.envelope,
            segment_id=transcript.segment_id,
            dominant_language=result.dominant_language,
            code_switch_score=result.code_switch_score,
            normalized_text=result.normalized_text,
            spans=result.spans,
        )

    def _route_text(self, text: str) -> LanguageRoutingResult:
        normalized_text = self._normalize_text(text)
        spans = self._build_spans(normalized_text)
        dominant = self._dominant_language(normalized_text)
        code_switch_score = self._code_switch_score(normalized_text, spans)
        return LanguageRoutingResult(
            dominant_language=dominant,
            code_switch_score=code_switch_score,
            normalized_text=normalized_text,
            spans=spans,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _build_spans(self, text: str) -> tuple[LanguageSpan, ...]:
        words = list(re.finditer(r"\S+", text))
        spans: list[LanguageSpan] = []
        for match in words:
            token = match.group(0).lower()
            language = self._label_token(token)
            spans.append(LanguageSpan(start_char=match.start(), end_char=match.end(), language=language))
        return tuple(spans)

    def _dominant_language(self, text: str) -> LanguageLabel:
        hindi = 0
        english = 0
        for word in re.findall(r"\S+", text.lower()):
            label = self._label_token(word)
            if label == "hi":
                hindi += 1
            elif label == "en":
                english += 1
        if hindi and english:
            return "mixed"
        if hindi:
            return "hi"
        if english:
            return "en"
        return "unknown"

    def _code_switch_score(self, text: str, spans: Sequence[LanguageSpan]) -> float:
        if not spans:
            return 0.0
        languages = {span.language for span in spans if span.language != "unknown"}
        if len(languages) <= 1:
            return 0.0
        return min(1.0, len(languages) / 3.0)

    def _label_token(self, token: str) -> LanguageLabel:
        token_lower = token.lower()
        if any(marker in token_lower for marker in self._HINDI_MARKERS):
            return "hi"
        if any(marker in token_lower for marker in self._ENGLISH_MARKERS):
            return "en"
        if re.search(r"[\u0900-\u097f]", token):
            return "hi"
        if re.search(r"[a-z]", token_lower):
            return "en"
        return "unknown"


__all__ = ["LanguageRouterImpl", "LanguageRoutingResult"]
