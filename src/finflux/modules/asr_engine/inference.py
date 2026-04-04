"""Replaceable inference contracts and a deterministic mock inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from finflux.contracts.events import LanguageLabel, SpeechSegmentEvent
from .model import ASRModelHandle


@dataclass(frozen=True)
class TokenPrediction:
    token: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass(frozen=True)
class ASRPrediction:
    text: str
    tokens: Sequence[TokenPrediction]
    language_hint: LanguageLabel


class InferenceEngine(Protocol):
    """Runs ASR inference for a single speech segment."""

    def infer(
        self,
        model: ASRModelHandle,
        samples: Sequence[float],
        sample_rate_hz: int,
        segment: SpeechSegmentEvent,
    ) -> ASRPrediction:
        ...


class MockInferenceEngine:
    """Simple deterministic inference implementation for contract-safe testing."""

    def infer(
        self,
        model: ASRModelHandle,
        samples: Sequence[float],
        sample_rate_hz: int,
        segment: SpeechSegmentEvent,
    ) -> ASRPrediction:
        if not samples:
            return ASRPrediction(
                text="[silence]",
                tokens=(
                    TokenPrediction(token="[silence]", start_ms=segment.start_ms, end_ms=segment.end_ms, confidence=0.3),
                ),
                language_hint="unknown",
            )

        duration_ms = max(1, segment.end_ms - segment.start_ms)
        token_count = max(1, min(8, len(samples) // max(1, sample_rate_hz // 5)))
        token_duration = max(1, duration_ms // token_count)

        avg_energy = sum(abs(sample) for sample in samples) / len(samples)
        base_confidence = max(0.35, min(0.95, 0.45 + avg_energy))

        tokens: list[TokenPrediction] = []
        for index in range(token_count):
            token_start = segment.start_ms + index * token_duration
            token_end = (
                segment.end_ms
                if index == token_count - 1
                else min(segment.end_ms, token_start + token_duration)
            )
            token_text = f"tok_{index + 1}"
            confidence = max(0.0, min(1.0, base_confidence - (index * 0.01)))
            tokens.append(
                TokenPrediction(
                    token=token_text,
                    start_ms=token_start,
                    end_ms=token_end,
                    confidence=confidence,
                )
            )

        return ASRPrediction(
            text=" ".join(token.token for token in tokens),
            tokens=tokens,
            language_hint="unknown",
        )


__all__ = ["ASRPrediction", "InferenceEngine", "MockInferenceEngine", "TokenPrediction"]
