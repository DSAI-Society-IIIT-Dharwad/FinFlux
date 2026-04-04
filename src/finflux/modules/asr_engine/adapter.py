"""Streaming-first ASR adapter with lineage and confidence propagation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Iterator, Sequence

from finflux.contracts.events import AudioChunkEvent, EventEnvelope, TranscriptSegmentEvent, TranscriptToken
from finflux.contracts.flow import validate_vad_to_asr, validate_vad_to_asr_lineage
from finflux.contracts.interfaces import ASREngine
from finflux.contracts.events import SpeechSegmentEvent

from .inference import InferenceEngine
from .model import ModelLoader


@dataclass(frozen=True)
class ASRQualityConfig:
    """Quality handling controls for confidence propagation."""

    low_quality_threshold: float = 0.35
    min_quality_factor: float = 0.4
    max_quality_factor: float = 1.0


class SegmentASRAdapter(ASREngine):
    """ASR adapter processing one speech segment at a time."""

    def __init__(
        self,
        model_loader: ModelLoader,
        inference_engine: InferenceEngine,
        chunk_lookup: Callable[[str], AudioChunkEvent],
        payload_loader: Callable[[str], Sequence[float]],
        quality_config: ASRQualityConfig | None = None,
    ) -> None:
        self._model = model_loader.load()
        self._inference = inference_engine
        self._chunk_lookup = chunk_lookup
        self._payload_loader = payload_loader
        self._quality = quality_config or ASRQualityConfig()

    def transcribe(self, segment: SpeechSegmentEvent) -> TranscriptSegmentEvent:
        chunk = self._chunk_lookup(segment.source_chunk_event_id)
        samples = self._slice_segment_samples(segment, chunk)

        prediction = self._inference.infer(
            model=self._model,
            samples=samples,
            sample_rate_hz=chunk.sample_rate_hz,
            segment=segment,
        )

        quality_factor = self._derive_quality_factor(segment.quality_score)
        low_quality = segment.quality_score < self._quality.low_quality_threshold

        transcript_tokens = tuple(
            TranscriptToken(
                token=token.token,
                start_ms=token.start_ms,
                end_ms=token.end_ms,
                confidence=max(0.0, min(1.0, token.confidence * quality_factor)),
            )
            for token in prediction.tokens
        )

        token_mean = fmean(token.confidence for token in transcript_tokens) if transcript_tokens else 0.0
        avg_confidence = max(0.0, min(1.0, token_mean * quality_factor))

        transcript = TranscriptSegmentEvent(
            envelope=EventEnvelope(
                trace_id=segment.envelope.trace_id,
                call_id=segment.envelope.call_id,
            ),
            segment_id=segment.segment_id,
            source_speech_segment_id=segment.segment_id,
            text=prediction.text,
            avg_confidence=avg_confidence,
            language_hint=prediction.language_hint,
            tokens=transcript_tokens,
            low_quality=low_quality,
        )

        validate_vad_to_asr(transcript)
        validate_vad_to_asr_lineage(segment, transcript)
        return transcript

    def stream_transcribe(
        self, segments: Sequence[SpeechSegmentEvent]
    ) -> Iterator[TranscriptSegmentEvent]:
        for segment in segments:
            yield self.transcribe(segment)

    def _slice_segment_samples(
        self, segment: SpeechSegmentEvent, chunk: AudioChunkEvent
    ) -> list[float]:
        chunk_samples = list(self._payload_loader(chunk.payload_uri))
        if not chunk_samples:
            return []

        start_ms = max(segment.start_ms, chunk.start_ms)
        end_ms = min(segment.end_ms, chunk.end_ms)
        if end_ms <= start_ms:
            return []

        start_offset_ms = start_ms - chunk.start_ms
        end_offset_ms = end_ms - chunk.start_ms

        start_index = int((start_offset_ms * chunk.sample_rate_hz) / 1000)
        end_index = int((end_offset_ms * chunk.sample_rate_hz) / 1000)

        start_index = max(0, min(start_index, len(chunk_samples)))
        end_index = max(start_index + 1, min(end_index, len(chunk_samples)))
        return chunk_samples[start_index:end_index]

    def _derive_quality_factor(self, quality_score: float) -> float:
        clipped = max(0.0, min(1.0, quality_score))
        span = self._quality.max_quality_factor - self._quality.min_quality_factor
        return self._quality.min_quality_factor + clipped * span


__all__ = ["ASRQualityConfig", "SegmentASRAdapter"]
