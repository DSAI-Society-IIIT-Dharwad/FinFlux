"""Deterministic energy-based VAD processing for chunk events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from uuid import NAMESPACE_URL, uuid5

from finflux.contracts.events import AudioChunkEvent, EventEnvelope, SpeechSegmentEvent
from finflux.contracts.flow import validate_audio_to_vad
from finflux.contracts.interfaces import VADProcessor


@dataclass(frozen=True)
class VADConfig:
    """Energy VAD settings optimized for deterministic behavior."""

    frame_ms: int = 20
    min_speech_ms: int = 200
    min_silence_ms: int = 120
    energy_threshold: float = 0.02
    overlap_energy_threshold: float = 0.45


class EnergyVADProcessor(VADProcessor):
    """Detects speech segments from normalized chunk payload energy."""

    def __init__(
        self,
        payload_loader: Callable[[str], Sequence[float]],
        config: VADConfig | None = None,
    ) -> None:
        self._payload_loader = payload_loader
        self._config = config or VADConfig()

    def detect(self, chunk: AudioChunkEvent) -> Sequence[SpeechSegmentEvent]:
        samples = list(self._payload_loader(chunk.payload_uri))
        if not samples:
            return []

        frame_samples = max(1, int(chunk.sample_rate_hz * self._config.frame_ms / 1000))
        frame_energies = self._frame_energies(samples, frame_samples)
        flags = [energy >= self._config.energy_threshold for energy in frame_energies]

        min_speech_frames = max(1, self._config.min_speech_ms // self._config.frame_ms)
        min_silence_frames = max(1, self._config.min_silence_ms // self._config.frame_ms)

        segments: list[SpeechSegmentEvent] = []
        start_idx: int | None = None
        silence_run = 0

        for idx, is_speech in enumerate(flags):
            if is_speech:
                if start_idx is None:
                    start_idx = idx
                silence_run = 0
                continue

            if start_idx is None:
                continue

            silence_run += 1
            if silence_run < min_silence_frames:
                continue

            end_idx = idx - silence_run + 1
            self._append_segment(
                segments=segments,
                chunk=chunk,
                frame_energies=frame_energies,
                start_frame=start_idx,
                end_frame=end_idx,
                min_speech_frames=min_speech_frames,
            )
            start_idx = None
            silence_run = 0

        if start_idx is not None:
            self._append_segment(
                segments=segments,
                chunk=chunk,
                frame_energies=frame_energies,
                start_frame=start_idx,
                end_frame=len(frame_energies),
                min_speech_frames=min_speech_frames,
            )

        return segments

    def _append_segment(
        self,
        segments: list[SpeechSegmentEvent],
        chunk: AudioChunkEvent,
        frame_energies: Sequence[float],
        start_frame: int,
        end_frame: int,
        min_speech_frames: int,
    ) -> None:
        frame_count = end_frame - start_frame
        if frame_count < min_speech_frames:
            return

        start_ms = chunk.start_ms + start_frame * self._config.frame_ms
        end_ms = chunk.start_ms + end_frame * self._config.frame_ms

        segment_energies = frame_energies[start_frame:end_frame]
        mean_energy = sum(segment_energies) / len(segment_energies)
        quality = self._quality_score(mean_energy)
        is_overlap = mean_energy >= self._config.overlap_energy_threshold

        segment_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{chunk.envelope.trace_id}:{chunk.envelope.event_id}:{start_ms}:{end_ms}",
            )
        )

        segment = SpeechSegmentEvent(
            envelope=EventEnvelope(
                trace_id=chunk.envelope.trace_id,
                call_id=chunk.envelope.call_id,
            ),
            segment_id=segment_id,
            source_chunk_event_id=chunk.envelope.event_id,
            speaker_id=None,
            start_ms=start_ms,
            end_ms=end_ms,
            is_overlap=is_overlap,
            quality_score=quality,
        )
        validate_audio_to_vad(segment)
        segments.append(segment)

    @staticmethod
    def _frame_energies(samples: Sequence[float], frame_samples: int) -> list[float]:
        energies: list[float] = []
        for offset in range(0, len(samples), frame_samples):
            frame = samples[offset : offset + frame_samples]
            if not frame:
                continue
            energies.append(sum(abs(sample) for sample in frame) / len(frame))
        return energies

    def _quality_score(self, mean_energy: float) -> float:
        normalized = mean_energy / max(self._config.energy_threshold, 1e-6)
        return float(max(0.0, min(1.0, normalized / 4.0)))


__all__ = ["EnergyVADProcessor", "VADConfig"]
