"""Deterministic streaming audio capture implementations.

This module keeps I/O adapters separate from normalization and chunking so the
pipeline remains testable without live hardware dependencies.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

from finflux.contracts.events import AudioChunkEvent, EventEnvelope
from finflux.contracts.interfaces import AudioCapture


@dataclass(frozen=True)
class RawAudioFrame:
    """Source frame carrying raw PCM bytes and source format metadata."""

    pcm_bytes: bytes
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


class AudioFrameSource(Protocol):
    """Streaming source abstraction for microphone or file adapters."""

    def frames(self) -> Iterator[RawAudioFrame]:
        ...


class PayloadStore(Protocol):
    """Stores normalized chunk payloads and returns stable payload URIs."""

    def put(self, payload: Sequence[float], trace_id: str, chunk_index: int) -> str:
        ...

    def get(self, payload_uri: str) -> Sequence[float]:
        ...


class InMemoryPayloadStore:
    """In-memory payload store used for deterministic testing and local runs."""

    def __init__(self) -> None:
        self._payloads: Dict[str, tuple[float, ...]] = {}

    def put(self, payload: Sequence[float], trace_id: str, chunk_index: int) -> str:
        payload_uri = f"memory://{trace_id}/chunk/{chunk_index}"
        self._payloads[payload_uri] = tuple(payload)
        return payload_uri

    def get(self, payload_uri: str) -> Sequence[float]:
        try:
            return self._payloads[payload_uri]
        except KeyError as exc:
            raise KeyError(f"Unknown payload URI: {payload_uri}") from exc


@dataclass(frozen=True)
class CaptureConfig:
    """Capture settings for canonical audio event generation."""

    stream_id: str = "default-stream"
    call_id: str = ""
    trace_id: str = ""
    chunk_size_samples: int = 1600
    target_sample_rate_hz: int = 16000
    target_channels: int = 1
    target_encoding: str = "float32"


class AudioNormalizer:
    """Normalizes arbitrary PCM frames into canonical 16k mono float samples."""

    def __init__(self, target_sample_rate_hz: int = 16000) -> None:
        self._target_sample_rate_hz = target_sample_rate_hz

    def normalize(self, frame: RawAudioFrame) -> list[float]:
        samples = self._decode_pcm(frame.pcm_bytes, frame.sample_width_bytes, frame.channels)
        return self._resample_linear(samples, frame.sample_rate_hz, self._target_sample_rate_hz)

    def _decode_pcm(self, pcm: bytes, sample_width: int, channels: int) -> list[float]:
        if sample_width not in (1, 2, 4):
            raise ValueError(f"Unsupported sample width: {sample_width}")
        if channels < 1:
            raise ValueError("channels must be >= 1")

        frame_width = sample_width * channels
        if frame_width == 0 or len(pcm) % frame_width != 0:
            raise ValueError("Invalid PCM frame size")

        frame_count = len(pcm) // frame_width
        mono_samples: list[float] = []

        for index in range(frame_count):
            frame_bytes = pcm[index * frame_width : (index + 1) * frame_width]
            channel_values = [
                self._decode_sample(frame_bytes[ch * sample_width : (ch + 1) * sample_width], sample_width)
                for ch in range(channels)
            ]
            mono_samples.append(sum(channel_values) / channels)

        return mono_samples

    @staticmethod
    def _decode_sample(sample_bytes: bytes, width: int) -> float:
        if width == 1:
            value = sample_bytes[0] - 128
            return max(-1.0, min(1.0, value / 128.0))
        if width == 2:
            value = struct.unpack("<h", sample_bytes)[0]
            return max(-1.0, min(1.0, value / 32768.0))

        value = struct.unpack("<i", sample_bytes)[0]
        return max(-1.0, min(1.0, value / 2147483648.0))

    @staticmethod
    def _resample_linear(samples: Sequence[float], source_rate: int, target_rate: int) -> list[float]:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        if not samples:
            return []
        if source_rate == target_rate:
            return [float(sample) for sample in samples]

        ratio = target_rate / source_rate
        output_length = max(1, int(round(len(samples) * ratio)))
        output: list[float] = []

        for out_index in range(output_length):
            src_position = out_index / ratio
            low = int(math.floor(src_position))
            high = min(low + 1, len(samples) - 1)
            blend = src_position - low
            interpolated = samples[low] * (1.0 - blend) + samples[high] * blend
            output.append(float(interpolated))

        return output


class AudioFileInputAdapter:
    """Streams raw PCM frames from a wave file."""

    def __init__(self, file_path: str | Path, read_frames: int = 2048) -> None:
        self._file_path = Path(file_path)
        self._read_frames = read_frames

    def frames(self) -> Iterator[RawAudioFrame]:
        with wave.open(str(self._file_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            while True:
                pcm_bytes = wav_file.readframes(self._read_frames)
                if not pcm_bytes:
                    break
                yield RawAudioFrame(
                    pcm_bytes=pcm_bytes,
                    sample_rate_hz=sample_rate,
                    channels=channels,
                    sample_width_bytes=sample_width,
                )


class MicrophoneInputAdapter:
    """Streams frames from an injected provider or optional hardware backend."""

    def __init__(
        self,
        frame_provider: Callable[[], Iterator[RawAudioFrame]] | None = None,
        device_name: str | None = None,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        block_size: int = 1024,
    ) -> None:
        self._frame_provider = frame_provider
        self._device_name = device_name
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._block_size = block_size

    def frames(self) -> Iterator[RawAudioFrame]:
        if self._frame_provider is not None:
            yield from self._frame_provider()
            return

        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for live microphone capture, "
                "or inject frame_provider for tests"
            ) from exc

        with sd.RawInputStream(
            samplerate=self._sample_rate_hz,
            blocksize=self._block_size,
            channels=self._channels,
            dtype="int16",
            device=self._device_name,
        ) as stream:
            while True:
                pcm_bytes, overflowed = stream.read(self._block_size)
                if overflowed:
                    continue
                yield RawAudioFrame(
                    pcm_bytes=bytes(pcm_bytes),
                    sample_rate_hz=self._sample_rate_hz,
                    channels=self._channels,
                    sample_width_bytes=2,
                )


class StreamAudioCapture(AudioCapture):
    """Streaming-first capture implementation emitting canonical chunk events."""

    def __init__(
        self,
        source: AudioFrameSource,
        payload_store: PayloadStore,
        config: CaptureConfig | None = None,
        normalizer: AudioNormalizer | None = None,
    ) -> None:
        self._source = source
        self._payload_store = payload_store
        self._config = config or CaptureConfig()
        self._normalizer = normalizer or AudioNormalizer(
            target_sample_rate_hz=self._config.target_sample_rate_hz
        )

    def stream(self) -> Iterator[AudioChunkEvent]:
        trace_id = self._config.trace_id or str(uuid4())
        envelope_base = EventEnvelope(trace_id=trace_id, call_id=self._config.call_id)

        chunk_index = 0
        emitted_samples = 0
        sample_buffer: list[float] = []

        for frame in self._source.frames():
            normalized = self._normalizer.normalize(frame)
            sample_buffer.extend(normalized)

            while len(sample_buffer) >= self._config.chunk_size_samples:
                chunk_samples = sample_buffer[: self._config.chunk_size_samples]
                sample_buffer = sample_buffer[self._config.chunk_size_samples :]

                payload_uri = self._payload_store.put(chunk_samples, trace_id, chunk_index)
                start_ms = int((emitted_samples * 1000) / self._config.target_sample_rate_hz)
                emitted_samples += len(chunk_samples)
                end_ms = int((emitted_samples * 1000) / self._config.target_sample_rate_hz)

                chunk_event = AudioChunkEvent(
                    envelope=EventEnvelope(
                        trace_id=envelope_base.trace_id,
                        call_id=envelope_base.call_id,
                    ),
                    stream_id=self._config.stream_id,
                    chunk_index=chunk_index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    sample_rate_hz=self._config.target_sample_rate_hz,
                    channels=self._config.target_channels,
                    encoding=self._config.target_encoding,
                    payload_uri=payload_uri,
                )
                yield chunk_event
                chunk_index += 1

        if sample_buffer:
            payload_uri = self._payload_store.put(sample_buffer, trace_id, chunk_index)
            start_ms = int((emitted_samples * 1000) / self._config.target_sample_rate_hz)
            emitted_samples += len(sample_buffer)
            end_ms = int((emitted_samples * 1000) / self._config.target_sample_rate_hz)

            yield AudioChunkEvent(
                envelope=EventEnvelope(trace_id=trace_id, call_id=self._config.call_id),
                stream_id=self._config.stream_id,
                chunk_index=chunk_index,
                start_ms=start_ms,
                end_ms=end_ms,
                sample_rate_hz=self._config.target_sample_rate_hz,
                channels=self._config.target_channels,
                encoding=self._config.target_encoding,
                payload_uri=payload_uri,
            )

    def capture(self) -> Sequence[AudioChunkEvent]:
        return list(self.stream())


__all__ = [
    "AudioFileInputAdapter",
    "AudioFrameSource",
    "AudioNormalizer",
    "CaptureConfig",
    "InMemoryPayloadStore",
    "MicrophoneInputAdapter",
    "PayloadStore",
    "RawAudioFrame",
    "StreamAudioCapture",
]
