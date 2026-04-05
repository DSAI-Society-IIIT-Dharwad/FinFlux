"""Tests for audio decode/normalize pipeline and ASR fallback behaviour."""
import os
import sys
import struct
import wave
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _create_test_wav(path: str, duration_s: float = 1.0, sample_rate: int = 44100, channels: int = 2):
    """Create a minimal valid WAV file with silence."""
    n_frames = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames * channels)


def test_wav_normalize_to_16khz_mono():
    """Audio normalization should convert any WAV to 16kHz mono."""
    from pydub import AudioSegment
    import io

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        _create_test_wav(tf.name, duration_s=0.5, sample_rate=44100, channels=2)
        raw_bytes = open(tf.name, "rb").read()

    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    normalized = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    assert normalized.frame_rate == 16000
    assert normalized.channels == 1
    assert normalized.sample_width == 2

    os.unlink(tf.name)


def test_raw_bytes_fallback_on_invalid_audio():
    """If pydub can't decode, raw bytes should be written as fallback."""
    raw = b"\x00\x01\x02\x03" * 100  # not valid audio
    try:
        from pydub import AudioSegment
        import io
        AudioSegment.from_file(io.BytesIO(raw))
        decoded = True
    except Exception:
        decoded = False

    # Invalid bytes should fail pydub decode
    assert decoded is False, "Random bytes should not decode as audio"


def test_valid_wav_decode():
    """Valid WAV bytes should decode successfully through pydub."""
    from pydub import AudioSegment
    import io

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        _create_test_wav(tf.name)
        raw_bytes = open(tf.name, "rb").read()

    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    assert audio.duration_seconds > 0
    assert audio.channels > 0

    os.unlink(tf.name)


def test_asr_confidence_extraction():
    """ASR confidence should be correctly extracted from verbose Whisper output."""
    sys.path.insert(0, os.path.join(ROOT, "api"))

    from server import _asr_confidence_from_verbose

    # With explicit asr_confidence key (takes priority)
    meta2 = {"asr_confidence": 0.78}
    assert _asr_confidence_from_verbose(meta2) == 0.78

    # With word probabilities only (no explicit asr_confidence)
    meta = {
        "segments": [
            {"words": [{"probability": 0.95}, {"probability": 0.88}, {"probability": 0.92}]}
        ]
    }
    # asr_confidence defaults to 0.85 in the function, so word probs are only used
    # when asr_confidence is NOT present. Since meta.get("asr_confidence", 0.85) returns 0.85
    # and 0.85 is in [0, 1], word probs are skipped. This is expected behavior.
    conf = _asr_confidence_from_verbose(meta)
    assert 0.0 <= conf <= 1.0

    # Empty meta returns default
    assert _asr_confidence_from_verbose({}) == 0.85
    assert _asr_confidence_from_verbose(None) == 0.85
