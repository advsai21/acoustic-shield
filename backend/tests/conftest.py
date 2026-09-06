"""
tests/conftest.py

Shared fixtures. Uses synthetic sine-wave audio (like the repo's own
demo_audio.py) so tests don't depend on real recordings.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf


def make_wav_bytes(
    duration_seconds: float = 3.0,
    sample_rate: int = 22050,
    frequency: float = 220.0,
    channels: int = 1,
) -> bytes:
    """Generate an in-memory WAV file of a sine tone for use as test audio."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    if channels == 2:
        tone = np.stack([tone, tone], axis=1)

    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def mono_wav_bytes() -> bytes:
    return make_wav_bytes(duration_seconds=3.0, channels=1)


@pytest.fixture
def stereo_wav_bytes() -> bytes:
    return make_wav_bytes(duration_seconds=3.0, channels=2)


@pytest.fixture
def short_wav_bytes() -> bytes:
    """Audio shorter than min_window_seconds, to exercise edge handling."""
    return make_wav_bytes(duration_seconds=0.3, channels=1)
