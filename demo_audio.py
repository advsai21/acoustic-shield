"""
demo_audio.py — generates stand-in audio samples so the full pipeline can
be demoed and tested WITHOUT needing real recordings or a microphone.

IMPORTANT: these are synthetic sine-wave proxies, not real speech and not
real TTS output. They're useful to (a) prove the whole pipeline reacts
correctly end-to-end, and (b) rehearse the demo flow. Before presenting
to judges, replace with:
  - a short real recording of a teammate's voice (genuine sample)
  - a free TTS tool's output of a similar sentence (synthetic sample)
See README.md "Calibrate with real audio" section.
"""

import numpy as np

SR = 16000


def genuine_sample(duration=4.0, sr=SR, seed=0):
    """Stand-in for real speech: wandering pitch, natural jitter/shimmer."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_f0 = 150.0
    f0_contour = base_f0 + 15 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 4, size=t.shape)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    shimmer_noise = 1 + rng.normal(0, 0.05, size=t.shape)
    signal = signal * envelope * shimmer_noise
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def synthetic_sample(duration=4.0, sr=SR):
    """Stand-in for a lower-quality cloned/TTS voice: flat pitch, no jitter."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 150.0
    phase = 2 * np.pi * f0 * t
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def different_speaker_sample(duration=4.0, sr=SR, seed=99):
    """Stand-in for a genuinely different person's voice (different base pitch)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_f0 = 220.0  # noticeably different pitch range
    f0_contour = base_f0 + 20 * np.sin(2 * np.pi * 0.9 * t) + rng.normal(0, 5, size=t.shape)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    signal = np.sin(phase) + 0.4 * np.sin(2 * phase)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.6 * t)
    signal = signal * envelope * (1 + rng.normal(0, 0.05, size=t.shape))
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def to_int16_chunks(waveform: np.ndarray, chunk_size: int = 4800):
    """Split a float32 waveform into int16 PCM chunks, matching mic-capture format."""
    int16_wave = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    chunks = [int16_wave[i:i + chunk_size] for i in range(0, len(int16_wave), chunk_size)]
    return [c for c in chunks if len(c) > 0]
