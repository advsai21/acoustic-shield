"""
app/ingestion/audio.py

FR-1: Audio ingestion / preprocessing pipeline.

No equivalent module existed in the repo before this backend, so this
is new code (not a duplicate of anything teammates built).

Pipeline:
    decode_audio(raw_bytes)                -> (waveform, sample_rate)
    preprocess_audio(waveform, sample_rate) -> (mono_waveform_16k_float32, target_sr)
    split_into_windows(waveform, sr)        -> List[np.ndarray]

Denoising is NOT implemented. `denoise_audio()` below is a clean,
explicitly-labeled no-op hook so a real denoising stage can be dropped
in later without changing any caller.
"""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

import numpy as np
import soundfile as sf

from app.config import get_settings

logger = logging.getLogger(__name__)


class AudioIngestionError(ValueError):
    """Raised for invalid/empty/unsupported audio input."""


def decode_audio(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode uploaded audio bytes into a waveform + sample rate.

    Args:
        raw_bytes: raw file content (wav/flac/ogg — whatever libsndfile
            supports via soundfile).

    Returns:
        (waveform, sample_rate) where waveform is a numpy array, shape
        (n_samples,) for mono or (n_samples, n_channels) for multi-channel.

    Raises:
        AudioIngestionError: if the bytes are empty or not decodable.
    """
    if not raw_bytes:
        raise AudioIngestionError("Empty audio payload.")

    try:
        waveform, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioIngestionError(f"Could not decode audio file: {exc}") from exc

    if waveform is None or waveform.size == 0:
        raise AudioIngestionError("Decoded audio contains no samples.")

    return waveform, sample_rate


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Average channels down to mono. No-op if already mono."""
    if waveform.ndim == 1:
        return waveform
    return np.mean(waveform, axis=1)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target_sr. No-op if already at target_sr."""
    if orig_sr == target_sr:
        return waveform.astype(np.float32)

    # librosa is already a hard dependency of prosody.py / spoof_heuristic.py,
    # so reuse it here rather than adding a new resampling dependency.
    import librosa

    resampled = librosa.resample(
        waveform.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    )
    return resampled.astype(np.float32)


def _normalize(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Safely peak-normalize to `peak` (default 0.95, leaving headroom).
    No-op on silence to avoid dividing by zero / amplifying noise.
    """
    max_abs = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if max_abs < 1e-6:
        return waveform.astype(np.float32)
    return (waveform * (peak / max_abs)).astype(np.float32)


def denoise_audio(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Placeholder hook for a future denoising stage.

    NOT IMPLEMENTED. Returns the input unchanged. Do not claim
    denoising happens anywhere that calls this — `audio.denoise` in
    config.yaml exists so callers can see it's explicitly off.
    """
    return waveform


def preprocess_audio(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Convert stereo -> mono, resample to target_sample_rate, normalize.

    Args:
        waveform: decoded waveform (mono or multi-channel)
        sample_rate: original sample rate
        target_sample_rate: defaults to config.yaml's audio.target_sample_rate

    Returns:
        (mono_waveform_float32, target_sample_rate)
    """
    settings = get_settings()
    target_sr = target_sample_rate or settings.audio.target_sample_rate

    mono = _to_mono(waveform)
    resampled = _resample(mono, sample_rate, target_sr)

    if settings.audio.denoise:
        resampled = denoise_audio(resampled, target_sr)

    normalized = _normalize(resampled)
    return normalized, target_sr


def split_into_windows(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float | None = None,
) -> List[np.ndarray]:
    """
    Split a mono waveform into fixed-length windows.

    Uses config.yaml's audio.window_seconds by default (2.0s), clamped
    between min_window_seconds and max_window_seconds. The final
    partial window is kept only if it meets min_window_seconds;
    otherwise it's merged into the previous window so no audio is
    silently dropped.

    Args:
        waveform: mono float32 waveform, already resampled/normalized
        sample_rate: sample rate of `waveform`
        window_seconds: override for the configured window size

    Returns:
        List of 1D numpy arrays, each one window. Always has at least
        one element for non-empty input.
    """
    settings = get_settings()
    win_s = window_seconds or settings.audio.window_seconds
    win_s = max(settings.audio.min_window_seconds, min(win_s, settings.audio.max_window_seconds))

    window_samples = int(round(win_s * sample_rate))
    min_samples = int(round(settings.audio.min_window_seconds * sample_rate))

    if window_samples <= 0:
        raise AudioIngestionError("Computed window size is <= 0 samples.")

    total_samples = len(waveform)
    if total_samples == 0:
        raise AudioIngestionError("Cannot window empty audio.")

    windows: List[np.ndarray] = []
    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        windows.append(waveform[start:end])
        start = end

    # If the trailing window is too short to be useful on its own,
    # merge it into the previous window instead of discarding it.
    if len(windows) > 1 and len(windows[-1]) < min_samples:
        last = windows.pop()
        windows[-1] = np.concatenate([windows[-1], last])

    return windows


def ingest(raw_bytes: bytes) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Convenience end-to-end pipeline: decode -> preprocess -> window.

    Returns:
        (full_waveform, sample_rate, windows)
    """
    waveform, sr = decode_audio(raw_bytes)
    processed, target_sr = preprocess_audio(waveform, sr)
    windows = split_into_windows(processed, target_sr)
    return processed, target_sr, windows
