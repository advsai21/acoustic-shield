"""
Tests for app/ingestion/audio.py: decoding, mono conversion,
resampling, and window splitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion.audio import (
    AudioIngestionError,
    decode_audio,
    preprocess_audio,
    split_into_windows,
)


def test_decode_audio_mono(mono_wav_bytes):
    waveform, sr = decode_audio(mono_wav_bytes)
    assert sr == 22050
    assert waveform.ndim == 1
    assert len(waveform) > 0


def test_decode_audio_rejects_empty_bytes():
    with pytest.raises(AudioIngestionError):
        decode_audio(b"")


def test_decode_audio_rejects_garbage_bytes():
    with pytest.raises(AudioIngestionError):
        decode_audio(b"this is not a real audio file")


def test_preprocess_converts_stereo_to_mono(stereo_wav_bytes):
    waveform, sr = decode_audio(stereo_wav_bytes)
    assert waveform.ndim == 2  # sanity check the fixture really is stereo

    mono, target_sr = preprocess_audio(waveform, sr, target_sample_rate=16000)
    assert mono.ndim == 1
    assert target_sr == 16000


def test_preprocess_resamples_to_target_rate(mono_wav_bytes):
    waveform, sr = decode_audio(mono_wav_bytes)
    assert sr == 22050

    processed, target_sr = preprocess_audio(waveform, sr, target_sample_rate=16000)
    assert target_sr == 16000
    # Roughly duration * target_sr samples, allowing for resampling rounding.
    expected_len = int(3.0 * 16000)
    assert abs(len(processed) - expected_len) < 100


def test_preprocess_normalizes_without_clipping(mono_wav_bytes):
    waveform, sr = decode_audio(mono_wav_bytes)
    processed, _ = preprocess_audio(waveform, sr)
    assert np.max(np.abs(processed)) <= 1.0 + 1e-6


def test_split_into_windows_default_size(mono_wav_bytes):
    waveform, sr = decode_audio(mono_wav_bytes)
    processed, target_sr = preprocess_audio(waveform, sr)
    windows = split_into_windows(processed, target_sr, window_seconds=2.0)

    # 3 seconds of audio at 2s windows -> 1 full window + 1 short window,
    # and the short trailing window (1s) meets min_window_seconds (1.0s)
    # so it should NOT be merged.
    assert len(windows) == 2
    assert all(len(w) > 0 for w in windows)


def test_split_into_windows_merges_too_short_trailing_window(mono_wav_bytes):
    # 3.0s of audio split into 1.6s windows -> windows of 1.6s + 1.4s remainder.
    # With min_window_seconds default (1.0s) the remainder is kept separate;
    # force a case where the remainder is below min by using a larger window.
    waveform, sr = decode_audio(mono_wav_bytes)
    processed, target_sr = preprocess_audio(waveform, sr)
    windows = split_into_windows(processed, target_sr, window_seconds=2.9)

    total_samples = sum(len(w) for w in windows)
    assert total_samples == len(processed)  # no audio silently dropped
    # trailing 0.1s remainder should have been merged into the previous window
    assert len(windows) == 1


def test_split_into_windows_rejects_empty_audio():
    with pytest.raises(AudioIngestionError):
        split_into_windows(np.array([], dtype=np.float32), 16000)
