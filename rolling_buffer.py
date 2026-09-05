"""
rolling_buffer.py — small utility to maintain a rolling window of recent
audio, needed because speaker verification (and to a lesser extent
prosody) need more context than a single ~300ms capture chunk provides.

Usage pattern in the live pipeline:
    buf = RollingAudioBuffer(sr=16000, window_seconds=1.8)
    for chunk in incoming_audio_chunks:
        buf.push(chunk)
        if buf.is_ready():
            speaker_sim = sv.verify(buf.get(), sr, enrolled_embedding)
        spoof_score = spoof.get_spoof_score(chunk, sr)   # fine on small chunks
"""

import numpy as np


class RollingAudioBuffer:
    def __init__(self, sr: int, window_seconds: float = 1.8):
        self.sr = sr
        self.window_samples = int(sr * window_seconds)
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray):
        chunk = chunk.astype(np.float32) / 32768.0 if chunk.dtype == np.int16 else chunk.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, chunk])
        if len(self._buffer) > self.window_samples:
            self._buffer = self._buffer[-self.window_samples:]

    def is_ready(self, min_fraction: float = 0.7) -> bool:
        """True once the buffer holds at least min_fraction of the target window."""
        return len(self._buffer) >= self.window_samples * min_fraction

    def get(self) -> np.ndarray:
        return self._buffer.copy()
