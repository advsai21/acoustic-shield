"""
spoof_heuristic.py — PLACEHOLDER for Role 1's spoof/anti-spoofing model.

This is the heuristic from the original demo script, kept isolated in its
own file so it's a clean drop-in point: Role 1 should replace
get_spoof_score() with their real pretrained/fine-tuned classifier
(e.g. RawNet2/LCNN on ASVspoof) without anyone else's code needing to change.

Contract Role 1 must preserve:
    def get_spoof_score(audio_chunk: np.ndarray, sr: int) -> float
    returns a value in [0, 1], higher = more likely synthetic/spoofed.
"""

import numpy as np
import librosa


def get_spoof_score(audio_chunk: np.ndarray, sr: int) -> float:
    """Heuristic placeholder — replace with Role 1's trained classifier."""
    wav = audio_chunk.astype(np.float32) / 32768.0 if audio_chunk.dtype == np.int16 else audio_chunk.astype(np.float32)

    if len(wav) < sr * 0.2:
        return 0.3  # neutral for near-silent/too-short chunks

    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    mfcc_var = np.var(mfccs)

    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr))

    pitches, _ = librosa.piptrack(y=wav, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    score = 0.0
    if pitch_std < 5.0:
        score += 0.35
    elif pitch_std < 15.0:
        score += 0.15

    if spec_centroid < 1000 or spec_centroid > 4000:
        score += 0.30

    if mfcc_var < 50.0:
        score += 0.35

    return float(np.clip(score, 0.05, 0.99))
