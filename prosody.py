"""
prosody.py
Role 2 deliverable — Prosody / Behavioral Feature Extraction.

Extracts pitch variability, speaking-rate proxy, pause structure, and
jitter/shimmer-style micro-variation from a short audio chunk, and
combines them into a single prosody_anomaly_score in [0, 1] where
higher = more likely synthetic/unnatural speech.

Rationale (why these features flag synthetic speech):
- Real human speech has natural pitch wobble; many TTS/vocoder outputs
  are unnaturally flat or too-regular in F0.
- Real speech has irregular pause/breath patterns; synthetic speech often
  has metronomic timing.
- Jitter/shimmer (cycle-to-cycle pitch/amplitude micro-variation) is a
  classic voice-quality measure that's harder for vocoders to reproduce
  faithfully, especially older/lighter-weight TTS systems.

This module intentionally does NOT try to do full spoof classification
(that's Role 1's job) — it's a complementary, interpretable signal fed
into the fusion engine alongside spoof_score and speaker_similarity.
"""

import numpy as np
import librosa


def extract_prosody_features(audio_chunk: np.ndarray, sr: int) -> dict:
    """
    Extract raw prosody measurements from an audio chunk.

    Args:
        audio_chunk: 1D waveform (int16 or float32)
        sr: sample rate

    Returns:
        dict of raw feature values (useful for the UI's "why this score"
        breakdown, not just the final anomaly score).
    """
    wav = _to_float32(audio_chunk)

    if len(wav) < sr * 0.2:  # too short to say anything meaningful
        return _empty_features()

    # --- Pitch (F0) tracking ---
    f0, voiced_flag, _ = librosa.pyin(
        wav,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz, covers most speech
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 3 else 0.0
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # --- Jitter: cycle-to-cycle pitch period variation ---
    jitter = _estimate_jitter(voiced_f0)

    # --- Shimmer: cycle-to-cycle amplitude variation (frame RMS proxy) ---
    shimmer = _estimate_shimmer(wav, sr)

    # --- Speaking rate proxy: onset rate (syllable-ish pulses per second) ---
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_sec = len(wav) / sr
    speaking_rate = float(len(onsets) / duration_sec) if duration_sec > 0 else 0.0

    # --- Pause structure: fraction of frame energy below a silence threshold ---
    pause_ratio = _estimate_pause_ratio(wav, sr)

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "speaking_rate": speaking_rate,
        "pause_ratio": pause_ratio,
    }


def get_prosody_anomaly_score(audio_chunk: np.ndarray, sr: int) -> float:
    """
    Combine raw prosody features into a single anomaly score (0-1),
    where higher = more likely synthetic/unnatural.

    This is a hand-tuned heuristic for the hackathon prototype — thresholds
    are approximate and should be calibrated against real vs. synthetic
    samples once you have a demo dataset (see calibrate_thresholds.py note
    at the bottom of this file).
    """
    feats = extract_prosody_features(audio_chunk, sr)

    if feats["pitch_mean"] == 0.0:
        # No reliable voiced signal in this chunk (silence/noise) —
        # neutral score rather than a false anomaly flag.
        return 0.3

    score = 0.0

    # 1. Flat pitch — natural speech usually has pitch_std > ~10-15 Hz
    if feats["pitch_std"] < 5.0:
        score += 0.35
    elif feats["pitch_std"] < 12.0:
        score += 0.15

    # 2. Low jitter — real voices have some micro-instability
    if feats["jitter"] < 0.005:
        score += 0.2
    elif feats["jitter"] < 0.01:
        score += 0.1

    # 3. Low shimmer — same idea for amplitude micro-variation
    if feats["shimmer"] < 0.02:
        score += 0.2
    elif feats["shimmer"] < 0.04:
        score += 0.1

    # 4. Unnaturally regular pause structure (very low or suspiciously
    #    uniform pause ratio can indicate machine-generated pacing)
    if feats["pause_ratio"] < 0.02:
        score += 0.15

    return float(np.clip(score, 0.0, 1.0))


# --- Helper functions ---

def _estimate_jitter(voiced_f0: np.ndarray) -> float:
    """Relative average perturbation of consecutive F0 values."""
    if len(voiced_f0) < 3:
        return 0.0
    diffs = np.abs(np.diff(voiced_f0))
    mean_f0 = np.mean(voiced_f0)
    if mean_f0 == 0:
        return 0.0
    return float(np.mean(diffs) / mean_f0)


def _estimate_shimmer(wav: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 256) -> float:
    """Relative average perturbation of consecutive frame RMS amplitudes."""
    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[rms > 1e-6]  # drop near-silent frames
    if len(rms) < 3:
        return 0.0
    diffs = np.abs(np.diff(rms))
    mean_rms = np.mean(rms)
    if mean_rms == 0:
        return 0.0
    return float(np.mean(diffs) / mean_rms)


def _estimate_pause_ratio(wav: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 256,
                           silence_db: float = -35.0) -> float:
    """Fraction of frames below a silence energy threshold (in dBFS)."""
    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) if np.max(rms) > 0 else rms
    silent_frames = np.sum(rms_db < silence_db)
    return float(silent_frames / len(rms_db)) if len(rms_db) > 0 else 0.0


def _empty_features() -> dict:
    return {
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "jitter": 0.0,
        "shimmer": 0.0,
        "speaking_rate": 0.0,
        "pause_ratio": 0.0,
    }


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


# NOTE for hackathon calibration:
# The thresholds in get_prosody_anomaly_score() are reasonable starting
# heuristics, not measured constants. Before the demo, run this module on
# a handful of genuine recordings and a handful of TTS-generated clips of
# the same sentences, print extract_prosody_features() for each, and adjust
# the thresholds so genuine samples score low and synthetic samples score
# high. A tiny script for this is worth 15 minutes and will make your demo
# much more convincing.
