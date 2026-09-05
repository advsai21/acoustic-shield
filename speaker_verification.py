"""
speaker_verification.py
Role 2 deliverable — Speaker Consistency Check.

Enrolls a reference voiceprint from a short sample, then compares live
audio chunks against it via cosine similarity. Flags "speaker drift" if
similarity drops sharply mid-call (possible voice swap/splice/clone).

Uses Resemblyzer (lightweight pretrained speaker-embedding model) —
good fit for a hackathon: no training required, CPU-friendly, single
pip install.
"""

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

# Loaded once and reused — the model itself is small and fast on CPU.
_encoder = VoiceEncoder()


def enroll(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Turn a reference voice sample into a fixed-length voiceprint (embedding).

    Args:
        audio: 1D float32 waveform in range [-1, 1] (or int16 — will be cast)
        sr: sample rate of `audio`

    Returns:
        256-dim numpy embedding vector representing the speaker's voice.
    """
    wav = _to_float32(audio)
    processed = preprocess_wav(wav, source_sr=sr)
    embedding = _encoder.embed_utterance(processed)
    return embedding


def verify(audio_chunk: np.ndarray, sr: int, enrolled_embedding: np.ndarray) -> float:
    """
    Compare a live audio chunk against the enrolled voiceprint.

    Args:
        audio_chunk: 1D float32 or int16 waveform of the live chunk
        sr: sample rate of `audio_chunk`
        enrolled_embedding: output of enroll()

    Returns:
        similarity_score in [0, 1] — 1.0 means near-identical voice,
        lower values suggest a different speaker or synthetic substitution.
        Returns None-safe fallback of 0.5 (neutral) if the chunk is too
        short/silent to embed reliably.

    IMPORTANT — minimum audio length: speaker embeddings need real content
    to work with. In testing, ~300ms chunks are too short and mostly hit
    the neutral fallback below; ~1.5-2s of audio gives much more reliable
    similarity scores (tested: 1.5s -> ~0.82 similarity for a same-speaker
    sample vs. ~0.56 for a different speaker, using synthetic test audio).
    Callers (e.g. the fusion pipeline) should maintain a rolling buffer of
    the last ~1.5-2s of audio and pass THAT to verify(), not the raw
    per-frame chunk used for other modules like spoof detection.
    """
    wav = _to_float32(audio_chunk)

    # Resemblyzer's VAD-based preprocessing can return an empty array on
    # very short or silent chunks — guard against that rather than crashing
    # the pipeline mid-call.
    try:
        processed = preprocess_wav(wav, source_sr=sr)
        if processed is None or len(processed) < sr * 0.3:  # < ~300ms of voiced audio
            return 0.5
        chunk_embedding = _encoder.embed_utterance(processed)
    except Exception:
        return 0.5

    similarity = _cosine_similarity(chunk_embedding, enrolled_embedding)
    # Cosine similarity from Resemblyzer embeddings is typically in [0,1]
    # already (embeddings are L2-normalized, largely non-negative in
    # practice for speech), but clip defensively.
    return float(np.clip(similarity, 0.0, 1.0))


def detect_drift(similarity_history: list, window: int = 5, drop_threshold: float = 0.25) -> bool:
    """
    Flag a sudden mid-call drop in speaker similarity — e.g. a voice swap
    partway through a call, which a single-chunk score might miss if it's
    noisy but a *trend* would catch.

    Args:
        similarity_history: list of similarity scores in call order
        window: number of recent scores to compare against the running baseline
        drop_threshold: minimum drop (baseline - recent) to flag drift

    Returns:
        True if drift detected, else False.
    """
    if len(similarity_history) < window * 2:
        return False

    baseline = np.mean(similarity_history[:-window])
    recent = np.mean(similarity_history[-window:])
    return bool((baseline - recent) >= drop_threshold)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Normalize int16 PCM to float32 [-1, 1] if needed; pass through float audio."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)
