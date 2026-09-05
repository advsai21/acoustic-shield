"""
app/ml/adapters.py

The integration boundary between the backend and the ML teammates'
modules (speaker_verification.py, prosody.py, spoof_heuristic.py,
context_signals.py, rolling_buffer.py — all living at the repo root,
NOT inside backend/).

CONTRACT (do not change these signatures/meanings):

    get_spoof_score(audio_chunk, sr)                       -> float in [0, 1]
        0 = genuine, 1 = highly likely synthetic/spoofed

    get_speaker_similarity(audio_chunk, sr, enrolled_embedding) -> float in [0, 1]
        0 = mismatch, 1 = strong speaker match

    get_prosody_anomaly_score(audio_chunk, sr)             -> float in [0, 1]
        0 = normal, 1 = highly anomalous

    get_context_risk_score(transcript_text, call_metadata) -> float in [0, 1]
        0 = safe, 1 = highly suspicious

    create_enrollment_embedding(audio, sr)                 -> np.ndarray

When the real Role 1 (spoof) and Role 4 (context) modules land, only
this file (plus repo-root spoof_heuristic.py / context_signals.py, per
their own drop-in contracts) needs to change — nothing in app/fusion,
app/api, or app/ingestion should need to know these are stubs.

Integration notes discovered during repo inspection:
  - speaker_verification.verify() already clips its own output to
    [0, 1] and internally returns a neutral 0.5 on too-short/silent
    audio, so no extra normalization is needed here.
  - context_signals.get_context_risk_score() returns a
    (score, matched_phrases) TUPLE, not a bare float — we use the
    repo's own get_context_risk_score_only() wrapper to get the [0, 1]
    float this adapter's contract promises.
  - RollingAudioBuffer (rolling_buffer.py) is re-exported here so the
    ingestion/fusion pipeline can maintain a per-session rolling
    buffer for speaker verification without importing repo-root
    modules directly anywhere else in the backend.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make the repo-root teammate modules importable.
#
# Repo layout:
#   acoustic-shield/
#     speaker_verification.py   <- Role 2 (real)
#     prosody.py                <- Role 2 (real)
#     spoof_heuristic.py        <- Role 1 (placeholder)
#     context_signals.py        <- Role 4 (placeholder)
#     rolling_buffer.py         <- shared utility
#     backend/
#       app/
#         ml/
#           adapters.py         <- this file
#
# backend/app/ml/adapters.py -> ../../.. = repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import speaker_verification as _speaker_verification
    import prosody as _prosody
    import spoof_heuristic as _spoof_heuristic
    import context_signals as _context_signals
    from rolling_buffer import RollingAudioBuffer  # noqa: F401  (re-exported)

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit if repo layout changes
    _speaker_verification = None
    _prosody = None
    _spoof_heuristic = None
    _context_signals = None
    RollingAudioBuffer = None  # type: ignore
    _IMPORT_ERROR = exc
    logger.error(
        "Could not import teammate ML modules from repo root %s: %s. "
        "Falling back to neutral stub scores — fix the import path or "
        "run the backend from within the repo.",
        _REPO_ROOT,
        exc,
    )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public adapter functions — this is the ONLY surface the rest of the
# backend (fusion pipeline, API routes) should call into for ML scores.
# ---------------------------------------------------------------------------


def get_spoof_score(audio_chunk: np.ndarray, sr: int) -> float:
    """Role 1 placeholder today; swap spoof_heuristic.py for the real model."""
    if _spoof_heuristic is None:
        logger.warning("spoof_heuristic unavailable (%s); returning neutral stub score.", _IMPORT_ERROR)
        return 0.3
    return _clip01(_spoof_heuristic.get_spoof_score(audio_chunk, sr))


def get_speaker_similarity(audio_chunk: np.ndarray, sr: int, enrolled_embedding: np.ndarray) -> float:
    """
    Real Resemblyzer-based comparison (Role 2).

    IMPORTANT: per speaker_verification.py's own docstring, this needs
    ~1.5-2s of audio for a reliable score. Callers should pass a
    RollingAudioBuffer's accumulated window here, not a single small
    ingestion chunk — see app/fusion pipeline wiring in api/routes.py.
    """
    if _speaker_verification is None:
        logger.warning("speaker_verification unavailable (%s); returning neutral stub score.", _IMPORT_ERROR)
        return 0.5
    return _clip01(_speaker_verification.verify(audio_chunk, sr, enrolled_embedding))


def get_prosody_anomaly_score(audio_chunk: np.ndarray, sr: int) -> float:
    """Real librosa-based prosody heuristic (Role 2)."""
    if _prosody is None:
        logger.warning("prosody module unavailable (%s); returning neutral stub score.", _IMPORT_ERROR)
        return 0.3
    return _clip01(_prosody.get_prosody_anomaly_score(audio_chunk, sr))


def get_context_risk_score(transcript_text: str, call_metadata: dict) -> float:
    """
    Role 4 placeholder today; swap context_signals.py for real ASR + NLP.

    Note: context_signals.get_context_risk_score() returns
    (score, matched_phrases) — we use the module's own
    get_context_risk_score_only() wrapper to honor this adapter's
    float-only contract.
    """
    if _context_signals is None:
        logger.warning("context_signals unavailable (%s); returning neutral stub score.", _IMPORT_ERROR)
        return 0.0
    return _clip01(_context_signals.get_context_risk_score_only(transcript_text, call_metadata))


def create_enrollment_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Real Resemblyzer enrollment (Role 2)."""
    if _speaker_verification is None:
        raise RuntimeError(
            f"speaker_verification module unavailable ({_IMPORT_ERROR}); "
            "cannot create a real enrollment embedding."
        )
    return _speaker_verification.enroll(audio, sr)
