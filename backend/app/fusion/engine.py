"""
app/fusion/engine.py

FR-6: Risk Fusion Engine.

Pure math, no I/O, no FastAPI imports, no knowledge of sessions or the
API layer — this is intentionally independent from api/ and models/ so
the formula can be unit-tested in isolation and reused unchanged
elsewhere.

Formula (exact, per SRS):

    IRS = 100 * (
        w_spoof   * spoof_score
      + w_speaker * (1 - speaker_similarity)
      + w_prosody * prosody_anomaly_score
      + w_context * context_risk_score
    )

Default weights (from config.yaml, validated to sum to 1.0):
    spoof=0.40, speaker=0.30, prosody=0.15, context=0.15

Note the demo's `fusion.py` at the repo root uses DIFFERENT weights
(speaker=0.25, prosody=0.20) — that file is an intentionally simplified
stand-in for a client-side Streamlit demo. This module implements the
SRS-specified weights instead and is what the real backend uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from app.config import FusionWeights, get_settings


@dataclass
class ModuleScores:
    """Input scores to the fusion engine, each expected in [0, 1]."""

    spoof_score: float
    speaker_similarity: float
    prosody_anomaly_score: float
    context_risk_score: float

    def clamped(self) -> "ModuleScores":
        return ModuleScores(
            spoof_score=_clamp01(self.spoof_score),
            speaker_similarity=_clamp01(self.speaker_similarity),
            prosody_anomaly_score=_clamp01(self.prosody_anomaly_score),
            context_risk_score=_clamp01(self.context_risk_score),
        )


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _clamp_irs(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def compute_irs(scores: ModuleScores, weights: Optional[FusionWeights] = None) -> float:
    """
    Compute the raw Impersonation Risk Score (IRS) for one set of module
    scores.

    Speaker similarity is deliberately converted to a *risk* value via
    (1 - speaker_similarity) before weighting — never treat similarity
    itself as risk.

    Args:
        scores: ModuleScores with the four module outputs.
        weights: overrides config.yaml's fusion.weights. Must already
            sum to 1.0 — validated at config load time, not re-checked
            here to keep this function cheap and side-effect-free.

    Returns:
        IRS clamped to [0, 100].
    """
    w = weights or get_settings().fusion.weights
    s = scores.clamped()

    speaker_risk = 1.0 - s.speaker_similarity

    irs = 100.0 * (
        w.spoof * s.spoof_score
        + w.speaker * speaker_risk
        + w.prosody * s.prosody_anomaly_score
        + w.context * s.context_risk_score
    )
    return _clamp_irs(irs)


def update_ema(previous_ema: Optional[float], new_irs: float, alpha: Optional[float] = None) -> float:
    """
    Exponential moving average update for a session's smoothed IRS.

        EMA_1 = IRS_1
        EMA_t = alpha * IRS_t + (1 - alpha) * EMA_(t-1)   for t > 1

    Args:
        previous_ema: the session's current smoothed IRS, or None if
            this is the session's first analysis.
        new_irs: the freshly computed raw IRS for this analysis.
        alpha: overrides config.yaml's fusion.ema_alpha.

    Returns:
        Updated smoothed IRS, clamped to [0, 100].
    """
    a = alpha if alpha is not None else get_settings().fusion.ema_alpha
    if previous_ema is None:
        return _clamp_irs(new_irs)
    ema = a * new_irs + (1 - a) * previous_ema
    return _clamp_irs(ema)


def contributing_factors(scores: ModuleScores) -> Dict[str, float]:
    """
    Present module scores in the "contributing_factors" shape used by
    GET /score/{session_id} — speaker similarity is reported as
    speaker_mismatch (i.e. speaker risk) to stay consistent with how
    it's weighted in the fusion formula.
    """
    s = scores.clamped()
    return {
        "spoof_score": s.spoof_score,
        "speaker_mismatch": _clamp01(1.0 - s.speaker_similarity),
        "prosody_anomaly_score": s.prosody_anomaly_score,
        "context_risk_score": s.context_risk_score,
    }
