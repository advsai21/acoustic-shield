"""
fusion.py — Risk Fusion Engine + Alert/Action Engine.

This mirrors what Role 3's real backend owns (SRS FR-6, FR-7). It's
included here so the full pipeline can be demoed as one self-contained
app; once Role 3's actual FastAPI service exists, this file can be
deleted from the demo and replaced by calls to their API.
"""

import numpy as np

DEFAULT_WEIGHTS = {
    "spoof": 0.40,
    "speaker": 0.25,    # applied as (1 - similarity)
    "prosody": 0.20,
    "context": 0.15,
}


def fuse_scores(spoof_score: float, speaker_similarity: float,
                 prosody_score: float, context_score: float,
                 weights: dict = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    speaker_risk = 1.0 - speaker_similarity
    irs = 100 * (
        w["spoof"] * spoof_score
        + w["speaker"] * speaker_risk
        + w["prosody"] * prosody_score
        + w["context"] * context_score
    )
    return float(np.clip(irs, 0, 100))


def recommended_action(irs: float) -> tuple:
    """Returns (risk_band, recommended_action_text)."""
    if irs >= 66:
        return "HIGH", "Require callback verification or MFA / escalate to supervisor"
    elif irs >= 31:
        return "MEDIUM", "Suggest soft callback verification, keep monitoring"
    else:
        return "LOW", "Allow — voice parameters within expected baseline"


def smooth(history: list, alpha: float = 0.4) -> float:
    """Exponential moving average of a score history, for a less jittery live score."""
    if not history:
        return 0.0
    ema = history[0]
    for val in history[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return float(ema)
