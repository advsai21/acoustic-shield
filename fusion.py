"""
fusion.py — Risk Fusion Engine + Alert/Action Engine.

This mirrors what Role 3's real backend owns (SRS FR-6, FR-7). It's
included here so the full pipeline can be demoed as one self-contained
app; once Role 3's actual FastAPI service exists, this file can be
deleted from the demo and replaced by calls to their API.

============================================================================
UPDATE (SIH extension — Feature 3: Confidence-Adaptive Fusion):
The functions above this note (fuse_scores, recommended_action, smooth)
are the ORIGINAL, ALREADY-IMPLEMENTED fusion logic and are left completely
unchanged for backward compatibility — anything already calling
fuse_scores() keeps working exactly as before.

Below that is a NEW, ADDITIONAL function, fuse_scores_adaptive(), which
is the newly implemented MVP version of confidence-adaptive fusion. It
lets each signal carry its own confidence value (e.g. "low audio quality
-> low confidence in speaker verification for this chunk"), so unreliable
signals get down-weighted instead of always being trusted at their full
base weight.

    effective_weight = base_weight * confidence
    (then effective weights are normalized to sum to 1 before fusing)

STATUS: fuse_scores_adaptive() is a newly implemented MVP for the
prototype — NOT production-ready. It is a straightforward weighted-sum
extension of the existing math, not a learned/ML fusion model.
============================================================================
"""

import numpy as np

DEFAULT_WEIGHTS = {
    "spoof": 0.40,
    "speaker": 0.25,  # applied as (1 - similarity)
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


# ============================================================================
# NEW MVP CODE BELOW — Feature 3: Confidence-Adaptive Fusion
# ============================================================================

# Base weights for the adaptive fusion path. Includes the two brand-new
# conversation-intelligence signals (trajectory, trust) alongside the
# original four. All six are optional at call time — see
# fuse_scores_adaptive() docstring — so this still works if e.g. no
# trust-contradiction signal is available yet.
DEFAULT_ADAPTIVE_WEIGHTS = {
    "spoof": 0.30,
    "speaker": 0.20,
    "prosody": 0.15,
    "context": 0.15,
    "trajectory": 0.10,
    "trust": 0.10,
}


def fuse_scores_adaptive(signal_risks: dict, signal_confidence: dict,
                          weights: dict = None) -> tuple:
    """
    Confidence-adaptive fusion (Feature 3 — MVP).

    Unlike fuse_scores() above, every signal here is already expressed as
    a RISK value in [0, 1] (higher = more suspicious) — the caller is
    responsible for any inversion first (e.g. pass speaker_risk =
    1 - speaker_similarity, exactly like the original fuse_scores() does
    internally). Keeping that conversion outside this function is what
    makes it generic enough to accept an arbitrary subset of signals
    (e.g. only 4 of the 6 known ones, or new ones added later) without
    changing this function.

    Args:
        signal_risks: dict like {"spoof": 0.8, "speaker": 0.3, ...} —
            each value in [0, 1], higher = riskier. Only include the
            signals that are actually available this round.
        signal_confidence: dict with the SAME keys as signal_risks,
            each value in [0, 1] — how much to trust that signal right
            now (e.g. low audio quality -> low confidence for "speaker").
            A missing key defaults to confidence 1.0 (fully trusted).
        weights: base weights per signal name; defaults to
            DEFAULT_ADAPTIVE_WEIGHTS. Only keys present in signal_risks
            are used.

    Returns:
        (irs, breakdown) where irs is the fused Impersonation Risk Score
        in [0, 100], and breakdown is a dict showing the effective and
        normalized weight that was actually applied to each signal —
        useful for the "Signal Confidence" dashboard panel and for
        explaining *why* the score came out the way it did.

    Example:
        # Audio quality is poor, so speaker verification confidence is low
        # -> speaker verification ends up contributing less to the final
        # score, per the FEATURE 3 spec.
        risks = {"spoof": 0.7, "speaker": 0.6, "prosody": 0.4}
        conf  = {"spoof": 1.0, "speaker": 0.3, "prosody": 0.9}
        irs, breakdown = fuse_scores_adaptive(risks, conf)
    """
    base_weights = weights or DEFAULT_ADAPTIVE_WEIGHTS

    effective_weights = {}
    for name in signal_risks:
        base_w = base_weights.get(name, 0.0)
        confidence = float(signal_confidence.get(name, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        effective_weights[name] = base_w * confidence

    total_weight = sum(effective_weights.values())

    if total_weight <= 0:
        # All available signals had zero confidence (or zero base weight)
        # -> nothing reliable to fuse. Fall back to a neutral 0 risk
        # rather than dividing by zero; the dashboard should make clear
        # this means "insufficient reliable signal", not "confirmed safe".
        return 0.0, {
            "effective_weights": effective_weights,
            "normalized_weights": {k: 0.0 for k in effective_weights},
            "note": "All signal confidences were 0 — insufficient reliable "
                    "signal to compute a risk score this round.",
        }

    normalized_weights = {
        name: w / total_weight for name, w in effective_weights.items()
    }

    irs = 100 * sum(
        normalized_weights[name] * float(np.clip(signal_risks[name], 0.0, 1.0))
        for name in signal_risks
    )

    breakdown = {
        "effective_weights": effective_weights,
        "normalized_weights": normalized_weights,
    }
    return float(np.clip(irs, 0, 100)), breakdown
