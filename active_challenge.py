"""
active_challenge.py — NEW MVP MODULE (Feature 4: Active Voice Challenge)

STATUS: Newly implemented MVP for the SIH prototype. NOT production-ready.

Goal
----
When the fused risk becomes suspicious, generate an unpredictable
challenge phrase (e.g. "Blue Elephant 47"), have the user/caller read it
back, and run the EXISTING spoof-detection and speaker-verification
modules on that response as one more advisory signal.

IMPORTANT — what this is NOT
-----------------------------
This does NOT prove the caller is human, and it does NOT prove they are
who they claim to be. It only adds one more data point (does the voice
in the response match the enrolled voiceprint? does it look synthetic?)
for a human reviewer or the fusion engine to weigh. A well-resourced
attacker could still record/replay or clone a response to a phrase they
anticipated. Treat this as a friction/signal mechanism, not proof.

What this MVP implements
-------------------------
- generate_challenge_phrase(): unpredictable "<adjective> <noun> <number>"
  phrase using the `secrets` module (cryptographically strong randomness),
  NOT the `random` module.
- evaluate_challenge_response(): reuses the EXISTING spoof_heuristic and
  speaker_verification modules on the uploaded/recorded response audio.
- should_trigger_challenge(): simple helper so the dashboard only shows
  the challenge when risk is MEDIUM/HIGH, per the fusion engine's output.

What this MVP does NOT implement (future/production version)
---------------------------------------------------------------
- Live telephony / real-time call injection of the challenge phrase.
  For this Streamlit prototype, the workflow is: show the phrase on
  screen, let the user upload or record a response file. That is a
  stand-in for a real IVR/telephony integration, not a claim that live
  telephony exists.
- Liveness/anti-replay detection (e.g. checking the response audio
  actually contains the challenge words via ASR) — could reuse the
  existing `asr` module in a future iteration; out of scope for this MVP.
"""

import secrets
from dataclasses import dataclass
from typing import Optional

import numpy as np

import spoof_heuristic
import speaker_verification


# Small, demo-friendly word lists. Kept short and readable so the phrase
# is easy for a human to say out loud and easy for a judge to see on
# screen — this is a UX choice, not a security-critical parameter (the
# unpredictability comes from `secrets`, not from list size).
_ADJECTIVES = [
    "Blue", "Silver", "Golden", "Crimson", "Purple", "Emerald",
    "Amber", "Coral", "Ivory", "Scarlet",
]
_NOUNS = [
    "Elephant", "Tiger", "Falcon", "Dolphin", "Panther", "Sparrow",
    "Turtle", "Eagle", "Otter", "Wolf",
]


def generate_challenge_phrase() -> str:
    """
    Generate an unpredictable challenge phrase like "Blue Elephant 47".

    Uses `secrets` (cryptographically strong RNG) instead of `random`,
    per the project requirement, since this value is used as a
    verification challenge rather than for cosmetic randomness.
    """
    adjective = secrets.choice(_ADJECTIVES)
    noun = secrets.choice(_NOUNS)
    number = secrets.randbelow(90) + 10  # two-digit number, 10-99
    return f"{adjective} {noun} {number}"


@dataclass
class ChallengeEvaluation:
    spoof_score: Optional[float]
    speaker_similarity: Optional[float]
    summary: str
    disclaimer: str = (
        "This is an additional verification signal only — it does not "
        "prove the caller is human or confirm their identity."
    )


def evaluate_challenge_response(
    response_audio: np.ndarray,
    sr: int,
    enrolled_embedding: Optional[np.ndarray] = None,
) -> ChallengeEvaluation:
    """
    Run the EXISTING spoof-detection and speaker-verification modules on
    a challenge-response recording. Reused as-is — no changes made to
    spoof_heuristic.py or speaker_verification.py.

    Args:
        response_audio: 1D waveform (float32 or int16) of the response.
        sr: sample rate of response_audio.
        enrolled_embedding: output of speaker_verification.enroll(), if
            available. If None, speaker similarity is skipped.

    Returns:
        ChallengeEvaluation with the two raw scores plus a short human
        -readable summary and the mandatory disclaimer.
    """
    spoof_score = spoof_heuristic.get_spoof_score(response_audio, sr)

    speaker_similarity = None
    if enrolled_embedding is not None:
        speaker_similarity = speaker_verification.verify(
            response_audio, sr, enrolled_embedding
        )

    parts = [f"spoof score {spoof_score:.2f}"]
    if speaker_similarity is not None:
        parts.append(f"speaker similarity {speaker_similarity:.2f}")
    summary = "Challenge response analyzed: " + ", ".join(parts) + "."

    return ChallengeEvaluation(
        spoof_score=spoof_score,
        speaker_similarity=speaker_similarity,
        summary=summary,
    )


def should_trigger_challenge(risk_band: str) -> bool:
    """
    Convenience helper for the dashboard: only surface the active
    challenge once risk is at least MEDIUM, matching the "when the risk
    becomes suspicious/high" requirement.
    """
    return risk_band in ("MEDIUM", "HIGH")


def challenge_result_as_risk_signal(evaluation: ChallengeEvaluation) -> Optional[float]:
    """
    Optional helper to fold the challenge result into the adaptive fusion
    engine (fusion.fuse_scores_adaptive) as one more risk-oriented signal,
    e.g. signal_risks["challenge"] = challenge_result_as_risk_signal(eval).

    Combines spoof risk and (lack of) speaker similarity into a single
    [0, 1] risk value. Returns None if there isn't enough information
    (e.g. no enrolled voiceprint) to produce a meaningful score, so the
    caller can simply omit the "challenge" key from fusion in that case.
    """
    if evaluation.spoof_score is None:
        return None
    if evaluation.speaker_similarity is None:
        # Only spoof information available.
        return float(np.clip(evaluation.spoof_score, 0.0, 1.0))
    speaker_risk = 1.0 - evaluation.speaker_similarity
    combined = 0.5 * evaluation.spoof_score + 0.5 * speaker_risk
    return float(np.clip(combined, 0.0, 1.0))
