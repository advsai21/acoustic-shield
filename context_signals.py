"""
context_signals.py — PLACEHOLDER for Role 4's contextual/NLP module.

Real version (per SRS FR-5) would run ASR (e.g. Whisper) on the live call
audio and scan the transcript for risky phrases, plus apply call-metadata
rules (unknown number, odd call time, first contact).

For this all-in-one demo, real-time ASR is skipped (heavy dependency,
not the point of this demo) — instead this takes a manually-entered
transcript string (simulating "what the caller said") and simple metadata
flags, so the fusion engine and dashboard can be demonstrated fully.
Role 4 should replace get_context_risk_score() internals with real ASR +
NLP, keeping the same function signature.
"""

RISK_PHRASES = [
    "otp", "one time password", "verification code", "share your pin",
    "urgent transfer", "wire the money", "don't tell anyone",
    "keep this confidential", "act now", "your account will be suspended",
    "send money immediately", "gift card",
]


def get_context_risk_score(transcript_text: str, call_metadata: dict) -> float:
    """
    Placeholder contextual risk scorer.

    Args:
        transcript_text: text of what's been said so far in the call
            (in the real system, produced by ASR; here, typed by the demo
            operator to simulate it)
        call_metadata: dict with optional keys:
            - "unknown_number": bool
            - "odd_call_time": bool
            - "first_contact": bool

    Returns:
        context_risk_score in [0, 1]
    """
    score = 0.0
    text_lower = (transcript_text or "").lower()

    matched_phrases = [p for p in RISK_PHRASES if p in text_lower]
    if matched_phrases:
        # More matches = more risk, capped so one keyword doesn't maximize score alone
        score += min(0.15 * len(matched_phrases), 0.6)

    metadata = call_metadata or {}
    if metadata.get("unknown_number"):
        score += 0.15
    if metadata.get("odd_call_time"):
        score += 0.1
    if metadata.get("first_contact"):
        score += 0.1

    return float(min(score, 1.0)), matched_phrases


def get_context_risk_score_only(transcript_text: str, call_metadata: dict) -> float:
    """Convenience wrapper matching the exact FR-5 contract (score only)."""
    score, _ = get_context_risk_score(transcript_text, call_metadata)
    return score
