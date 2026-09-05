"""
app/fusion/actions.py

Risk band + recommended action engine — deliberately independent from
engine.py's fusion math (engine.py has no idea risk bands exist; this
module has no idea how IRS was computed). This keeps "what does a
score mean" separable from "how is the score calculated", per the
brief's requirement #9.

Bands (exact, per SRS; thresholds are configurable in config.yaml but
default to):
    0-30    -> LOW    -> ALLOW
    31-65   -> MEDIUM -> CALLBACK_VERIFICATION
    66-100  -> HIGH   -> REQUIRE_MFA_OR_ESCALATE
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


class InvalidIRSError(ValueError):
    pass


@dataclass
class RiskDecision:
    risk_band: str
    recommended_action: str


def classify(irs: float) -> RiskDecision:
    """
    Map an IRS value to a risk band + recommended action, using
    thresholds and action text loaded from config.yaml.

    Args:
        irs: Impersonation Risk Score, expected in [0, 100].

    Raises:
        InvalidIRSError: if irs is outside [0, 100] or not a real number.
    """
    if irs is None or not isinstance(irs, (int, float)) or irs != irs:  # NaN check
        raise InvalidIRSError(f"IRS must be a real number, got {irs!r}")
    if not (0.0 <= irs <= 100.0):
        raise InvalidIRSError(f"IRS must be within [0, 100], got {irs}")

    settings = get_settings()
    bands = settings.risk_bands
    actions = settings.actions

    if irs <= bands.low_max:
        band = "LOW"
    elif irs <= bands.medium_max:
        band = "MEDIUM"
    else:
        band = "HIGH"

    return RiskDecision(risk_band=band, recommended_action=actions[band])
