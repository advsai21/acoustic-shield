"""
trust_contradiction.py — NEW MVP MODULE (Feature 2: Trust Contradiction Engine)

STATUS: Newly implemented MVP for the SIH prototype. NOT production-ready.

Goal
----
Flag the mismatch between who a caller CLAIMS to be and WHAT they are
asking for. E.g. a "bank employee" has no legitimate reason to ask for
an OTP; a "delivery agent" has no reason to ask for a banking password.

Design choice (per project instructions)
-----------------------------------------
Rule-based/keyword matching, kept as plain data tables (ROLE_PATTERNS,
REQUEST_PATTERNS, CONTRADICTION_RULES) rather than an ML model, so this
is an easy drop-in point for a future NLP/ML classifier — anyone
replacing this later only needs to keep the detect_contradiction()
function signature the same.

FUTURE / PRODUCTION VERSION (not implemented here):
- Replace keyword tables with a trained intent/role classifier.
- Use conversation-turn structure instead of scanning one text blob, so
  role and request are correctly attributed even in longer calls.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# Claimed-role patterns. Kept simple and explainable for the demo.
ROLE_PATTERNS: Dict[str, List[str]] = {
    "bank_employee": [
        "bank employee", "calling from the bank", "bank representative",
        "banking official", "from your bank", "bank staff",
    ],
    "delivery_agent": [
        "delivery agent", "courier", "delivery boy", "your parcel",
        "your package", "from the courier company", "logistics partner",
    ],
    "government_official": [
        "government official", "income tax department", "police department",
        "customs office", "cyber cell", "government office",
        "calling from the government", "rbi official", "reserve bank",
    ],
    "tech_support": [
        "technical support", "tech support", "microsoft support",
        "computer support", "it department", "your computer has a virus",
        "software support team",
    ],
}

# Requested-action patterns — things a caller might ask the victim for.
REQUEST_PATTERNS: Dict[str, List[str]] = {
    "otp": ["otp", "one time password", "verification code"],
    "banking_password": [
        "banking password", "net banking password", "login password",
        "internet banking password",
    ],
    "upi_pin": ["upi pin", "upi password"],
    "card_cvv": ["cvv", "card verification value", "3 digit code on the back",
                 "3-digit code on the back"],
    "card_number": ["card number", "debit card number", "credit card number"],
}

# (claimed_role, requested_action) -> human-readable reason it's a red flag.
# This table is the "modular rule set" the task description asks for —
# swap or extend it freely without touching detect_contradiction().
CONTRADICTION_RULES: Dict[tuple, str] = {
    ("bank_employee", "otp"): "Banks never ask customers for an OTP over a phone call.",
    ("bank_employee", "card_cvv"): "Bank staff should never ask for a card's CVV.",
    ("bank_employee", "upi_pin"): "Bank staff should never ask for a UPI PIN.",
    ("bank_employee", "banking_password"): "Bank staff should never ask for your net-banking password.",
    ("delivery_agent", "banking_password"): "A delivery agent has no legitimate reason to ask for a banking password.",
    ("delivery_agent", "upi_pin"): "A delivery agent has no legitimate reason to ask for a UPI PIN.",
    ("delivery_agent", "otp"): "A delivery agent asking for an OTP is a classic parcel-delivery scam pattern.",
    ("government_official", "upi_pin"): "Government departments do not collect UPI PINs over the phone.",
    ("government_official", "otp"): "Government departments do not ask for OTPs over the phone.",
    ("government_official", "card_cvv"): "Government departments do not ask for card CVVs.",
    ("tech_support", "card_cvv"): "Technical support has no legitimate reason to ask for a card CVV.",
    ("tech_support", "otp"): "Technical support has no legitimate reason to ask for an OTP.",
    ("tech_support", "banking_password"): "Technical support has no legitimate reason to ask for a banking password.",
}


@dataclass
class ContradictionResult:
    contradiction_score: float  # 0.0 - 1.0
    contradiction_detected: bool
    claimed_role: Optional[str]
    requested_action: Optional[str]
    reason: str


def _find_first_match(text_lower: str, patterns: Dict[str, List[str]]) -> Optional[str]:
    for key, phrases in patterns.items():
        if any(p in text_lower for p in phrases):
            return key
    return None


def _find_all_matches(text_lower: str, patterns: Dict[str, List[str]]) -> List[str]:
    return [key for key, phrases in patterns.items() if any(p in text_lower for p in phrases)]


def detect_contradiction(transcript_text: str) -> ContradictionResult:
    """
    Scan transcript text for a claimed role and a requested action, and
    check whether that combination is a known contradiction.

    Returns a ContradictionResult with:
      - contradiction_score in [0, 1]
      - contradiction_detected: bool
      - claimed_role / requested_action: best-guess matches (or None)
      - reason: human-readable explanation
    """
    text_lower = (transcript_text or "").lower()

    roles_found = _find_all_matches(text_lower, ROLE_PATTERNS)
    requests_found = _find_all_matches(text_lower, REQUEST_PATTERNS)

    if not roles_found and not requests_found:
        return ContradictionResult(
            contradiction_score=0.0,
            contradiction_detected=False,
            claimed_role=None,
            requested_action=None,
            reason="No claimed role or sensitive request detected.",
        )

    # Check every (role, request) pair found for a known contradiction.
    best_role, best_request, best_reason = None, None, None
    for role in roles_found or [None]:
        for request in requests_found or [None]:
            if role is not None and request is not None:
                rule = CONTRADICTION_RULES.get((role, request))
                if rule is not None:
                    best_role, best_request, best_reason = role, request, rule
                    break
        if best_reason:
            break

    if best_reason:
        return ContradictionResult(
            contradiction_score=0.9,
            contradiction_detected=True,
            claimed_role=best_role,
            requested_action=best_request,
            reason=best_reason,
        )

    # Role and/or request detected, but not a known-bad combination.
    # Still worth a small non-zero score if a sensitive request was made
    # at all, since that alone is worth flagging even without a role clash.
    partial_score = 0.3 if requests_found else 0.1
    role_txt = roles_found[0] if roles_found else "no specific role claimed"
    request_txt = requests_found[0] if requests_found else "no sensitive request detected"
    return ContradictionResult(
        contradiction_score=partial_score,
        contradiction_detected=False,
        claimed_role=roles_found[0] if roles_found else None,
        requested_action=requests_found[0] if requests_found else None,
        reason=f"No known contradiction rule for ({role_txt}, {request_txt}), "
               f"but the request itself may still be worth monitoring.",
    )


def evaluate(role: Optional[str], request: Optional[str]) -> tuple:
    """
    Pure lookup helper: given an already-identified role and request,
    return (is_contradiction, reason). Exposed separately so a future
    ML/NLP role+intent classifier can plug straight into the same rule
    table without going through text matching again.
    """
    if role is None or request is None:
        return False, "Insufficient information to evaluate a contradiction."
    reason = CONTRADICTION_RULES.get((role, request))
    if reason:
        return True, reason
    return False, "No known contradiction rule for this role/request combination."
