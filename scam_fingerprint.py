"""
scam_fingerprint.py — NEW MVP MODULE (Feature 5: Scam Fingerprinting)

STATUS: Newly implemented MVP for the SIH prototype. NOT production-ready.

Goal
----
Build a small, structured "behavioral fingerprint" for a suspicious
conversation (claimed role, problem type, urgency, credential/financial
requests, trajectory stage reached), store it locally, and let a later
call be compared against previously stored fingerprints so the dashboard
can say something like:

    "Similar scam pattern detected from previous interaction."

IMPORTANT — scope of this MVP
------------------------------
This is a LOCAL, single-machine JSON file used purely to demonstrate the
concept for the SIH prototype. It is explicitly NOT:
  - a large-scale or shared scam database,
  - a cryptographically secure fingerprint,
  - deduplicated/cleaned over time (a real system would need retention
    and privacy policies for this kind of data).

FUTURE / PRODUCTION VERSION (not implemented here):
- Central, shared, access-controlled datastore instead of a local file.
- Fuzzy/embedding-based similarity instead of exact categorical matching.
- Privacy-preserving fingerprinting (e.g. no raw transcript stored).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

DEFAULT_STORE_PATH = "scam_fingerprints.json"

# The categorical fields used for similarity comparison. Keeping this as
# a plain list (rather than hard-coding field names throughout) makes it
# easy to add/remove fields later.
_SIMILARITY_FIELDS = [
    "claimed_role",
    "problem_detected",
    "urgency_detected",
    "credential_requested",
    "financial_action_requested",
    "trajectory_stage",
]


@dataclass
class ScamFingerprint:
    claimed_role: str
    problem_detected: bool
    urgency_detected: bool
    credential_requested: bool
    financial_action_requested: bool
    trajectory_stage: str
    created_at: float
    fingerprint_id: Optional[str] = None


def build_fingerprint(
    claimed_role: Optional[str],
    problem_detected: bool,
    urgency_detected: bool,
    credential_requested: bool,
    financial_action_requested: bool,
    trajectory_stage: Optional[str],
) -> ScamFingerprint:
    """
    Build a structured fingerprint from the signals already computed
    elsewhere in the pipeline this call (trust_contradiction's claimed
    role, conversation_trajectory's detected stage/flags, etc.) — this
    function does no detection of its own, it just packages results that
    other modules already produced.
    """
    return ScamFingerprint(
        claimed_role=claimed_role or "unknown",
        problem_detected=bool(problem_detected),
        urgency_detected=bool(urgency_detected),
        credential_requested=bool(credential_requested),
        financial_action_requested=bool(financial_action_requested),
        trajectory_stage=trajectory_stage or "none",
        created_at=time.time(),
    )


def fingerprint_similarity(fp_a: dict, fp_b: dict) -> float:
    """
    Simple categorical-match similarity in [0, 1]: fraction of the
    tracked fields that are identical between the two fingerprints.
    Deliberately simple/explainable for the MVP — see module docstring
    for the production-version direction (embeddings/fuzzy matching).
    """
    if not _SIMILARITY_FIELDS:
        return 0.0
    matches = sum(1 for f in _SIMILARITY_FIELDS if fp_a.get(f) == fp_b.get(f))
    return matches / len(_SIMILARITY_FIELDS)


def load_fingerprints(path: str = DEFAULT_STORE_PATH) -> List[dict]:
    """Load all stored fingerprints (empty list if the file doesn't exist yet)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Don't crash the demo over a corrupted/empty local file.
        return []


def save_fingerprint(fingerprint: ScamFingerprint, path: str = DEFAULT_STORE_PATH) -> str:
    """
    Append a fingerprint to the local JSON store. Returns the assigned
    fingerprint_id.

    NOTE: intentionally explicit/opt-in (call this only when the user
    chooses to save) rather than automatic on every pipeline run, so a
    Streamlit rerun doesn't silently spam duplicate entries.
    """
    data = load_fingerprints(path)
    fp_dict = asdict(fingerprint)
    fp_dict["fingerprint_id"] = hashlib.md5(
        f"{fingerprint.created_at}-{len(data)}".encode()
    ).hexdigest()[:10]
    data.append(fp_dict)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return fp_dict["fingerprint_id"]


def find_similar(
    fingerprint: ScamFingerprint,
    path: str = DEFAULT_STORE_PATH,
    threshold: float = 0.7,
) -> List[Tuple[float, dict]]:
    """
    Compare `fingerprint` against everything already stored, returning
    (similarity_score, stored_fingerprint) pairs at or above `threshold`,
    sorted highest-similarity first.
    """
    fp_dict = asdict(fingerprint)
    stored = load_fingerprints(path)
    scored = [(fingerprint_similarity(fp_dict, s), s) for s in stored]
    scored = [pair for pair in scored if pair[0] >= threshold]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def describe_matches(matches: List[Tuple[float, dict]]) -> str:
    """
    Human-readable summary for the dashboard, per the spec's example
    wording. Kept honest about this being a local prototype demo.
    """
    if not matches:
        return "No similar scam pattern found in this local prototype's stored history."
    best_score, best_fp = matches[0]
    return (
        f"Similar scam pattern detected from a previous interaction "
        f"(similarity {best_score:.0%}, role: {best_fp.get('claimed_role', 'unknown')}). "
        f"Note: this is a local prototype demonstration, not a real "
        f"large-scale scam database."
    )
