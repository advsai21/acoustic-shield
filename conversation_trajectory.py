"""
conversation_trajectory.py — NEW MVP MODULE (Feature 1: Conversation Trajectory Engine)

STATUS: Newly implemented MVP for the SIH prototype. NOT production-ready.

Goal
----
Instead of scoring isolated keywords, this tracks HOW a conversation
progresses through the classic social-engineering / vishing script:

    1. authority_claim       ("I am calling from your bank")
    2. problem_issue         ("There is a problem with your account")
    3. urgency_pressure      ("You need to act immediately")
    4. credential_request    ("Tell me the OTP")
    5. financial_action      ("Now transfer the money")

A conversation that hits several of these stages *in order* is far more
suspicious than one that just contains a stray risky word, so the score
rewards both (a) how many stages have fired and (b) whether they fired
in the expected escalating order.

Design choice (per project instructions)
-----------------------------------------
This is a deliberately simple, explainable rule-based/substring matcher —
NOT an ML model. It is intentionally kept modular (STAGE_PATTERNS is a
plain dict) so it can later be swapped for an NLP/ML classifier without
changing the public functions below (analyze_transcript / the
ConversationTrajectoryEngine class).

Two ways to use it
-------------------
1. Stateless, single-shot (used by the Streamlit MVP, since the current
   dashboard already treats the transcript as "whatever has been said so
   far in the call", same pattern as context_signals.py):

       result = analyze_transcript(full_transcript_text)

2. Stateful, turn-by-turn (for anyone wiring this into a real multi-turn
   pipeline later, e.g. a live call): keep a ConversationTrajectoryEngine
   instance per call and feed it each new utterance as it arrives.

Both paths share the same underlying stage-detection logic, so behaviour
is consistent.

FUTURE / PRODUCTION VERSION (not implemented here):
- Replace substring matching with an intent classifier / embeddings.
- Track stage transitions per-speaker-turn instead of on raw concatenated
  text, so trajectory reflects true conversational order.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# Ordered stages of the classic scam script. Order matters — it defines
# what "progressing through the script" means.
STAGE_ORDER = [
    "authority_claim",
    "problem_issue",
    "urgency_pressure",
    "credential_request",
    "financial_action",
]

# Simple, explainable keyword/phrase patterns per stage. Lower-cased
# substring match — good enough for an SIH demo, easy to extend.
# NOTE: kept as plain data (not code) so it's an easy swap point for a
# future NLP/ML model — see module docstring.
STAGE_PATTERNS: Dict[str, List[str]] = {
    "authority_claim": [
        "i am calling from", "i'm calling from", "this is your bank",
        "calling from the bank", "bank representative", "banking official",
        "government official", "i am an officer", "from the income tax",
        "from customs", "i represent", "on behalf of your bank",
        "calling on behalf of", "cyber cell", "from the police",
        "reserve bank", "rbi official",
    ],
    "problem_issue": [
        "problem with your account", "issue with your account",
        "your account has been", "your card has been blocked",
        "suspicious activity", "account is compromised",
        "account will be suspended", "unauthorized transaction",
        "your kyc", "kyc is pending", "account has been flagged",
        "policy has lapsed", "package could not be delivered",
    ],
    "urgency_pressure": [
        "immediately", "right now", "act now", "urgent", "urgently",
        "within the next", "before it's too late", "act fast",
        "final warning", "last chance", "act quickly",
        "time is running out", "failure to respond",
    ],
    "credential_request": [
        "otp", "one time password", "verification code", "the pin",
        "your pin", "share your password", "tell me the code",
        "aadhaar number", "card number", "cvv", "net banking password",
    ],
    "financial_action": [
        "transfer the money", "transfer now", "send money",
        "make a payment", "pay now", "upi", "wire the money",
        "deposit the amount", "click this link to pay", "gift card",
        "pay the fine", "processing fee",
    ],
}

# Weight increases the later the stage sits in the script — reaching
# "financial_action" is far more alarming than only ever hitting
# "authority_claim".
_STAGE_WEIGHT = {stage: idx + 1 for idx, stage in enumerate(STAGE_ORDER)}
_MAX_WEIGHT_SUM = sum(_STAGE_WEIGHT.values())  # 1+2+3+4+5 = 15

# Bonus applied when detected stages appear in the expected escalating
# order (a coherent script), on top of the raw coverage score.
_ORDER_BONUS = 0.15


def _detect_stages_in_text(text: str) -> List[str]:
    """Return the list of stage names whose patterns appear in `text`."""
    text_lower = (text or "").lower()
    hits = []
    for stage in STAGE_ORDER:
        patterns = STAGE_PATTERNS[stage]
        if any(p in text_lower for p in patterns):
            hits.append(stage)
    return hits


def _is_monotonic(indices: List[int]) -> bool:
    """True if indices are non-decreasing (script progressing forward)."""
    return all(a <= b for a, b in zip(indices, indices[1:]))


@dataclass
class TrajectoryResult:
    detected_stages: List[str]
    current_stage: Optional[str]
    trajectory_risk: float  # 0.0 - 1.0
    in_order: bool
    explanation: str


def _score_stage_set(ordered_first_seen: List[str]) -> TrajectoryResult:
    """
    Shared scoring logic. `ordered_first_seen` is the list of stage names
    in the order they were FIRST detected (chronological).
    """
    if not ordered_first_seen:
        return TrajectoryResult(
            detected_stages=[],
            current_stage=None,
            trajectory_risk=0.0,
            in_order=True,
            explanation="No scam-script stages detected yet.",
        )

    unique_stages = list(dict.fromkeys(ordered_first_seen))  # preserve order, dedupe
    coverage = sum(_STAGE_WEIGHT[s] for s in unique_stages) / _MAX_WEIGHT_SUM

    indices = [STAGE_ORDER.index(s) for s in unique_stages]
    in_order = _is_monotonic(indices)

    risk = coverage + (_ORDER_BONUS if (in_order and len(unique_stages) > 1) else 0.0)
    risk = float(min(risk, 1.0))

    current_stage = STAGE_ORDER[max(indices)]

    readable = " -> ".join(unique_stages)
    if in_order and len(unique_stages) > 1:
        explanation = (
            f"Detected stages in escalating order: {readable}. "
            f"This matches the classic authority -> problem -> urgency -> "
            f"credential/financial-action social-engineering script."
        )
    else:
        explanation = f"Detected stages (order not fully sequential): {readable}."

    return TrajectoryResult(
        detected_stages=unique_stages,
        current_stage=current_stage,
        trajectory_risk=risk,
        in_order=in_order,
        explanation=explanation,
    )


def analyze_transcript(full_transcript_text: str) -> TrajectoryResult:
    """
    Stateless single-shot analysis over the whole transcript seen so far.

    This mirrors how context_signals.get_context_risk_score() is used in
    the current dashboard (call-level, re-evaluated as more transcript
    becomes available), so it drops into full_demo.py the same way.

    NOTE: because this looks at one blob of text, "order of first
    detection" is approximated by scanning stage patterns in
    STAGE_ORDER's own sequence when they all appear in the same string.
    For a real ordering guarantee, use ConversationTrajectoryEngine and
    feed it utterances one at a time as they are actually spoken.
    """
    hits = _detect_stages_in_text(full_transcript_text)
    return _score_stage_set(hits)


class ConversationTrajectoryEngine:
    """
    Stateful engine for turn-by-turn use (future real multi-turn call
    integration). Call `update(utterance_text)` each time a new line of
    transcript becomes available; the engine remembers which stages have
    fired and in what order.
    """

    def __init__(self):
        self._first_seen_order: List[str] = []
        self._seen_set = set()

    def reset(self) -> None:
        self._first_seen_order = []
        self._seen_set = set()

    def update(self, utterance_text: str) -> TrajectoryResult:
        """Feed one new utterance and get the updated trajectory result."""
        for stage in _detect_stages_in_text(utterance_text):
            if stage not in self._seen_set:
                self._seen_set.add(stage)
                self._first_seen_order.append(stage)
        return _score_stage_set(self._first_seen_order)

    def current_result(self) -> TrajectoryResult:
        """Get the current result without adding new text."""
        return _score_stage_set(self._first_seen_order)
