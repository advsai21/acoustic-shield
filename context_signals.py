"""
Role 4: Contextual / NLP Risk Signals

This module analyzes a transcript for suspicious conversational patterns.

It uses:
- Explicit risk phrases
- Controlled fuzzy matching for minor transcription errors
- Clause-aware negation handling
- Urgency + sensitive-request combinations
- Call metadata signals

The public function signatures are kept compatible with the existing
Acoustic Shield pipeline.

CHANGES vs. the previous version (bug fixes found during testing):
1. Negation was checked over a blind fixed 6-word window before a match.
   That worked for short lines but wrongly FLAGGED realistic, longer bank
   disclaimers as risky (e.g. "...will never, under any circumstances,
   ask you to share your otp or your pin with anyone" -- "never" was more
   than 6 words away from "otp"/"pin"). Negation is now scoped to the
   current clause instead of a fixed token count: sentence-ending
   punctuation (. ! ?) is tracked as a clause boundary before it gets
   stripped, and negation only looks back as far as the start of the
   current clause. Commas are intentionally NOT treated as clause breaks,
   since disclaimers naturally use commas mid-sentence.
2. A few single generic words ("pin", "password", "urgent", "urgently",
   "immediately", "right now") were matching completely ordinary sentences
   ("please enter your pin to continue", "reset my password on the
   website"). These have been replaced with more specific multi-word
   phrases that require an actual sharing/request context ("share your
   pin", "give me your password", "act immediately", etc.). "otp" and
   "cvv" are intentionally kept as bare words -- they're rarely spoken in
   unrelated everyday conversation, so the false-positive risk is much
   lower. If your test scripts show otherwise, apply the same
   compound-phrase pattern to them.
"""

import re
import difflib
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# Risk phrase banks
# ---------------------------------------------------------------------

RISK_PHRASES = {
    "credential_request": {
        "weight": 0.45,
        "phrases": [
            "otp",
            "cvv",
            "one time password",
            "verification code",
            "security code",
            "share your pin",
            "share the pin",
            "share your otp",
            "share your password",
            "share your cvv",
            "give me your pin",
            "give me your password",
            "give me your otp",
            "tell me your pin",
            "tell me your password",
            "tell me your otp",
            "read out your pin",
            "read out your otp",
            "read out the otp",
            "send me your otp",
            "what is your pin",
            "what is your password",
        ],
    },

    "financial_urgency": {
        "weight": 0.40,
        "phrases": [
            "send the money",
            "transfer the money",
            "transfer funds",
            "send the amount",
            "make the payment",
            "pay the amount",
            "send money now",
        ],
    },

    "secrecy_coercion": {
        "weight": 0.35,
        "phrases": [
            "do not tell anyone",
            "don't tell anyone",
            "keep this secret",
            "keep this confidential",
            "do not tell your family",
            "don't tell your family",
            "do not contact anyone",
            "don't contact anyone",
        ],
    },

    "authority_impersonation": {
        "weight": 0.30,
        "phrases": [
            "this is your bank",
            "this is the bank",
            "this is your bank calling",
            "this is bank calling",
            "i am calling from your bank",
            "calling from the bank",
            "this is from the bank",
        ],
    },

    "urgency_pressure": {
        "weight": 0.20,
        "phrases": [
            "act now",
            "act immediately",
            "do it immediately",
            "do this right now",
            "you must do this now",
            "call back immediately",
            "respond immediately",
            "share it immediately",
            "last warning",
            "final notice",
            "before it is too late",
            "your account will be suspended",
            "your account will be blocked",
            "your sim will be blocked",
        ],
    },
}


# ---------------------------------------------------------------------
# Common transcription / conversational normalizations
# ---------------------------------------------------------------------

NORMALIZATIONS = {
    "ur": "your",
    "u": "you",
    "pls": "please",
    "plz": "please",
}

# Internal marker used to remember where a sentence ended, so negation
# scope can be limited to "this clause" rather than a blind word count.
# Deliberately plain lowercase letters so it behaves like a normal token
# under \b word-boundary regex and can't collide with any real phrase.
CLAUSE_BOUNDARY = "sentencesplit"


def _normalize_text(text: str) -> str:
    """Normalize text for safer matching."""
    text = str(text).lower()

    # Mark sentence boundaries BEFORE stripping punctuation. Commas are
    # deliberately left alone (turned into spaces below like other
    # punctuation) so a comma-heavy disclaimer sentence stays one clause.
    text = re.sub(r"[.!?]+", f" {CLAUSE_BOUNDARY} ", text)

    # Remove remaining punctuation while preserving word boundaries.
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Normalize common shorthand.
    words = text.split()
    words = [NORMALIZATIONS.get(word, word) for word in words]

    return " ".join(words)


# ---------------------------------------------------------------------
# Negation handling
# ---------------------------------------------------------------------

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "dont",
    "don't",
    "isn't",
    "isnt",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
}


def _is_negated(text: str, start_index: int) -> bool:
    """
    Check whether a matched phrase is preceded by a nearby negation,
    scoped to the current clause (i.e. it won't look back past a
    sentence-ending . ! or ? -- and it also won't miss a negation that's
    further back in a long comma-separated disclaimer sentence).

    Example that must stay NOT risky:
        "our bank staff will never, under any circumstances, ask you to
         share your otp or your pin with anyone"

    Example that must NOT be suppressed (different clause):
        "we will never lie to you. now please share your otp immediately"
    """

    prefix_words = text[:start_index].split()

    if not prefix_words:
        return False

    # Restrict the search window to the current clause only.
    if CLAUSE_BOUNDARY in prefix_words:
        last_boundary = len(prefix_words) - 1 - prefix_words[::-1].index(CLAUSE_BOUNDARY)
        window = prefix_words[last_boundary + 1:]
    else:
        window = prefix_words

    if not window:
        return False

    for word in NEGATION_WORDS:
        if word in window:
            return True

    # Handle two-word negations ("do not", "did not", "will not") --
    # normalization keeps apostrophes but these are written as separate
    # words, so check the immediately preceding pair.
    if len(window) >= 2:
        previous = " ".join(window[-2:])
        if previous in {"do not", "did not", "will not"}:
            return True

    return False


# ---------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------

def _exact_match(text: str, phrase: str):
    """
    Find an exact word-boundary match.

    Returns the match object or None.
    """
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text)


# ---------------------------------------------------------------------
# Controlled fuzzy matching
# ---------------------------------------------------------------------

def _fuzzy_phrase_match(text: str, phrase: str, threshold: float = 0.86):
    """
    Controlled fuzzy matching.

    Important:
    - Single-word phrases are NOT fuzzy matched.
    - Multi-word phrases must have the same number of words.
    - Matching is performed token-by-token.

    This prevents bad matches such as:
        "send the money"
    being incorrectly matched to:
        "send the amount"
    """

    text_words = text.split()
    phrase_words = phrase.split()

    # Do not fuzzy-match single words.
    if len(phrase_words) < 2:
        return None

    phrase_len = len(phrase_words)

    if len(text_words) < phrase_len:
        return None

    best_score = 0.0
    best_start = None

    for i in range(len(text_words) - phrase_len + 1):
        window = text_words[i:i + phrase_len]

        # Don't let a fuzzy match span across a clause boundary.
        if CLAUSE_BOUNDARY in window:
            continue

        similarities = []

        for actual, expected in zip(window, phrase_words):
            similarity = difflib.SequenceMatcher(
                None,
                actual,
                expected
            ).ratio()

            similarities.append(similarity)

        score = sum(similarities) / len(similarities)

        if score > best_score:
            best_score = score
            best_start = i

    if best_score >= threshold:
        return best_start, best_score

    return None


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def get_context_risk_score(
    transcript: str,
    metadata: Dict
) -> Tuple[float, List[str]]:
    """
    Analyze transcript + call metadata.

    Returns:
        (risk_score, matched_phrases)

    risk_score is normalized to 0.0 - 1.0.
    """

    if not transcript:
        transcript = ""

    text = _normalize_text(transcript)

    matched_phrases: List[str] = []
    matched_categories = set()

    # -------------------------------------------------------------
    # Exact phrase matching
    # -------------------------------------------------------------

    for category, config in RISK_PHRASES.items():
        for phrase in config["phrases"]:

            match = _exact_match(text, phrase)

            if match:
                if _is_negated(text, match.start()):
                    continue

                if phrase not in matched_phrases:
                    matched_phrases.append(phrase)

                matched_categories.add(category)

    # -------------------------------------------------------------
    # Fuzzy matching
    # -------------------------------------------------------------

    for category, config in RISK_PHRASES.items():
        for phrase in config["phrases"]:

            # Don't fuzzy-match something already matched exactly.
            if phrase in matched_phrases:
                continue

            fuzzy_result = _fuzzy_phrase_match(text, phrase)

            if fuzzy_result is None:
                continue

            word_index, _score = fuzzy_result

            text_words = text.split()

            # Convert word index into a character position.
            char_position = len(" ".join(text_words[:word_index]))

            if word_index > 0:
                char_position += 1

            if _is_negated(text, char_position):
                continue

            if phrase not in matched_phrases:
                matched_phrases.append(phrase)

            matched_categories.add(category)

    # -------------------------------------------------------------
    # Calculate base score
    # -------------------------------------------------------------

    score = 0.0

    for category in matched_categories:
        score += RISK_PHRASES[category]["weight"]

    # -------------------------------------------------------------
    # Urgency + sensitive request combination
    # -------------------------------------------------------------

    has_urgency = "urgency_pressure" in matched_categories
    has_sensitive_request = (
        "credential_request" in matched_categories
        or "financial_urgency" in matched_categories
    )

    if has_urgency and has_sensitive_request:
        score += 0.20

    # -------------------------------------------------------------
    # Metadata signals
    # -------------------------------------------------------------

    metadata = metadata or {}

    if metadata.get("unknown_number", False):
        score += 0.15

    if metadata.get("odd_call_time", False):
        score += 0.10

    if metadata.get("first_contact", False):
        score += 0.10

    # -------------------------------------------------------------
    # Final normalization
    # -------------------------------------------------------------

    score = min(score, 1.0)

    return round(score, 2), matched_phrases


# ---------------------------------------------------------------------
# Compatibility wrapper
# ---------------------------------------------------------------------

def get_context_risk_score_only(
    transcript: str,
    metadata: Dict
) -> float:
    """
    Return only the numeric context risk score.

    Kept for compatibility with code that only needs the score.
    """

    score, _ = get_context_risk_score(transcript, metadata)

    return score


# ---------------------------------------------------------------------
# Detailed analysis helper
# ---------------------------------------------------------------------

def get_context_analysis(
    transcript: str,
    metadata: Dict
) -> Dict:
    """
    Return a structured explanation of the contextual analysis.
    """

    score, matched_phrases = get_context_risk_score(
        transcript,
        metadata
    )

    return {
        "score": score,
        "matched_phrases": matched_phrases,
        "metadata": metadata or {},
    }
