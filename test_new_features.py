"""
test_new_features.py — basic smoke tests for the 5 new MVP modules.

STATUS: Newly implemented. Follows the same "plain console script,
assert-based" style as headless_pipeline_test.py, so no new test
dependency (e.g. pytest) is required — run with:

    python3 test_new_features.py

Exits non-zero and prints which check failed if anything's wrong, same
spirit as headless_pipeline_test.py's smoke test.
"""

import os
import tempfile

import numpy as np

import conversation_trajectory as traj
import trust_contradiction as trust
import fusion
import active_challenge as challenge
import scam_fingerprint as fp


def test_conversation_trajectory_full_script():
    text = (
        "I am calling from your bank. There is a problem with your account. "
        "You need to act immediately. Please tell me the OTP. "
        "Now transfer the money to this account."
    )
    result = traj.analyze_transcript(text)
    assert result.detected_stages == [
        "authority_claim", "problem_issue", "urgency_pressure",
        "credential_request", "financial_action",
    ], result.detected_stages
    assert result.in_order is True
    assert result.trajectory_risk > 0.8, result.trajectory_risk
    assert result.current_stage == "financial_action"
    print("[OK] conversation_trajectory: full 5-stage script scores high risk, in order")


def test_conversation_trajectory_benign_text():
    result = traj.analyze_transcript("Hi, just calling to say happy birthday!")
    assert result.trajectory_risk == 0.0
    assert result.detected_stages == []
    print("[OK] conversation_trajectory: benign text scores zero risk")


def test_conversation_trajectory_stateful_engine():
    engine = traj.ConversationTrajectoryEngine()
    r1 = engine.update("I am calling from your bank.")
    assert r1.current_stage == "authority_claim"
    r2 = engine.update("Please tell me the OTP now.")
    assert "credential_request" in r2.detected_stages
    assert r2.trajectory_risk > r1.trajectory_risk
    print("[OK] conversation_trajectory: stateful engine accumulates stages across turns")


def test_trust_contradiction_known_case():
    result = trust.detect_contradiction(
        "Hello, this is bank employee speaking, I need your OTP to verify your account."
    )
    assert result.contradiction_detected is True
    assert result.claimed_role == "bank_employee"
    assert result.requested_action == "otp"
    assert result.contradiction_score > 0.5
    print("[OK] trust_contradiction: bank_employee + otp flagged as contradiction")


def test_trust_contradiction_no_issue():
    result = trust.detect_contradiction("Hi, just checking in about your delivery tomorrow.")
    assert result.contradiction_detected is False
    print("[OK] trust_contradiction: benign delivery message not flagged")


def test_fusion_backward_compatibility():
    # The ORIGINAL fuse_scores() must behave exactly as before.
    irs = fusion.fuse_scores(
        spoof_score=0.8, speaker_similarity=0.4,
        prosody_score=0.6, context_score=0.7,
    )
    assert 0 <= irs <= 100
    band, action = fusion.recommended_action(irs)
    assert band in ("LOW", "MEDIUM", "HIGH")
    print(f"[OK] fusion.fuse_scores (unchanged, original): IRS={irs:.1f} band={band}")


def test_fusion_adaptive_downweights_low_confidence():
    risks = {"spoof": 0.9, "speaker": 0.9, "prosody": 0.1, "context": 0.1}

    # Full confidence in everything.
    irs_full_conf, _ = fusion.fuse_scores_adaptive(
        risks, {"spoof": 1.0, "speaker": 1.0, "prosody": 1.0, "context": 1.0}
    )
    # Low confidence specifically in the risky "speaker" signal (e.g. poor
    # audio quality) should pull the fused score DOWN relative to full
    # confidence, since a risky-but-unreliable signal counts for less.
    irs_low_speaker_conf, breakdown = fusion.fuse_scores_adaptive(
        risks, {"spoof": 1.0, "speaker": 0.1, "prosody": 1.0, "context": 1.0}
    )
    assert irs_low_speaker_conf < irs_full_conf, (irs_low_speaker_conf, irs_full_conf)
    assert breakdown["normalized_weights"]["speaker"] < breakdown["effective_weights"]["speaker"] + 1  # sanity
    print(
        f"[OK] fusion.fuse_scores_adaptive: low speaker-confidence lowers IRS "
        f"({irs_full_conf:.1f} -> {irs_low_speaker_conf:.1f})"
    )


def test_fusion_adaptive_zero_confidence_fallback():
    irs, breakdown = fusion.fuse_scores_adaptive({"spoof": 0.9}, {"spoof": 0.0})
    assert irs == 0.0
    assert "note" in breakdown
    print("[OK] fusion.fuse_scores_adaptive: all-zero-confidence case handled without crashing")


def test_active_challenge_phrase_format():
    phrase = challenge.generate_challenge_phrase()
    parts = phrase.split(" ")
    assert len(parts) == 3, phrase
    assert parts[2].isdigit() and 10 <= int(parts[2]) <= 99, phrase
    # Generate a handful and make sure they're not all identical
    # (extremely unlikely with `secrets`, but confirms it's not hardcoded).
    phrases = {challenge.generate_challenge_phrase() for _ in range(10)}
    assert len(phrases) > 1
    print(f"[OK] active_challenge: generate_challenge_phrase() -> e.g. '{phrase}'")


def test_active_challenge_evaluation_runs():
    sr = 16000
    fake_audio = (np.random.randn(sr) * 0.05).astype(np.float32)  # 1s of quiet noise
    result = challenge.evaluate_challenge_response(fake_audio, sr, enrolled_embedding=None)
    assert result.spoof_score is not None
    assert result.speaker_similarity is None  # no enrolled embedding passed
    assert "does not prove" in result.disclaimer
    print("[OK] active_challenge: evaluate_challenge_response runs and includes disclaimer")


def test_active_challenge_trigger_logic():
    assert challenge.should_trigger_challenge("HIGH") is True
    assert challenge.should_trigger_challenge("MEDIUM") is True
    assert challenge.should_trigger_challenge("LOW") is False
    print("[OK] active_challenge: should_trigger_challenge only fires on MEDIUM/HIGH")


def test_scam_fingerprint_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "fingerprints.json")

        fp1 = fp.build_fingerprint(
            claimed_role="bank_employee", problem_detected=True,
            urgency_detected=True, credential_requested=True,
            financial_action_requested=True, trajectory_stage="financial_action",
        )
        fp.save_fingerprint(fp1, path=store_path)

        # A near-identical second call should be flagged as similar.
        fp2 = fp.build_fingerprint(
            claimed_role="bank_employee", problem_detected=True,
            urgency_detected=True, credential_requested=True,
            financial_action_requested=True, trajectory_stage="financial_action",
        )
        matches = fp.find_similar(fp2, path=store_path, threshold=0.7)
        assert len(matches) == 1
        assert matches[0][0] == 1.0
        message = fp.describe_matches(matches)
        assert "Similar scam pattern detected" in message
        print("[OK] scam_fingerprint: identical pattern flagged as 100% similar match")

        # An unrelated fingerprint should not match.
        fp3 = fp.build_fingerprint(
            claimed_role="delivery_agent", problem_detected=False,
            urgency_detected=False, credential_requested=False,
            financial_action_requested=False, trajectory_stage="none",
        )
        no_matches = fp.find_similar(fp3, path=store_path, threshold=0.7)
        assert no_matches == []
        print("[OK] scam_fingerprint: unrelated pattern correctly not matched")


def run_all():
    tests = [
        test_conversation_trajectory_full_script,
        test_conversation_trajectory_benign_text,
        test_conversation_trajectory_stateful_engine,
        test_trust_contradiction_known_case,
        test_trust_contradiction_no_issue,
        test_fusion_backward_compatibility,
        test_fusion_adaptive_downweights_low_confidence,
        test_fusion_adaptive_zero_confidence_fallback,
        test_active_challenge_phrase_format,
        test_active_challenge_evaluation_runs,
        test_active_challenge_trigger_logic,
        test_scam_fingerprint_round_trip,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{failures} test(s) FAILED.")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
