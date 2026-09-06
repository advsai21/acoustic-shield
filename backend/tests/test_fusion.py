"""
Tests for app/fusion/engine.py and app/fusion/actions.py — the fusion
formula, EMA smoothing, and risk-band classification.
"""

from __future__ import annotations

import pytest

from app.config import FusionWeights
from app.fusion import actions as action_engine
from app.fusion import engine as fusion_engine
from app.fusion.engine import ModuleScores


SRS_WEIGHTS = FusionWeights(spoof=0.40, speaker=0.30, prosody=0.15, context=0.15)


def test_fusion_formula_all_risky_gives_100():
    # spoof=1.0, speaker_similarity=0.0 (=> speaker_risk=1.0), prosody=1.0, context=1.0
    scores = ModuleScores(
        spoof_score=1.0, speaker_similarity=0.0, prosody_anomaly_score=1.0, context_risk_score=1.0
    )
    irs = fusion_engine.compute_irs(scores, weights=SRS_WEIGHTS)
    assert irs == pytest.approx(100.0)


def test_fusion_formula_all_safe_gives_0():
    # spoof=0, speaker_similarity=1 (=> speaker_risk=0), prosody=0, context=0
    scores = ModuleScores(
        spoof_score=0.0, speaker_similarity=1.0, prosody_anomaly_score=0.0, context_risk_score=0.0
    )
    irs = fusion_engine.compute_irs(scores, weights=SRS_WEIGHTS)
    assert irs == pytest.approx(0.0)


def test_fusion_formula_known_midpoint_value():
    # spoof=0.5, speaker_similarity=0.5 (=> speaker_risk=0.5), prosody=0.5, context=0.5
    # every term is 0.5, weights sum to 1.0, so IRS should be 100 * 0.5 = 50.
    scores = ModuleScores(
        spoof_score=0.5, speaker_similarity=0.5, prosody_anomaly_score=0.5, context_risk_score=0.5
    )
    irs = fusion_engine.compute_irs(scores, weights=SRS_WEIGHTS)
    assert irs == pytest.approx(50.0)


def test_fusion_uses_speaker_risk_not_raw_similarity():
    """A strong speaker match (similarity=1.0) must REDUCE risk, not add to it."""
    high_similarity = ModuleScores(
        spoof_score=0.0, speaker_similarity=1.0, prosody_anomaly_score=0.0, context_risk_score=0.0
    )
    low_similarity = ModuleScores(
        spoof_score=0.0, speaker_similarity=0.0, prosody_anomaly_score=0.0, context_risk_score=0.0
    )
    irs_high_sim = fusion_engine.compute_irs(high_similarity, weights=SRS_WEIGHTS)
    irs_low_sim = fusion_engine.compute_irs(low_similarity, weights=SRS_WEIGHTS)
    assert irs_high_sim < irs_low_sim


def test_scores_are_clamped_to_valid_range():
    scores = ModuleScores(
        spoof_score=1.5, speaker_similarity=-0.5, prosody_anomaly_score=2.0, context_risk_score=-1.0
    )
    irs = fusion_engine.compute_irs(scores, weights=SRS_WEIGHTS)
    assert 0.0 <= irs <= 100.0


def test_contributing_factors_reports_speaker_mismatch_not_similarity():
    scores = ModuleScores(
        spoof_score=0.2, speaker_similarity=0.9, prosody_anomaly_score=0.3, context_risk_score=0.1
    )
    factors = fusion_engine.contributing_factors(scores)
    assert factors["speaker_mismatch"] == pytest.approx(0.1)  # 1 - 0.9


def test_ema_first_score_equals_raw_irs():
    ema = fusion_engine.update_ema(previous_ema=None, new_irs=72.0, alpha=0.4)
    assert ema == pytest.approx(72.0)


def test_ema_subsequent_score_uses_smoothing_formula():
    # EMA_2 = 0.4 * 80 + 0.6 * 50 = 62.0
    ema = fusion_engine.update_ema(previous_ema=50.0, new_irs=80.0, alpha=0.4)
    assert ema == pytest.approx(62.0)


@pytest.mark.parametrize(
    "irs,expected_band",
    [
        (0, "LOW"),
        (30, "LOW"),
        (31, "MEDIUM"),
        (65, "MEDIUM"),
        (66, "HIGH"),
        (100, "HIGH"),
    ],
)
def test_risk_band_thresholds(irs, expected_band):
    decision = action_engine.classify(irs)
    assert decision.risk_band == expected_band


def test_risk_band_actions_match_spec():
    assert action_engine.classify(10).recommended_action == "ALLOW"
    assert action_engine.classify(50).recommended_action == "CALLBACK_VERIFICATION"
    assert action_engine.classify(90).recommended_action == "REQUIRE_MFA_OR_ESCALATE"


def test_classify_rejects_out_of_range_irs():
    with pytest.raises(action_engine.InvalidIRSError):
        action_engine.classify(150)
    with pytest.raises(action_engine.InvalidIRSError):
        action_engine.classify(-1)
