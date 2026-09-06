"""
Tests for the FastAPI endpoints: /health, /enroll, /analyze,
/score/{session_id}, and error handling for missing sessions /
invalid audio.

ML adapter calls (spoof/prosody/speaker/context scoring) are
monkeypatched to deterministic stub values so these tests don't need
resemblyzer's pretrained model downloaded, and so the fusion math can
be checked against known inputs end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml import adapters

client = TestClient(app)


@pytest.fixture(autouse=True)
def stub_ml_adapters(monkeypatch):
    """Replace real ML calls with deterministic stubs for every test in this module."""
    monkeypatch.setattr(adapters, "create_enrollment_embedding", lambda audio, sr: np.zeros(256, dtype=np.float32))
    monkeypatch.setattr(adapters, "get_spoof_score", lambda audio_chunk, sr: 0.8)
    monkeypatch.setattr(adapters, "get_prosody_anomaly_score", lambda audio_chunk, sr: 0.6)
    monkeypatch.setattr(adapters, "get_speaker_similarity", lambda audio_chunk, sr, emb: 0.3)
    monkeypatch.setattr(adapters, "get_context_risk_score", lambda text, meta: 0.7)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_enroll_creates_session(mono_wav_bytes):
    resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "enrolled"
    assert body["speaker_id"].startswith("spk_")
    assert body["session_id"].startswith("sess_")


def test_enroll_rejects_invalid_audio():
    resp = client.post("/enroll", files={"audio": ("bad.wav", b"not audio", "audio/wav")})
    assert resp.status_code == 400


def test_analyze_full_pipeline(mono_wav_bytes):
    enroll_resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    session_id = enroll_resp.json()["session_id"]

    resp = client.post(
        "/analyze",
        files={"audio": ("chunk.wav", mono_wav_bytes, "audio/wav")},
        data={
            "session_id": session_id,
            "transcript_text": "please share your otp now",
            "call_metadata": '{"unknown_number": true}',
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["session_id"] == session_id
    assert body["spoof_score"] == pytest.approx(0.8)
    assert body["prosody_anomaly_score"] == pytest.approx(0.6)
    assert body["speaker_similarity"] == pytest.approx(0.3)
    assert body["context_risk_score"] == pytest.approx(0.7)

    # IRS = 100 * (0.40*0.8 + 0.30*(1-0.3) + 0.15*0.6 + 0.15*0.7)
    #     = 100 * (0.32 + 0.21 + 0.09 + 0.105) = 100 * 0.725 = 72.5
    assert body["raw_irs"] == pytest.approx(72.5)
    assert body["smoothed_irs"] == pytest.approx(72.5)  # first analysis: EMA == raw IRS
    assert body["risk_band"] == "HIGH"
    assert body["recommended_action"] == "REQUIRE_MFA_OR_ESCALATE"


def test_analyze_missing_session_returns_404(mono_wav_bytes):
    resp = client.post(
        "/analyze",
        files={"audio": ("chunk.wav", mono_wav_bytes, "audio/wav")},
        data={"session_id": "sess_doesnotexist"},
    )
    assert resp.status_code == 404


def test_analyze_invalid_audio_returns_400(mono_wav_bytes):
    enroll_resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    session_id = enroll_resp.json()["session_id"]

    resp = client.post(
        "/analyze",
        files={"audio": ("bad.wav", b"garbage", "audio/wav")},
        data={"session_id": session_id},
    )
    assert resp.status_code == 400


def test_analyze_invalid_call_metadata_returns_400(mono_wav_bytes):
    enroll_resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    session_id = enroll_resp.json()["session_id"]

    resp = client.post(
        "/analyze",
        files={"audio": ("chunk.wav", mono_wav_bytes, "audio/wav")},
        data={"session_id": session_id, "call_metadata": "not json"},
    )
    assert resp.status_code == 400


def test_score_after_analysis(mono_wav_bytes):
    enroll_resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    session_id = enroll_resp.json()["session_id"]

    client.post(
        "/analyze",
        files={"audio": ("chunk.wav", mono_wav_bytes, "audio/wav")},
        data={"session_id": session_id},
    )

    resp = client.get(f"/score/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session_id
    assert body["risk_band"] == "HIGH"
    assert body["contributing_factors"]["speaker_mismatch"] == pytest.approx(0.7)  # 1 - 0.3


def test_score_before_any_analysis_returns_404(mono_wav_bytes):
    enroll_resp = client.post("/enroll", files={"audio": ("sample.wav", mono_wav_bytes, "audio/wav")})
    session_id = enroll_resp.json()["session_id"]

    resp = client.get(f"/score/{session_id}")
    assert resp.status_code == 404


def test_score_missing_session_returns_404():
    resp = client.get("/score/sess_doesnotexist")
    assert resp.status_code == 404
