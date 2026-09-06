"""
app/api/routes.py

FR-9: API layer.

Orchestrates: ingestion -> ML adapters -> fusion engine -> action
engine -> storage, per session. This file should stay thin — the
actual math/logic lives in app.fusion / app.ingestion / app.ml, so
this module is mostly wiring + HTTP concerns (validation, status
codes, response shaping).

Aggregation policy for /analyze (documented per the brief's ask):
    When a single request's audio splits into multiple windows, this
    backend uses MEAN aggregation of each module's per-window scores
    before fusion (i.e. mean spoof_score across windows, mean
    prosody_anomaly_score across windows, etc.), except speaker
    similarity is computed against a per-session ROLLING BUFFER (see
    rolling_buffer.py) rather than per-window, then averaged across
    however many buffer-ready windows occurred in this request. This
    is a reasonable MVP choice — see backend/README.md "Design
    decisions" for the reasoning and alternatives (latest-window,
    weighted-recency) that a production version might use instead.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.config import get_settings
from app.fusion import actions as action_engine
from app.fusion import engine as fusion_engine
from app.ingestion.audio import AudioIngestionError, ingest
from app.ml import adapters
from app.models.schemas import (
    AnalyzeResponse,
    EnrollResponse,
    HealthResponse,
    ScoreResponse,
)
from app.models.session import (
    LastAnalysis,
    SessionNotFoundError,
    new_id,
    session_store,
)
from app.storage.repository import (
    AnalysisRecord,
    get_analysis_repository,
    get_session_repository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(audio: UploadFile) -> EnrollResponse:
    """
    Enroll a reference voiceprint from a short audio sample and create
    a new session for it.

    For this hackathon prototype, enrollment and session creation are
    combined into one call for simplicity (see backend/README.md
    "Design decisions" for why a production version might separate
    them — e.g. one speaker enrolling once, then starting many
    sessions).
    """
    raw_bytes = await audio.read()

    try:
        _, sr, windows = ingest(raw_bytes)
    except AudioIngestionError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}") from exc

    # Use the full preprocessed clip (not just one window) for enrollment —
    # more audio generally means a more reliable voiceprint.
    full_audio = windows[0] if len(windows) == 1 else _concat(windows)

    try:
        embedding = adapters.create_enrollment_embedding(full_audio, sr)
    except Exception as exc:
        logger.exception("Enrollment embedding failed")
        raise HTTPException(status_code=500, detail="Failed to process enrollment audio.") from exc

    speaker_id = new_id("spk")
    session = session_store.create_session(speaker_id=speaker_id)
    session_store.set_enrollment(session.session_id, speaker_id, embedding)

    # Speaker's rolling audio buffer for this session, sized per config.
    settings = get_settings()
    session.speaker_buffer = adapters.RollingAudioBuffer(
        sr=sr, window_seconds=settings.audio.speaker_buffer_seconds
    )

    # Only privacy-safe metadata is persisted — never the embedding or raw audio.
    get_session_repository().create(session.session_id, speaker_id)

    logger.info("enrollment completed speaker_id=%s session_id=%s", speaker_id, session.session_id)

    return EnrollResponse(speaker_id=speaker_id, session_id=session.session_id, status="enrolled")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    audio: UploadFile,
    session_id: str = Form(...),
    transcript_text: str = Form(default=""),
    call_metadata: str = Form(default="{}"),
) -> AnalyzeResponse:
    """
    Run one chunk/file of live audio through the full pipeline for an
    existing session and return the latest fused risk assessment.

    `call_metadata` is a JSON-encoded object (sent as a form field
    because this endpoint is multipart/form-data for the audio file),
    e.g. '{"unknown_number": true, "odd_call_time": false}'.
    """
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from exc

    if session.enrolled_embedding is None:
        raise HTTPException(
            status_code=404,
            detail="No enrolled speaker for this session. Call /enroll first.",
        )

    try:
        metadata_dict = json.loads(call_metadata) if call_metadata else {}
        if not isinstance(metadata_dict, dict):
            raise ValueError("call_metadata must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid call_metadata JSON: {exc}") from exc

    raw_bytes = await audio.read()
    try:
        _, sr, windows = ingest(raw_bytes)
    except AudioIngestionError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}") from exc

    logger.info("analysis started session_id=%s windows=%d", session_id, len(windows))

    if session.speaker_buffer is None:
        settings = get_settings()
        session.speaker_buffer = adapters.RollingAudioBuffer(
            sr=sr, window_seconds=settings.audio.speaker_buffer_seconds
        )

    spoof_scores, prosody_scores, speaker_sims = [], [], []

    for window in windows:
        try:
            spoof_scores.append(adapters.get_spoof_score(window, sr))
            prosody_scores.append(adapters.get_prosody_anomaly_score(window, sr))

            session.speaker_buffer.push(window)
            if session.speaker_buffer.is_ready():
                speaker_sims.append(
                    adapters.get_speaker_similarity(
                        session.speaker_buffer.get(), sr, session.enrolled_embedding
                    )
                )
        except Exception as exc:
            logger.exception("ML pipeline error on a window for session_id=%s", session_id)
            raise HTTPException(status_code=500, detail="Internal error while analyzing audio.") from exc

    # Context risk is computed once per request (transcript/metadata apply
    # to the whole request, not per-window).
    try:
        context_score = adapters.get_context_risk_score(transcript_text, metadata_dict)
    except Exception as exc:
        logger.exception("Context scoring failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Internal error while scoring context.") from exc

    if not speaker_sims:
        # Not enough buffered audio yet this session to trust a speaker
        # comparison — neutral fallback, consistent with how
        # speaker_verification.verify() itself handles too-short audio.
        speaker_sims = [0.5]

    mean_spoof = sum(spoof_scores) / len(spoof_scores)
    mean_prosody = sum(prosody_scores) / len(prosody_scores)
    mean_speaker_sim = sum(speaker_sims) / len(speaker_sims)

    scores = fusion_engine.ModuleScores(
        spoof_score=mean_spoof,
        speaker_similarity=mean_speaker_sim,
        prosody_anomaly_score=mean_prosody,
        context_risk_score=context_score,
    )
    raw_irs = fusion_engine.compute_irs(scores)
    smoothed_irs = fusion_engine.update_ema(session.smoothed_irs, raw_irs)

    decision = action_engine.classify(smoothed_irs)

    analysis = LastAnalysis(
        spoof_score=mean_spoof,
        speaker_similarity=mean_speaker_sim,
        prosody_anomaly_score=mean_prosody,
        context_risk_score=context_score,
        raw_irs=raw_irs,
        smoothed_irs=smoothed_irs,
        risk_band=decision.risk_band,
        recommended_action=decision.recommended_action,
    )

    previous_band = session.last_analysis.risk_band if session.last_analysis else None
    session_store.update_after_analysis(session_id, analysis, mean_speaker_sim)
    if previous_band and previous_band != decision.risk_band:
        logger.info(
            "risk band changed session_id=%s %s -> %s", session_id, previous_band, decision.risk_band
        )

    get_analysis_repository().save(
        AnalysisRecord(
            session_id=session_id,
            speaker_id=session.speaker_id,
            timestamp=_now_iso(),
            spoof_score=mean_spoof,
            speaker_similarity=mean_speaker_sim,
            prosody_anomaly_score=mean_prosody,
            context_risk_score=context_score,
            raw_irs=raw_irs,
            smoothed_irs=smoothed_irs,
            risk_band=decision.risk_band,
            recommended_action=decision.recommended_action,
        )
    )

    logger.info(
        "analysis completed session_id=%s risk_band=%s smoothed_irs=%.1f",
        session_id,
        decision.risk_band,
        smoothed_irs,
    )

    return AnalyzeResponse(
        session_id=session_id,
        spoof_score=mean_spoof,
        speaker_similarity=mean_speaker_sim,
        prosody_anomaly_score=mean_prosody,
        context_risk_score=context_score,
        raw_irs=raw_irs,
        smoothed_irs=smoothed_irs,
        risk_band=decision.risk_band,
        recommended_action=decision.recommended_action,
        windows_processed=len(windows),
    )


@router.get("/score/{session_id}", response_model=ScoreResponse)
def get_score(session_id: str) -> ScoreResponse:
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from exc

    if session.last_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet for this session.")

    last = session.last_analysis
    factors = fusion_engine.contributing_factors(
        fusion_engine.ModuleScores(
            spoof_score=last.spoof_score,
            speaker_similarity=last.speaker_similarity,
            prosody_anomaly_score=last.prosody_anomaly_score,
            context_risk_score=last.context_risk_score,
        )
    )

    return ScoreResponse(
        session_id=session_id,
        irs=last.smoothed_irs,
        risk_band=last.risk_band,
        contributing_factors=factors,
        recommended_action=last.recommended_action,
    )


def _concat(windows):
    import numpy as np

    return np.concatenate(windows)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
