"""
app/models/schemas.py

Pydantic request/response models. FastAPI auto-generates /docs from
these, so field names/types here are what a frontend teammate will see
in Swagger UI without reading any implementation code.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class EnrollResponse(BaseModel):
    speaker_id: str = Field(..., description="Generated identifier for the enrolled voiceprint.")
    session_id: str = Field(..., description="New session created for this speaker.")
    status: str = Field(default="enrolled")


class CallMetadata(BaseModel):
    """Optional call-context flags consumed by the (placeholder) contextual risk module."""

    unknown_number: bool = False
    odd_call_time: bool = False
    first_contact: bool = False


class AnalyzeResponse(BaseModel):
    session_id: str

    spoof_score: float = Field(..., ge=0.0, le=1.0)
    speaker_similarity: float = Field(..., ge=0.0, le=1.0)
    prosody_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    context_risk_score: float = Field(..., ge=0.0, le=1.0)

    raw_irs: float = Field(..., ge=0.0, le=100.0)
    smoothed_irs: float = Field(..., ge=0.0, le=100.0)

    risk_band: str
    recommended_action: str

    windows_processed: int = Field(
        ..., description="Number of audio windows this request was split into and aggregated over."
    )


class ScoreResponse(BaseModel):
    session_id: str
    irs: float = Field(..., ge=0.0, le=100.0, description="Latest smoothed IRS for this session.")
    risk_band: str
    contributing_factors: Dict[str, float]
    recommended_action: str


class ErrorResponse(BaseModel):
    detail: str


class TranscriptPayload(BaseModel):
    """Optional structured payload for context, if not sent as plain form fields."""

    transcript_text: Optional[str] = None
    call_metadata: Optional[CallMetadata] = None
