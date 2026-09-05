"""
app/models/session.py

In-memory session/enrollment state.

Per the privacy requirements (FR-10), enrolled speaker embeddings and
per-session rolling audio buffers are kept ONLY in memory for this
prototype — never written to disk. Only privacy-safe scalar analysis
results are persisted (see app/storage/repository.py).

This is intentionally a simple in-process dict, not a database model:
for a hackathon MVP with a single backend process, that's enough, and
it keeps raw audio/embeddings guaranteed off disk without needing to
trust every code path to remember not to log/persist them.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.ml.adapters import RollingAudioBuffer


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class LastAnalysis:
    spoof_score: float
    speaker_similarity: float
    prosody_anomaly_score: float
    context_risk_score: float
    raw_irs: float
    smoothed_irs: float
    risk_band: str
    recommended_action: str


@dataclass
class SessionState:
    session_id: str
    speaker_id: Optional[str] = None
    enrolled_embedding: Optional[np.ndarray] = None  # in-memory only, never persisted
    speaker_buffer: Optional[RollingAudioBuffer] = None
    similarity_history: List[float] = field(default_factory=list)
    smoothed_irs: Optional[float] = None
    last_analysis: Optional[LastAnalysis] = None


class SessionNotFoundError(KeyError):
    pass


class SpeakerNotFoundError(KeyError):
    pass


class SessionStore:
    """Thread-safe in-memory store for active sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(self, speaker_id: Optional[str] = None) -> SessionState:
        session_id = new_id("sess")
        state = SessionState(session_id=session_id, speaker_id=speaker_id)
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)
        return state

    def set_enrollment(self, session_id: str, speaker_id: str, embedding: np.ndarray) -> None:
        state = self.get(session_id)
        state.speaker_id = speaker_id
        state.enrolled_embedding = embedding

    def update_after_analysis(self, session_id: str, analysis: LastAnalysis, similarity: float) -> None:
        state = self.get(session_id)
        state.smoothed_irs = analysis.smoothed_irs
        state.last_analysis = analysis
        state.similarity_history.append(similarity)


# Module-level singleton — simplest option for a single-process hackathon
# backend. If this ever needs multi-process/horizontal scaling, swap this
# for a shared store (e.g. Redis) behind the same interface.
session_store = SessionStore()
