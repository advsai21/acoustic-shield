"""
app/storage/repository.py

FR-10: Privacy-preserving logging / storage.

Uses plain sqlite3 (stdlib) rather than an ORM — sufficient for a
hackathon MVP and avoids adding a dependency just for two simple
tables. The repository classes are the ONLY code allowed to touch the
database, so swapping SQLite for Postgres later means changing this
file alone (change _connect() and the SQL dialect bits), not any
caller in api/ or fusion/.

NEVER persisted here, by construction — there is no code path in this
file that accepts or writes: raw audio, uploaded files, waveforms, or
audio bytes of any kind. Only scalar analysis results are written.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def _db_path_from_url(database_url: str) -> str:
    """Extract a filesystem path from a sqlite:/// URL."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Only sqlite:/// URLs are supported by this repository, got: {database_url}")
    return database_url[len(prefix):]


@dataclass
class AnalysisRecord:
    session_id: str
    speaker_id: Optional[str]
    timestamp: str
    spoof_score: float
    speaker_similarity: float
    prosody_anomaly_score: float
    context_risk_score: float
    raw_irs: float
    smoothed_irs: float
    risk_band: str
    recommended_action: str


@dataclass
class SessionRecord:
    session_id: str
    speaker_id: Optional[str]
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    speaker_id  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT NOT NULL,
    speaker_id              TEXT,
    timestamp               TEXT NOT NULL,
    spoof_score             REAL NOT NULL,
    speaker_similarity      REAL NOT NULL,
    prosody_anomaly_score   REAL NOT NULL,
    context_risk_score      REAL NOT NULL,
    raw_irs                 REAL NOT NULL,
    smoothed_irs            REAL NOT NULL,
    risk_band               TEXT NOT NULL,
    recommended_action      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);
"""


class Database:
    """Owns the sqlite connection/schema. One instance per process is fine for sqlite."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        url = database_url or get_settings().storage.database_url
        self._path = _db_path_from_url(url)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, session_id: str, speaker_id: Optional[str]) -> SessionRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT OR REPLACE INTO sessions (session_id, speaker_id, created_at) VALUES (?, ?, ?)",
            (session_id, speaker_id, created_at),
        )
        return SessionRecord(session_id=session_id, speaker_id=speaker_id, created_at=created_at)


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, record: AnalysisRecord) -> None:
        self._db.execute(
            """
            INSERT INTO analyses (
                session_id, speaker_id, timestamp,
                spoof_score, speaker_similarity, prosody_anomaly_score, context_risk_score,
                raw_irs, smoothed_irs, risk_band, recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.speaker_id,
                record.timestamp,
                record.spoof_score,
                record.speaker_similarity,
                record.prosody_anomaly_score,
                record.context_risk_score,
                record.raw_irs,
                record.smoothed_irs,
                record.risk_band,
                record.recommended_action,
            ),
        )
        logger.info("analysis persisted for session_id=%s risk_band=%s", record.session_id, record.risk_band)


# Module-level singletons, built lazily so importing this module doesn't
# immediately touch disk (helpful for unit tests that stub config).
_db_instance: Optional[Database] = None


def get_database() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def get_session_repository() -> SessionRepository:
    return SessionRepository(get_database())


def get_analysis_repository() -> AnalysisRepository:
    return AnalysisRepository(get_database())
