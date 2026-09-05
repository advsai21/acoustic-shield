# Voice Integrity Verification — Backend (Role 3)

FastAPI orchestration layer sitting on top of the teammates' ML modules
at the repo root (`speaker_verification.py`, `prosody.py`,
`spoof_heuristic.py`, `context_signals.py`, `rolling_buffer.py`).

```
Client
  |
FastAPI (app/api/routes.py)
  |
Audio Ingestion (app/ingestion/audio.py)      -> decode, mono, resample, normalize, window
  |
ML Adapters (app/ml/adapters.py)              -> wraps repo-root teammate modules
  |  spoof detection | speaker verification | prosody | contextual risk
  |
Risk Fusion Engine (app/fusion/engine.py)     -> IRS formula, EMA smoothing
  |
Risk Band / Action Engine (app/fusion/actions.py)
  |
Privacy-preserving storage (app/storage/repository.py)  -> SQLite, no raw audio ever
  |
API response
```

## Project structure

```
backend/
  app/
    main.py              FastAPI app, global error handlers, logging setup
    config.py             loads + validates config.yaml (weights, thresholds, sizes)
    api/
      routes.py            /health, /enroll, /analyze, /score/{session_id}
    ingestion/
      audio.py             decode_audio, preprocess_audio, split_into_windows
    fusion/
      engine.py            IRS formula + EMA (pure math, no I/O)
      actions.py           risk band + recommended action (independent of engine.py)
    ml/
      adapters.py          integration boundary to repo-root ML modules
    models/
      schemas.py           Pydantic request/response models (drives /docs)
      session.py           in-memory session + enrollment state
    storage/
      repository.py        SQLite SessionRepository / AnalysisRepository
  config.yaml
  requirements.txt
  pytest.ini
  tests/
```

## Installation

From the repo root:

```bash
pip install -r requirements.txt              # teammates' ML deps (numpy, librosa, resemblyzer, ...)
pip install -r backend/requirements.txt      # backend deps (fastapi, uvicorn, ...)
```

If you hit `ModuleNotFoundError: No module named 'pkg_resources'` (resemblyzer's
issue, noted in the repo's top-level README): `pip install "setuptools<81"`.

## Running the server

```bash
cd backend
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI.

## Running tests

```bash
cd backend
pytest
```

Tests monkeypatch the ML adapter calls (spoof/prosody/speaker/context
scoring) to deterministic stub values, so they run fast and don't
require resemblyzer's pretrained model to download — the fusion math,
ingestion pipeline, and API contracts are what's actually being
verified.

## API endpoints

### `GET /health`
```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok"}
```

### `POST /enroll`
Enrolls a reference voiceprint and creates a session in one call (see
"Design decisions" below for why).

```bash
curl -X POST http://127.0.0.1:8000/enroll \
  -F "audio=@sample_genuine.wav"
```
```json
{"speaker_id": "spk_a1b2c3d4e5f6", "session_id": "sess_1a2b3c4d5e6f", "status": "enrolled"}
```

### `POST /analyze`
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "audio=@live_chunk.wav" \
  -F "session_id=sess_1a2b3c4d5e6f" \
  -F "transcript_text=please share your otp now" \
  -F 'call_metadata={"unknown_number": true, "odd_call_time": false}'
```
```json
{
  "session_id": "sess_1a2b3c4d5e6f",
  "spoof_score": 0.82,
  "speaker_similarity": 0.31,
  "prosody_anomaly_score": 0.67,
  "context_risk_score": 0.74,
  "raw_irs": 81.4,
  "smoothed_irs": 78.9,
  "risk_band": "HIGH",
  "recommended_action": "REQUIRE_MFA_OR_ESCALATE",
  "windows_processed": 2
}
```

### `GET /score/{session_id}`
```bash
curl http://127.0.0.1:8000/score/sess_1a2b3c4d5e6f
```
```json
{
  "session_id": "sess_1a2b3c4d5e6f",
  "irs": 78.9,
  "risk_band": "HIGH",
  "contributing_factors": {
    "spoof_score": 0.82,
    "speaker_mismatch": 0.69,
    "prosody_anomaly_score": 0.67,
    "context_risk_score": 0.74
  },
  "recommended_action": "REQUIRE_MFA_OR_ESCALATE"
}
```
Returns `404` if no analysis has run yet for that session, or if the
session doesn't exist.

## Configuration (`config.yaml`)

Everything tunable lives here — nothing is hardcoded in the app code:

- `audio.window_seconds` (default 2.0s), clamped between `min_window_seconds`
  (1.0s) and `max_window_seconds` (3.0s)
- `audio.speaker_buffer_seconds` (default 1.8s) — the rolling buffer size fed
  to `speaker_verification.verify()`, per the repo's own "important gotcha"
  about needing ~1.5-2s of audio for a reliable comparison
- `fusion.weights` — spoof 0.40 / speaker 0.30 / prosody 0.15 / context 0.15
  (validated at startup to sum to 1.0; startup fails loudly otherwise)
- `fusion.ema_alpha` (default 0.40)
- `risk_bands.low_max` / `medium_max` (default 30 / 65)
- `storage.database_url` (SQLite by default; swap by changing this file +
  `app/storage/repository.py`'s `_db_path_from_url`)

## ML adapter integration

`app/ml/adapters.py` is the **only** file that imports the repo-root
teammate modules. It adds the repo root to `sys.path` at import time
(two directories up from `backend/app/ml/`) so it works regardless of
whether you run `uvicorn` from `backend/` or the repo root.

| Adapter function | Wraps | Status |
|---|---|---|
| `get_spoof_score` | `spoof_heuristic.get_spoof_score` | Role 1 placeholder — swap the file, signature unchanged |
| `get_speaker_similarity` | `speaker_verification.verify` | Real (Resemblyzer) |
| `get_prosody_anomaly_score` | `prosody.get_prosody_anomaly_score` | Real (librosa heuristics) |
| `get_context_risk_score` | `context_signals.get_context_risk_score_only` | Role 4 placeholder — swap the file, signature unchanged |
| `create_enrollment_embedding` | `speaker_verification.enroll` | Real (Resemblyzer) |

**Gotcha carried over from the repo's own README:** `context_signals.get_context_risk_score()`
returns a `(score, matched_phrases)` tuple, not a bare float — the adapter
calls the repo's own `get_context_risk_score_only()` wrapper instead, so the
adapter's contract (`-> float`) holds regardless of which of the two you'd
call directly.

When Role 1's real spoof classifier and Role 4's real ASR+NLP pipeline are
ready, only `spoof_heuristic.py` and `context_signals.py` need to change —
`adapters.py`, `fusion/`, and `api/` don't need to know or care.

## Privacy model (FR-10)

- Raw audio, uploaded files, and waveforms are **never** written to disk —
  not in `app/storage/`, not in logs. `app/storage/repository.py` has no code
  path that accepts audio bytes.
- Enrolled speaker embeddings live only in the in-memory `SessionStore`
  (`app/models/session.py`) — lost on process restart, which is an accepted
  hackathon-MVP tradeoff (see below).
- SQLite persists only: `session_id`, `speaker_id`, `timestamp`, the four
  module scores, `raw_irs`, `smoothed_irs`, `risk_band`, `recommended_action`.
- Logging (`app/api/routes.py`, `app/storage/repository.py`) logs session
  lifecycle events (created, enrolled, analysis started/completed, risk band
  changes, errors) — never audio bytes or transcript content.

## Design decisions & assumptions

- **Enroll + session creation combined into one call.** The brief noted this
  could reasonably be split (e.g. one enrollment reused across many
  sessions). For the hackathon flow (one call = one session = one speaker)
  combining them is simpler and matches how `full_demo.py` uses the pipeline.
  Splitting later just means adding a separate `POST /sessions` that takes an
  existing `speaker_id`.
- **Multi-window aggregation in `/analyze` uses MEAN aggregation** across a
  request's windows for spoof/prosody scores, and mean-of-buffer-ready
  speaker similarity scores for speaker similarity (context is scored once
  per request, since transcript/metadata apply to the whole request, not a
  single window). Alternatives considered: latest-window-only (loses signal
  from the rest of the request) and weighted-recency (adds complexity not
  justified for an MVP). Mean is simple, hard to game with one outlier
  window, and easy to explain to judges.
- **In-memory session store, not a database-backed session model.** Keeps
  enrolled embeddings and rolling audio buffers guaranteed off disk without
  relying on every code path to remember not to persist them. Tradeoff:
  sessions don't survive a process restart — acceptable for a hackathon demo,
  flagged here for anyone hardening this later.
- **SQLite via stdlib `sqlite3`, not an ORM.** Two simple tables
  (`sessions`, `analyses`) don't need SQLAlchemy's overhead for this scope;
  `Database`/`SessionRepository`/`AnalysisRepository` in
  `app/storage/repository.py` are the only place that would need to change
  to move to Postgres.
- **Speaker similarity fallback of 0.5** when a session's rolling buffer
  hasn't accumulated enough audio yet this request — consistent with
  `speaker_verification.verify()`'s own neutral fallback for too-short audio,
  rather than inventing a different convention.
- **Denoising is explicitly not implemented** (`audio.denoise: false` in
  config.yaml). `denoise_audio()` in `app/ingestion/audio.py` is a no-op hook
  so a real implementation can be dropped in later without touching callers.

## Remaining TODOs

- Wire Role 1's real spoof classifier into `spoof_heuristic.py` (adapter
  needs no changes).
- Wire Role 4's real ASR + NLP into `context_signals.py` (adapter needs no
  changes).
- Decide whether sessions should survive a backend restart (would mean
  moving `SessionStore` state — minus the embedding/audio buffer, which
  should stay in-memory-only — into SQLite or another store).
- `detect_drift()` in `speaker_verification.py` (mid-call voice-swap
  detection from the similarity trend) isn't wired into `/analyze` yet — it's
  available via `session.similarity_history` if you want to surface it in a
  future response field.
- No auth/rate-limiting on the API — fine for a local hackathon demo, not for
  anything beyond that.
