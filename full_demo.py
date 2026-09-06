"""
full_demo.py — Voice Integrity Verification Framework: FULL PIPELINE DEMO

Shows every module working together at once, so the whole team can see
how their pieces combine:

- Speaker Verification (Role 2)
- Prosody Analysis (Role 2)
- Spoof Detection (Role 1 — placeholder heuristic for now)
- Contextual Signals (Role 4 — real ASR + NLP risk scoring)
- Risk Fusion + Action (Role 3 — simplified stand-in for their backend)
- Dashboard (Role 5 — simplified stand-in for their UI)

TWO WAYS TO RUN THIS:

1. Built-in synthetic samples (no mic, no files needed) — good for a
   first end-to-end look and for testing before real audio is ready.
   NOTE: these are sine-wave proxies, not real speech, so ASR will
   transcribe little/nothing from them by design (see demo_audio.py).
   Role 4's context scoring only has something to work with once real
   speech is used — see option 2 below.

2. Upload your own WAV/MP3 files (a genuine sample + a "spoofed" sample,
   e.g. from a free TTS tool) for a more convincing/realistic demo,
   and to actually exercise the ASR + context-risk pipeline.

Run with: streamlit run full_demo.py

CHANGE vs. the previous version:
ASR transcription is now cached per audio clip instead of re-running on
every Streamlit rerun. Streamlit re-executes this whole script top
to bottom on any widget interaction — previously, ticking a checkbox
like "unknown number" would silently re-run Whisper transcription
from scratch every time. The transcript is now cached in session_state
keyed by a hash of the actual audio content, and only recomputed when
the audio itself changes. The NLP risk score is still recalculated every
rerun so toggling the metadata checkboxes updates the score immediately.

============================================================================
SIH EXTENSION — 5 new MVP features added on top of the pipeline above.
Everything above this note is ALREADY IMPLEMENTED and unchanged in
behavior. New/changed pieces are marked "NEW MVP" inline below, so it's
always clear what's demonstrable-but-basic vs. the original real modules.

  Feature 1 — Conversation Trajectory Engine   (conversation_trajectory.py)
  Feature 2 — Trust Contradiction Engine       (trust_contradiction.py)
  Feature 3 — Confidence-Adaptive Fusion       (fusion.fuse_scores_adaptive)
  Feature 4 — Active Voice Challenge           (active_challenge.py)
  Feature 5 — Scam Fingerprinting              (scam_fingerprint.py)

None of the original functions/files were removed. fusion.fuse_scores()
(the original) is left untouched; the dashboard below now calls the new
fusion.fuse_scores_adaptive() instead, per the Feature 3 requirement,
but the old function remains available for anything else that imports it.
============================================================================
"""

import hashlib
import time

import numpy as np
import streamlit as st
import librosa

import speaker_verification as sv
import prosody
import spoof_heuristic as spoof
import context_signals as ctx
import asr
import fusion
import demo_audio

from rolling_buffer import RollingAudioBuffer

# --- NEW MVP: Features 1, 2, 4, 5 ------------------------------------------
import conversation_trajectory as trajectory
import trust_contradiction as trust
import active_challenge as challenge
import scam_fingerprint as fingerprint
# -----------------------------------------------------------------------------

SR = 16000
CHUNK_SIZE = 4800  # 300ms at 16kHz

st.set_page_config(
    page_title="Voice Integrity — Full Pipeline Demo",
    layout="wide"
)

st.title("🛡️ Voice Integrity Verification Framework — Full Pipeline Demo")
st.caption(
    "All modules wired together: Spoof Detection · Speaker Verification · "
    "Prosody Analysis · Contextual Signals · Conversation Intelligence · "
    "Confidence-Adaptive Fusion · Active Verification · Scam Fingerprinting"
)

# ============================================================
# SESSION STATE
# ============================================================
if "enrolled_embedding" not in st.session_state:
    st.session_state.enrolled_embedding = None

if "history" not in st.session_state:
    st.session_state.history = {
        "irs": [],
        "spoof": [],
        "speaker_sim": [],
        "prosody": [],
        "context": []
    }

if "cached_audio_key" not in st.session_state:
    st.session_state.cached_audio_key = None

if "cached_transcript" not in st.session_state:
    st.session_state.cached_transcript = ""

# --- NEW MVP: session state for active challenge (Feature 4) and the
# fingerprint of the most recently processed call (Feature 5) -----------
if "challenge_phrase" not in st.session_state:
    st.session_state.challenge_phrase = None
if "challenge_evaluation" not in st.session_state:
    st.session_state.challenge_evaluation = None
if "last_fingerprint" not in st.session_state:
    st.session_state.last_fingerprint = None
# -------------------------------------------------------------------------


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_wav_file(uploaded_file):
    """Load an uploaded WAV/MP3 file as a float32 mono waveform at SR."""
    wav, _ = librosa.load(uploaded_file, sr=SR, mono=True)
    return wav.astype(np.float32)


def reset_history():
    st.session_state.history = {
        "irs": [],
        "spoof": [],
        "speaker_sim": [],
        "prosody": [],
        "context": []
    }
    # NEW MVP: clear any pending challenge/fingerprint tied to the old call.
    st.session_state.challenge_phrase = None
    st.session_state.challenge_evaluation = None
    st.session_state.last_fingerprint = None


def _audio_cache_key(audio_source: str, audio: np.ndarray) -> str:
    """
    Fingerprint the actual audio content so the transcript cache
    correctly invalidates when a new file is uploaded.
    """
    digest = hashlib.md5(
        np.asarray(audio).tobytes()
    ).hexdigest()
    return f"{audio_source}:{digest}"


def get_transcript_cached(
    audio_source: str,
    call_audio: np.ndarray
) -> str:
    """
    Run ASR only when the audio actually changes.
    Otherwise reuse the cached transcript.
    """
    key = _audio_cache_key(audio_source, call_audio)
    if st.session_state.cached_audio_key != key:
        st.session_state.cached_transcript = (
            asr.transcribe_audio(
                call_audio,
                sample_rate=SR
            )
        )
        st.session_state.cached_audio_key = key
    return st.session_state.cached_transcript


# ============================================================
# STEP 1 — ENROLLMENT   (ALREADY IMPLEMENTED — unchanged)
# ============================================================
st.header("Step 1 — Enroll Reference Voice")

enroll_source = st.radio(
    "Enrollment audio source",
    [
        "Use built-in synthetic sample (quick test)",
        "Upload a real WAV/MP3 file"
    ],
    horizontal=True
)

enroll_audio = None

if enroll_source == "Use built-in synthetic sample (quick test)":
    if st.button("🎙️ Enroll using synthetic 'genuine' sample"):
        enroll_audio = demo_audio.genuine_sample(seed=1)
else:
    enroll_file = st.file_uploader(
        "Upload reference voice sample",
        type=["wav", "mp3"],
        key="enroll_upload"
    )
    if enroll_file is not None:
        if st.button("🎙️ Enroll from uploaded file"):
            enroll_audio = load_wav_file(enroll_file)

if enroll_audio is not None:
    st.session_state.enrolled_embedding = sv.enroll(
        enroll_audio,
        SR
    )
    reset_history()
    st.success(
        f"Reference voiceprint enrolled "
        f"({len(enroll_audio) / SR:.1f}s sample)."
    )

if st.session_state.enrolled_embedding is None:
    st.warning(
        "Enroll a reference speaker above before running "
        "the pipeline below."
    )
    st.stop()

st.divider()

# ============================================================
# STEP 2 — CONTEXTUAL / CALL METADATA   (ALREADY IMPLEMENTED — unchanged)
# ============================================================
st.header("Step 2 — Contextual Signals")
st.caption(
    "Context risk is calculated from the call transcript "
    "generated by ASR and from call metadata."
)

ctx_col1, ctx_col2 = st.columns([2, 1])

with ctx_col1:
    st.info(
        "The transcript will be generated automatically from "
        "the call audio selected in Step 3."
    )

with ctx_col2:
    unknown_number = st.checkbox("Unknown number")
    odd_call_time = st.checkbox("Odd call time")
    first_contact = st.checkbox("First contact")

call_metadata = {
    "unknown_number": unknown_number,
    "odd_call_time": odd_call_time,
    "first_contact": first_contact
}

# Context score will be calculated after call audio is selected.
transcript_text = ""
context_score = 0.0
matched_phrases = []

# --- NEW MVP: placeholders for conversation-intelligence results, filled
# in once a transcript exists (Features 1 & 2). Defined here so later
# sections can reference them even if no call audio is selected yet.
trajectory_result = trajectory.analyze_transcript("")
contradiction_result = trust.detect_contradiction("")
# -------------------------------------------------------------------------

st.divider()

# ============================================================
# STEP 3 — CALL AUDIO + FULL PIPELINE   (ALREADY IMPLEMENTED — unchanged)
# ============================================================
st.header("Step 3 — Run the Full Pipeline on Call Audio")

audio_source = st.radio(
    "Call audio source",
    [
        "Synthetic: genuine-like",
        "Synthetic: cloned/TTS-like",
        "Synthetic: different speaker",
        "Upload a WAV/MP3 file"
    ],
    horizontal=True
)

call_audio = None

if audio_source == "Synthetic: genuine-like":
    call_audio = demo_audio.genuine_sample(seed=2)
elif audio_source == "Synthetic: cloned/TTS-like":
    call_audio = demo_audio.synthetic_sample()
elif audio_source == "Synthetic: different speaker":
    call_audio = demo_audio.different_speaker_sample()
else:
    call_file = st.file_uploader(
        "Upload call audio",
        type=["wav", "mp3"],
        key="call_upload"
    )
    if call_file is not None:
        call_audio = load_wav_file(call_file)

# ============================================================
# ROLE 4 — ASR + CONTEXT ANALYSIS   (ALREADY IMPLEMENTED — unchanged)
# plus NEW MVP: conversation trajectory + trust contradiction, which run
# on the SAME transcript text, the same way context_signals already does.
# ============================================================
if call_audio is not None:
    with st.spinner("Generating transcript with ASR..."):
        try:
            transcript_text = get_transcript_cached(
                audio_source,
                call_audio
            )
            context_score, matched_phrases = (
                ctx.get_context_risk_score(
                    transcript_text,
                    call_metadata
                )
            )
        except Exception as e:
            st.error(
                f"ASR/context analysis failed: {e}"
            )
            transcript_text = ""
            context_score = 0.0
            matched_phrases = []

    # --- NEW MVP: Features 1 & 2 run on the same transcript -------------
    trajectory_result = trajectory.analyze_transcript(transcript_text)
    contradiction_result = trust.detect_contradiction(transcript_text)
    # ----------------------------------------------------------------------

    st.markdown("### 📝 Generated Transcript")
    if transcript_text:
        st.write(transcript_text)
    else:
        st.write("No speech detected.")

    st.write(
        f"**Context risk score:** {context_score:.2f}"
        + (
            f" — matched: {matched_phrases}"
            if matched_phrases
            else ""
        )
    )

    # --- NEW MVP: Conversation Intelligence panel (Features 1 & 2) ------
    st.markdown("### 🧭 Conversation Intelligence (new MVP)")
    ci_col1, ci_col2 = st.columns(2)
    with ci_col1:
        st.write(
            f"**Current stage:** "
            f"{trajectory_result.current_stage or '—'}"
        )
        st.write(
            f"**Conversation trajectory risk:** "
            f"{trajectory_result.trajectory_risk:.2f}"
        )
        st.caption(trajectory_result.explanation)
    with ci_col2:
        st.write(
            f"**Trust contradiction score:** "
            f"{contradiction_result.contradiction_score:.2f}"
        )
        if contradiction_result.contradiction_detected:
            st.warning(contradiction_result.reason)
        else:
            st.caption(contradiction_result.reason)
    # ------------------------------------------------------------------------

st.divider()

# ============================================================
# RUN BUTTON   (ALREADY IMPLEMENTED — unchanged)
# ============================================================
run_btn = st.button(
    "▶️ Run Pipeline on This Audio"
)

# ============================================================
# LIVE DASHBOARD
# ============================================================
st.markdown("#### Voice Integrity")
col1, col2, col3, col4, col5 = st.columns(5)
metric_irs = col1.metric(
    "Impersonation Risk Score",
    "—"
)
metric_spoof = col2.metric(
    "Spoof Score",
    "—"
)
metric_speaker = col3.metric(
    "Speaker Similarity",
    "—"
)
metric_prosody = col4.metric(
    "Prosody Anomaly",
    "—"
)
metric_context = col5.metric(
    "Context Risk",
    "—"
)

# --- NEW MVP: extra metrics row for Features 1-3 ---------------------------
st.markdown("#### Conversation Intelligence & Adaptive Fusion (new MVP)")
col6, col7, col8 = st.columns(3)
metric_trajectory = col6.metric("Conversation Trajectory Risk", "—")
metric_trust = col7.metric("Trust Contradiction", "—")
metric_confidence_note = col8.empty()
# ----------------------------------------------------------------------------

status_box = st.empty()
chart_box = st.empty()
breakdown_box = st.empty()

# ============================================================
# PROCESS AUDIO
# ============================================================
if run_btn and call_audio is not None:
    reset_history()

    chunks = demo_audio.to_int16_chunks(
        call_audio,
        CHUNK_SIZE
    )

    status_box.info(
        f"Processing {len(chunks)} chunks "
        f"(~{len(call_audio) / SR:.1f}s of audio)..."
    )

    # Speaker verification needs more context than a single
    # ~300ms chunk.
    speaker_buffer = RollingAudioBuffer(
        sr=SR,
        window_seconds=1.8
    )

    last_speaker_sim = 0.5
    band, action, smoothed_irs = "LOW", "—", 0.0  # defined before loop for safety

    for chunk in chunks:
        # ----------------------------------------------------
        # Role 1 — Spoof Detection   (ALREADY IMPLEMENTED)
        # ----------------------------------------------------
        spoof_score = spoof.get_spoof_score(
            chunk,
            SR
        )

        # ----------------------------------------------------
        # Role 2 — Prosody Analysis   (ALREADY IMPLEMENTED)
        # ----------------------------------------------------
        prosody_score = (
            prosody.get_prosody_anomaly_score(
                chunk,
                SR
            )
        )

        # ----------------------------------------------------
        # Role 2 — Speaker Verification   (ALREADY IMPLEMENTED)
        # ----------------------------------------------------
        speaker_buffer.push(chunk)
        speaker_confidence = 0.3  # NEW MVP: low confidence until buffer fills
        if speaker_buffer.is_ready():
            last_speaker_sim = sv.verify(
                speaker_buffer.get(),
                SR,
                st.session_state.enrolled_embedding
            )
            speaker_confidence = 0.9  # NEW MVP: enough audio -> trust it more
        speaker_sim = last_speaker_sim

        # ----------------------------------------------------
        # Role 4 — Contextual Risk   (ALREADY IMPLEMENTED)
        # context_score is call-level, not chunk-level,
        # so it stays constant through the call.
        # NEW MVP: same is true of trajectory_result and
        # contradiction_result — computed once above from the
        # full transcript, constant through the call.
        # ----------------------------------------------------
        # NEW MVP: confidence is low if ASR produced no transcript at all
        # (e.g. the synthetic sine-wave demo clips) — there's nothing
        # reliable for context/trajectory/trust to have judged.
        text_confidence = 0.9 if transcript_text.strip() else 0.2

        # ----------------------------------------------------
        # Role 3 — Risk Fusion   (NEW MVP: now calls
        # fusion.fuse_scores_adaptive() — Feature 3. The ORIGINAL
        # fusion.fuse_scores() is untouched in fusion.py and still
        # works for anything else that calls it.)
        # ----------------------------------------------------
        signal_risks = {
            "spoof": spoof_score,
            "speaker": 1.0 - speaker_sim,
            "prosody": prosody_score,
            "context": context_score,
            "trajectory": trajectory_result.trajectory_risk,
            "trust": contradiction_result.contradiction_score,
        }
        signal_confidence = {
            "spoof": 1.0,
            "speaker": speaker_confidence,
            "prosody": 1.0,
            "context": text_confidence,
            "trajectory": text_confidence,
            "trust": text_confidence,
        }

        # NEW MVP (Feature 4): if a challenge response has already been
        # evaluated this session, fold it in as one more (optional) signal.
        if st.session_state.challenge_evaluation is not None:
            challenge_risk = challenge.challenge_result_as_risk_signal(
                st.session_state.challenge_evaluation
            )
            if challenge_risk is not None:
                signal_risks["challenge"] = challenge_risk
                signal_confidence["challenge"] = 1.0

        irs, fusion_breakdown = fusion.fuse_scores_adaptive(
            signal_risks,
            signal_confidence,
        )
        smoothed_irs = fusion.smooth(
            st.session_state.history["irs"][-5:] + [irs]
        )
        band, action = fusion.recommended_action(
            smoothed_irs
        )

        # ----------------------------------------------------
        # Save History   (ALREADY IMPLEMENTED — unchanged)
        # ----------------------------------------------------
        st.session_state.history["irs"].append(
            smoothed_irs
        )
        st.session_state.history["spoof"].append(
            spoof_score
        )
        st.session_state.history["speaker_sim"].append(
            speaker_sim
        )
        st.session_state.history["prosody"].append(
            prosody_score
        )
        st.session_state.history["context"].append(
            context_score
        )

        # ----------------------------------------------------
        # Update Dashboard
        # ----------------------------------------------------
        metric_irs.metric(
            "Impersonation Risk Score",
            f"{smoothed_irs:.1f} / 100"
        )
        metric_spoof.metric(
            "Spoof Score",
            f"{spoof_score:.2f}"
        )
        metric_speaker.metric(
            "Speaker Similarity",
            f"{speaker_sim:.2f}"
        )
        metric_prosody.metric(
            "Prosody Anomaly",
            f"{prosody_score:.2f}"
        )
        metric_context.metric(
            "Context Risk",
            f"{context_score:.2f}"
        )
        # NEW MVP metrics
        metric_trajectory.metric(
            "Conversation Trajectory Risk",
            f"{trajectory_result.trajectory_risk:.2f}"
        )
        metric_trust.metric(
            "Trust Contradiction",
            f"{contradiction_result.contradiction_score:.2f}"
        )
        metric_confidence_note.caption(
            "Signal confidence this chunk: "
            f"speaker={signal_confidence['speaker']:.1f}, "
            f"context/trajectory/trust={text_confidence:.1f}"
        )

        # ----------------------------------------------------
        # Risk Status   (ALREADY IMPLEMENTED — unchanged)
        # ----------------------------------------------------
        if band == "HIGH":
            status_box.error(
                f"⚠️ HIGH RISK — {action}"
            )
        elif band == "MEDIUM":
            status_box.warning(
                f"⚡ MEDIUM RISK — {action}"
            )
        else:
            status_box.success(
                f"✅ LOW RISK — {action}"
            )

        # ----------------------------------------------------
        # IRS Chart   (ALREADY IMPLEMENTED — unchanged)
        # ----------------------------------------------------
        chart_box.line_chart(
            {
                "IRS": st.session_state.history["irs"]
            }
        )

        # ----------------------------------------------------
        # Speaker Drift Detection   (ALREADY IMPLEMENTED — unchanged)
        # ----------------------------------------------------
        if len(
            st.session_state.history["speaker_sim"]
        ) >= 10:
            if sv.detect_drift(
                st.session_state.history["speaker_sim"]
            ):
                st.toast(
                    "⚠️ Speaker drift detected mid-call."
                )

        # Small delay so the dashboard visibly animates.
        time.sleep(0.08)

    # ========================================================
    # FINAL BREAKDOWN   (ALREADY IMPLEMENTED, extended with NEW MVP fields)
    # ========================================================
    breakdown_box.write(
        "### Final factor breakdown (last chunk)"
    )
    breakdown_box.json(
        {
            "spoof_score": round(spoof_score, 3),
            "speaker_similarity": round(speaker_sim, 3),
            "prosody_anomaly_score": round(prosody_score, 3),
            "context_risk_score": round(context_score, 3),
            "trajectory_risk_score": round(trajectory_result.trajectory_risk, 3),
            "trust_contradiction_score": round(contradiction_result.contradiction_score, 3),
            "fused_IRS": round(smoothed_irs, 1),
            "risk_band": band,
            "recommended_action": action,
            "adaptive_fusion_weights": fusion_breakdown,
        }
    )

    status_box.success(
        f"Done — processed {len(chunks)} chunks. "
        f"Final IRS: {smoothed_irs:.1f} ({band})"
    )

    # ========================================================
    # NEW MVP — Feature 5: build this call's fingerprint now that we
    # know the final trajectory/trust results. Not saved automatically
    # (see Scam Fingerprint section below) — just held in session_state.
    # ========================================================
    st.session_state.last_fingerprint = fingerprint.build_fingerprint(
        claimed_role=contradiction_result.claimed_role,
        problem_detected="problem_issue" in trajectory_result.detected_stages,
        urgency_detected="urgency_pressure" in trajectory_result.detected_stages,
        credential_requested="credential_request" in trajectory_result.detected_stages,
        financial_action_requested="financial_action" in trajectory_result.detected_stages,
        trajectory_stage=trajectory_result.current_stage,
    )

st.divider()

# ============================================================
# NEW MVP — Feature 4: ACTIVE VOICE CHALLENGE
# ============================================================
st.header("Active Verification (new MVP)")
st.caption(
    "Shown once risk reaches MEDIUM or HIGH. This is an ADDITIONAL "
    "verification signal only — it does not prove the caller is human "
    "or confirm their identity. For this Streamlit prototype, the "
    "'response' is an uploaded/recorded file, not live telephony."
)

current_band = None
if st.session_state.history["irs"]:
    current_band, _ = fusion.recommended_action(st.session_state.history["irs"][-1])

if current_band and challenge.should_trigger_challenge(current_band):
    if st.session_state.challenge_phrase is None:
        st.session_state.challenge_phrase = challenge.generate_challenge_phrase()

    st.warning(
        f"🔐 Ask the caller to repeat this phrase: "
        f"**{st.session_state.challenge_phrase}**"
    )

    challenge_file = st.file_uploader(
        "Upload the challenge response recording",
        type=["wav", "mp3"],
        key="challenge_upload"
    )
    if challenge_file is not None and st.button("✅ Evaluate Challenge Response"):
        response_audio = load_wav_file(challenge_file)
        evaluation = challenge.evaluate_challenge_response(
            response_audio,
            SR,
            st.session_state.enrolled_embedding,
        )
        st.session_state.challenge_evaluation = evaluation
        st.info(evaluation.summary)
        st.caption(evaluation.disclaimer)
        st.caption(
            "Re-run the pipeline above to fold this result into the "
            "fused Impersonation Risk Score as an extra signal."
        )
else:
    st.caption("No active challenge — current risk is LOW (or pipeline not yet run).")

st.divider()

# ============================================================
# NEW MVP — Feature 5: SCAM FINGERPRINT
# ============================================================
st.header("Scam Fingerprint (new MVP)")
st.caption(
    "Local prototype only — fingerprints are stored in a local JSON file "
    "(scam_fingerprints.json), NOT a real large-scale scam database."
)

if st.session_state.last_fingerprint is None:
    st.caption("Run the pipeline above to generate this call's fingerprint.")
else:
    fp = st.session_state.last_fingerprint
    st.json({
        "claimed_role": fp.claimed_role,
        "problem_detected": fp.problem_detected,
        "urgency_detected": fp.urgency_detected,
        "credential_requested": fp.credential_requested,
        "financial_action_requested": fp.financial_action_requested,
        "trajectory_stage": fp.trajectory_stage,
    })

    matches = fingerprint.find_similar(fp, threshold=0.7)
    st.write(fingerprint.describe_matches(matches))

    if st.button("💾 Save this interaction's fingerprint"):
        saved_id = fingerprint.save_fingerprint(fp)
        st.success(f"Fingerprint saved (id: {saved_id}).")
