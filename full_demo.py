"""
full_demo.py — Voice Integrity Verification Framework: FULL PIPELINE DEMO

Shows every module working together at once, so the whole team can see
how their pieces combine:
  - Speaker Verification   (Role 2)
  - Prosody Analysis       (Role 2)
  - Spoof Detection        (Role 1 — placeholder heuristic for now)
  - Contextual Signals     (Role 4 — real ASR + NLP risk scoring)
  - Risk Fusion + Action    (Role 3 — simplified stand-in for their backend)
  - Dashboard               (Role 5 — simplified stand-in for their UI)

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


SR = 16000
CHUNK_SIZE = 4800  # 300ms at 16kHz


st.set_page_config(
    page_title="Voice Integrity — Full Pipeline Demo",
    layout="wide"
)

st.title("🛡️ Voice Integrity Verification Framework — Full Pipeline Demo")

st.caption(
    "All modules wired together: Spoof Detection · Speaker Verification · "
    "Prosody Analysis · Contextual Signals · Risk Fusion · Recommended Action"
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
# STEP 1 — ENROLLMENT
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
# STEP 2 — CONTEXTUAL / CALL METADATA
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


st.divider()


# ============================================================
# STEP 3 — CALL AUDIO + FULL PIPELINE
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
# ROLE 4 — ASR + CONTEXT ANALYSIS
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


# ============================================================
# RUN BUTTON
# ============================================================

run_btn = st.button(
    "▶️ Run Pipeline on This Audio"
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

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


    for chunk in chunks:

        # ----------------------------------------------------
        # Role 1 — Spoof Detection
        # ----------------------------------------------------

        spoof_score = spoof.get_spoof_score(
            chunk,
            SR
        )


        # ----------------------------------------------------
        # Role 2 — Prosody Analysis
        # ----------------------------------------------------

        prosody_score = (
            prosody.get_prosody_anomaly_score(
                chunk,
                SR
            )
        )


        # ----------------------------------------------------
        # Role 2 — Speaker Verification
        # ----------------------------------------------------

        speaker_buffer.push(chunk)


        if speaker_buffer.is_ready():

            last_speaker_sim = sv.verify(
                speaker_buffer.get(),
                SR,
                st.session_state.enrolled_embedding
            )


        speaker_sim = last_speaker_sim


        # ----------------------------------------------------
        # Role 4 — Contextual Risk
        # ----------------------------------------------------
        # context_score is call-level, not chunk-level,
        # so it stays constant through the call.


        # ----------------------------------------------------
        # Role 3 — Risk Fusion
        # ----------------------------------------------------

        irs = fusion.fuse_scores(
            spoof_score,
            speaker_sim,
            prosody_score,
            context_score
        )


        smoothed_irs = fusion.smooth(
            st.session_state.history["irs"][-5:] + [irs]
        )


        band, action = fusion.recommended_action(
            smoothed_irs
        )


        # ----------------------------------------------------
        # Save History
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


        # ----------------------------------------------------
        # Risk Status
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
        # IRS Chart
        # ----------------------------------------------------

        chart_box.line_chart(
            {
                "IRS": st.session_state.history["irs"]
            }
        )


        # ----------------------------------------------------
        # Speaker Drift Detection
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
    # FINAL BREAKDOWN
    # ========================================================

    breakdown_box.write(
        "### Final factor breakdown (last chunk)"
    )


    breakdown_box.json(
        {
            "spoof_score": round(
                spoof_score,
                3
            ),

            "speaker_similarity": round(
                speaker_sim,
                3
            ),

            "prosody_anomaly_score": round(
                prosody_score,
                3
            ),

            "context_risk_score": round(
                context_score,
                3
            ),

            "fused_IRS": round(
                smoothed_irs,
                1
            ),

            "risk_band": band,

            "recommended_action": action
        }
    )


    status_box.success(
        f"Done — processed {len(chunks)} chunks. "
        f"Final IRS: {smoothed_irs:.1f} ({band})"
    )