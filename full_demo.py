"""
full_demo.py — Voice Integrity Verification Framework: FULL PIPELINE DEMO

Shows every module working together at once, so the whole team can see
how their pieces combine:
  - Speaker Verification   (Role 2)
  - Prosody Analysis        (Role 2)
  - Spoof Detection         (Role 1 — placeholder heuristic for now)
  - Contextual Signals      (Role 4 — placeholder keyword/metadata rules)
  - Risk Fusion + Action    (Role 3 — simplified stand-in for their backend)
  - Dashboard               (Role 5 — simplified stand-in for their UI)

TWO WAYS TO RUN THIS:
  1. Built-in synthetic samples (no mic, no files needed) — good for a
     first end-to-end look and for testing before real audio is ready.
  2. Upload your own WAV files (a genuine sample + a "spoofed" sample,
     e.g. from a free TTS tool) for a more convincing/realistic demo.

Run with: streamlit run full_demo.py
"""

import time
import numpy as np
import streamlit as st
import librosa

import speaker_verification as sv
import prosody
import spoof_heuristic as spoof
import context_signals as ctx
import fusion
import demo_audio
from rolling_buffer import RollingAudioBuffer

SR = 16000
CHUNK_SIZE = 4800  # 300ms at 16kHz

st.set_page_config(page_title="Voice Integrity — Full Pipeline Demo", layout="wide")
st.title("🛡️ Voice Integrity Verification Framework — Full Pipeline Demo")
st.caption(
    "All modules wired together: Spoof Detection · Speaker Verification · "
    "Prosody Analysis · Contextual Signals · Risk Fusion · Recommended Action"
)

# --- Session state ---
if "enrolled_embedding" not in st.session_state:
    st.session_state.enrolled_embedding = None
if "history" not in st.session_state:
    st.session_state.history = {"irs": [], "spoof": [], "speaker_sim": [], "prosody": [], "context": []}


def load_wav_file(uploaded_file):
    """Load an uploaded WAV/MP3 file as a float32 mono waveform at SR."""
    wav, _ = librosa.load(uploaded_file, sr=SR, mono=True)
    return wav.astype(np.float32)


def reset_history():
    st.session_state.history = {"irs": [], "spoof": [], "speaker_sim": [], "prosody": [], "context": []}


# ============================================================
# STEP 1 — ENROLLMENT
# ============================================================
st.header("Step 1 — Enroll Reference Voice")

enroll_source = st.radio(
    "Enrollment audio source",
    ["Use built-in synthetic sample (quick test)", "Upload a real WAV/MP3 file"],
    horizontal=True,
)

enroll_audio = None
if enroll_source == "Use built-in synthetic sample (quick test)":
    if st.button("🎙️ Enroll using synthetic 'genuine' sample"):
        enroll_audio = demo_audio.genuine_sample(seed=1)
else:
    enroll_file = st.file_uploader("Upload reference voice sample", type=["wav", "mp3"], key="enroll_upload")
    if enroll_file is not None and st.button("🎙️ Enroll from uploaded file"):
        enroll_audio = load_wav_file(enroll_file)

if enroll_audio is not None:
    st.session_state.enrolled_embedding = sv.enroll(enroll_audio, SR)
    reset_history()
    st.success(f"Reference voiceprint enrolled ({len(enroll_audio)/SR:.1f}s sample).")

if st.session_state.enrolled_embedding is None:
    st.warning("Enroll a reference speaker above before running the pipeline below.")
    st.stop()

st.divider()

# ============================================================
# STEP 2 — CONTEXTUAL / CALL METADATA (Role 4 placeholder)
# ============================================================
st.header("Step 2 — Contextual Signals (simulated)")
st.caption("In the real system this comes from ASR transcript + call metadata (Role 4). Here, enter it manually for the demo.")

ctx_col1, ctx_col2 = st.columns([2, 1])
with ctx_col1:
    transcript_text = st.text_input(
        "Simulated live transcript (try: 'please share your OTP now')",
        value=""
    )
with ctx_col2:
    unknown_number = st.checkbox("Unknown number")
    odd_call_time = st.checkbox("Odd call time")
    first_contact = st.checkbox("First contact")

call_metadata = {
    "unknown_number": unknown_number,
    "odd_call_time": odd_call_time,
    "first_contact": first_contact,
}
context_score, matched_phrases = ctx.get_context_risk_score(transcript_text, call_metadata)
st.write(f"**Context risk score:** {context_score:.2f}" + (f" — matched: {matched_phrases}" if matched_phrases else ""))

st.divider()

# ============================================================
# STEP 3 — CALL AUDIO + LIVE PIPELINE
# ============================================================
st.header("Step 3 — Run the Full Pipeline on Call Audio")

audio_source = st.radio(
    "Call audio source",
    ["Synthetic: genuine-like", "Synthetic: cloned/TTS-like", "Synthetic: different speaker", "Upload a WAV/MP3 file"],
    horizontal=True,
)

call_audio = None
if audio_source == "Synthetic: genuine-like":
    call_audio = demo_audio.genuine_sample(seed=2)
elif audio_source == "Synthetic: cloned/TTS-like":
    call_audio = demo_audio.synthetic_sample()
elif audio_source == "Synthetic: different speaker":
    call_audio = demo_audio.different_speaker_sample()
else:
    call_file = st.file_uploader("Upload call audio", type=["wav", "mp3"], key="call_upload")
    if call_file is not None:
        call_audio = load_wav_file(call_file)

run_btn = st.button("▶️ Run Pipeline on This Audio")

# --- Live dashboard placeholders ---
col1, col2, col3, col4, col5 = st.columns(5)
metric_irs = col1.metric("Impersonation Risk Score", "—")
metric_spoof = col2.metric("Spoof Score", "—")
metric_speaker = col3.metric("Speaker Similarity", "—")
metric_prosody = col4.metric("Prosody Anomaly", "—")
metric_context = col5.metric("Context Risk", "—")

status_box = st.empty()
chart_box = st.empty()
breakdown_box = st.empty()

if run_btn and call_audio is not None:
    reset_history()
    chunks = demo_audio.to_int16_chunks(call_audio, CHUNK_SIZE)
    status_box.info(f"Processing {len(chunks)} chunks (~{len(call_audio)/SR:.1f}s of audio)...")

    # Speaker verification needs more context than a single ~300ms chunk to
    # be reliable (tested: 1.8s window gives a real gap between same-speaker
    # and different-speaker audio, whereas 300ms mostly hits the neutral
    # fallback). Spoof and prosody modules work fine on the smaller chunks.
    speaker_buffer = RollingAudioBuffer(sr=SR, window_seconds=1.8)
    last_speaker_sim = 0.5

    for chunk in chunks:
        spoof_score = spoof.get_spoof_score(chunk, SR)
        prosody_score = prosody.get_prosody_anomaly_score(chunk, SR)

        speaker_buffer.push(chunk)
        if speaker_buffer.is_ready():
            last_speaker_sim = sv.verify(speaker_buffer.get(), SR, st.session_state.enrolled_embedding)
        speaker_sim = last_speaker_sim
        # context_score is call-level, not chunk-level, so it stays constant through the call

        irs = fusion.fuse_scores(spoof_score, speaker_sim, prosody_score, context_score)
        smoothed_irs = fusion.smooth(st.session_state.history["irs"][-5:] + [irs])
        band, action = fusion.recommended_action(smoothed_irs)

        st.session_state.history["irs"].append(smoothed_irs)
        st.session_state.history["spoof"].append(spoof_score)
        st.session_state.history["speaker_sim"].append(speaker_sim)
        st.session_state.history["prosody"].append(prosody_score)
        st.session_state.history["context"].append(context_score)

        metric_irs.metric("Impersonation Risk Score", f"{smoothed_irs:.1f} / 100")
        metric_spoof.metric("Spoof Score", f"{spoof_score:.2f}")
        metric_speaker.metric("Speaker Similarity", f"{speaker_sim:.2f}")
        metric_prosody.metric("Prosody Anomaly", f"{prosody_score:.2f}")
        metric_context.metric("Context Risk", f"{context_score:.2f}")

        if band == "HIGH":
            status_box.error(f"⚠️ HIGH RISK — {action}")
        elif band == "MEDIUM":
            status_box.warning(f"⚡ MEDIUM RISK — {action}")
        else:
            status_box.success(f"✅ LOW RISK — {action}")

        chart_box.line_chart({"IRS": st.session_state.history["irs"]})

        if len(st.session_state.history["speaker_sim"]) >= 10:
            if sv.detect_drift(st.session_state.history["speaker_sim"]):
                st.toast("⚠️ Speaker drift detected mid-call.")

        time.sleep(0.08)  # small delay so the dashboard visibly animates

    breakdown_box.write("### Final factor breakdown (last chunk)")
    breakdown_box.json({
        "spoof_score": round(spoof_score, 3),
        "speaker_similarity": round(speaker_sim, 3),
        "prosody_anomaly_score": round(prosody_score, 3),
        "context_risk_score": round(context_score, 3),
        "fused_IRS": round(smoothed_irs, 1),
        "risk_band": band,
        "recommended_action": action,
    })
    status_box.success(f"Done — processed {len(chunks)} chunks. Final IRS: {smoothed_irs:.1f} ({band})")
