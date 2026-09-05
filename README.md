# Voice Integrity Verification — FULL PIPELINE DEMO

This shows **all four modules working together at once**, so the whole
team can see how everyone's piece fits before the real backend (Role 3)
and real ASR/NLP (Role 4) and real spoof model (Role 1) exist.

```
Spoof Detection (Role 1 placeholder)  ─┐
Speaker Verification (Role 2)         ─┼─► Fusion Engine ─► Risk Score ─► Recommended Action ─► Dashboard
Prosody Analysis (Role 2)             ─┤
Contextual Signals (Role 4 placeholder)┘
```

## Files
| File | Owner | Status |
|---|---|---|
| `speaker_verification.py` | Role 2 | Real implementation |
| `prosody.py` | Role 2 | Real implementation |
| `spoof_heuristic.py` | Role 1 | **Placeholder** — swap internals for the real model |
| `context_signals.py` | Role 4 | **Placeholder** — swap internals for real ASR + NLP |
| `fusion.py` | Role 3 | Simplified stand-in for their backend fusion logic |
| `rolling_buffer.py` | shared | Utility — see "Important gotcha" below |
| `demo_audio.py` | shared | Synthetic sample generator, for testing without real recordings |
| `full_demo.py` | — | The unified Streamlit dashboard, run this to see everything |
| `headless_pipeline_test.py` | — | Console version of the same pipeline, no browser needed |

## Setup
```bash
pip install -r requirements.txt
```
If you hit `ModuleNotFoundError: No module named 'pkg_resources'`:
```bash
pip install "setuptools<81"
```

## Run the console test first (fastest way to see it work)
```bash
python3 headless_pipeline_test.py
```
This runs three scenarios end-to-end and prints every chunk's scores:
1. **Genuine call, no risky context** — should land LOW risk
2. **Cloned/TTS-like voice + risky phrases + unknown number** — should land MEDIUM/HIGH
3. **Different speaker, neutral context** — should land between the two

It asserts the risk ordering is correct and will fail loudly if something's
off — good smoke test to run after anyone changes a module.

## Run the visual dashboard
```bash
streamlit run full_demo.py
```
Then in the browser:
1. **Step 1 — Enroll**: click the synthetic-sample button, or upload a real
   short voice recording.
2. **Step 2 — Contextual signals**: type a fake transcript line (try
   "please share your OTP now") and toggle the metadata checkboxes — see
   the context risk score react immediately.
3. **Step 3 — Run the pipeline**: pick one of the synthetic scenarios (or
   upload real/TTS audio) and click Run. Watch all four scores, the fused
   Impersonation Risk Score, and the recommended action update live, plus
   a rolling chart and a final JSON breakdown.

## Important gotcha we found (and fixed) while building this
Speaker verification needs **at least ~1.5-2 seconds** of audio to produce
a reliable voiceprint comparison — a single ~300ms capture chunk (fine for
spoof detection and prosody) mostly returns a meaningless neutral score for
speaker similarity. We fixed this with `rolling_buffer.py`: it accumulates
the last ~1.8s of audio and only calls `speaker_verification.verify()` on
that window, while spoof detection and prosody still run on individual
small chunks. **If Role 3 wires the real backend without this buffer,
speaker-drift detection won't work in production either** — flag this in
your integration sync.

## What's fake vs. real right now
- **Real:** speaker verification, prosody analysis, the fusion math, the
  rolling-buffer fix, the dashboard.
- **Placeholder (needs real teammate work before the actual demo):**
  - `spoof_heuristic.py` — replace with Role 1's trained classifier
  - `context_signals.py` — replace with Role 4's ASR + NLP pipeline
  - `fusion.py` — Role 3 may want to move this into their FastAPI backend
    instead of running it client-side in Streamlit
- **Synthetic test audio (`demo_audio.py`)** — sine-wave stand-ins, not
  real speech. Good enough to prove the pipeline reacts correctly and to
  rehearse the demo flow, but before presenting to judges, swap in:
  - a real short recording of a teammate's voice (genuine sample)
  - a free TTS tool's output of a similar sentence (cloned/synthetic sample)

## Calibration checklist before the live demo
1. Record 5-10 genuine clips + generate 5-10 TTS clips of similar sentences
2. Run `headless_pipeline_test.py`-style scenarios with real files (see
   `full_demo.py`'s "Upload a WAV/MP3 file" option) instead of synthetic ones
3. Check that genuine clips consistently land LOW and TTS clips land
   MEDIUM/HIGH; adjust thresholds in `prosody.py` / `spoof_heuristic.py` /
   `fusion.py` weights if not
4. Once Role 1 and Role 4's real modules are ready, swap the placeholder
   files in this folder for theirs (same function signatures, so nothing
   else changes)
