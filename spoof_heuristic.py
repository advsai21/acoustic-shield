"""
spoof_heuristic.py
Role 1 deliverable — Spoof / Synthetic Voice Detection (SRS FR-3).

Classifies a short audio chunk as bona fide vs. synthetic/spoofed and
returns a confidence score in [0, 1] (higher = more likely synthetic/
cloned/replayed).

Model approach (given hackathon time constraints):
    We do NOT train a spoof classifier from scratch. Instead this module
    wraps a pretrained wav2vec2-style anti-spoofing/deepfake-audio
    classifier pulled from the HuggingFace Hub via `transformers`
    (loaded once, CPU-friendly, no fine-tuning required — "used
    pretrained as-is" per SRS FR-3 / the architecture doc's model note).

    IMPORTANT — verify the checkpoint yourself before the demo: I can't
    browse the HuggingFace Hub from here, so `_HF_MODEL_ID` below is my
    best recollection of a public wav2vec2 bona-fide/spoof classifier,
    not a guarantee it still exists at that exact path or that its label
    scheme matches what `_extract_spoof_prob()` expects. Budget 15-20 min
    early on to:
      1. Search the HF Hub for "audio deepfake detection" / "anti
         spoofing" / "ASVspoof wav2vec2" and pick a model card that (a)
         loads via `pipeline("audio-classification", model=<id>)` and
         (b) states its label names.
      2. Set that id in `_HF_MODEL_ID` (or the SPOOF_MODEL_ID env var —
         no code change needed).
      3. Run this file directly (`python3 spoof_heuristic.py`) — it
         prints the raw label list on first real inference so you can
         confirm `_extract_spoof_prob()` is reading the right label.
    If nothing suitable turns up in time, the module still works: see
    "Graceful degradation" below.

Graceful degradation: if `transformers`/`torch` aren't installed, the
Hub is unreachable (offline demo, flaky venue wifi), or the checkpoint
fails to load for any reason, this module silently falls back to the
original signal-processing heuristic (pitch/centroid/MFCC-variance
based) so the rest of the pipeline never crashes waiting on Role 1's
model. This mirrors the NFR-6 "graceful degradation" requirement for
the pipeline as a whole.

Contract this module preserves (do not change the signature —
fusion.py / rolling_buffer.py / full_demo.py / headless_pipeline_test.py
all call this directly):
    def get_spoof_score(audio_chunk: np.ndarray, sr: int) -> float
    returns a value in [0, 1], higher = more likely synthetic/spoofed.
"""

import os
import numpy as np
import librosa

# transformers + torch are heavier than the rest of this repo's deps
# (see requirements.txt) — kept optional so teammates who only need the
# dashboard/other modules aren't forced to install them. See
# requirements-role1.txt for what to add to actually run the real model.
try:
    from transformers import pipeline as _hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

TARGET_SR = 16000

# Best-effort pretrained checkpoint — see the module docstring above for
# why this needs a quick manual sanity check before the live demo.
# Override without touching code: `export SPOOF_MODEL_ID=<other-hf-id>`
_HF_MODEL_ID = os.environ.get("SPOOF_MODEL_ID", "MelodyMachine/Deepfake-audio-detection-V2")

# Substrings used to figure out which output label means "spoof" and
# which means "bona fide" without hardcoding one model's exact label
# strings (different anti-spoofing checkpoints name them differently:
# "spoof"/"bonafide", "fake"/"real", "synthetic"/"genuine", etc.).
_SPOOF_LABEL_HINTS = ("spoof", "fake", "synthetic", "clone", "generated")
_BONAFIDE_LABEL_HINTS = ("bona", "real", "genuine", "human")

_classifier = None          # lazily loaded, cached across calls
_model_load_attempted = False
_warned_unrecognized_labels = False


def get_spoof_score(audio_chunk: np.ndarray, sr: int) -> float:
    """
    Score one audio chunk for likelihood of being synthetic/spoofed.

    Args:
        audio_chunk: 1D waveform (int16 or float32)
        sr: sample rate of `audio_chunk`

    Returns:
        spoof_score in [0, 1] — higher means more likely AI-generated/
        cloned/replayed speech; lower means more likely bona fide human
        speech. Returns a neutral 0.3 for chunks too short/silent to
        judge (consistent with the other modules' near-silence handling).
    """
    wav = _to_float32(audio_chunk)

    if len(wav) < sr * 0.2:
        return 0.3  # too short/near-silent to say anything meaningful

    clf = _get_classifier()
    if clf is not None:
        try:
            return _score_with_model(clf, wav, sr)
        except Exception as e:
            # Don't let one bad chunk (or a mid-demo model hiccup) take
            # down the whole pipeline — fall back for this chunk only.
            print(f"[spoof_heuristic] model inference failed ({e}); "
                  f"falling back to heuristic for this chunk")

    return _heuristic_score(wav, sr)


def _get_classifier():
    """Lazily load and cache the HF pipeline; remembers a failed load so
    we don't retry (and re-print warnings) on every single chunk."""
    global _classifier, _model_load_attempted

    if _classifier is not None:
        return _classifier
    if _model_load_attempted:
        return None
    _model_load_attempted = True

    if not _TRANSFORMERS_AVAILABLE:
        print("[spoof_heuristic] transformers/torch not installed — "
              "using signal-processing heuristic instead. "
              "See requirements-role1.txt to enable the real model.")
        return None

    try:
        _classifier = _hf_pipeline(
            "audio-classification",
            model=_HF_MODEL_ID,
            device=-1,  # CPU — don't assume a demo laptop has a GPU
        )
        print(f"[spoof_heuristic] loaded pretrained model: {_HF_MODEL_ID}")
    except Exception as e:
        print(f"[spoof_heuristic] could not load '{_HF_MODEL_ID}' ({e}) — "
              f"using signal-processing heuristic instead. Check the "
              f"model id in _HF_MODEL_ID / SPOOF_MODEL_ID, and your "
              f"internet connection (first load downloads the checkpoint).")
        _classifier = None

    return _classifier


def _score_with_model(clf, wav: np.ndarray, sr: int) -> float:
    """Run the pretrained classifier on one chunk and reduce its output
    to a single spoof_score in [0, 1]."""
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    results = clf({"array": wav, "sampling_rate": TARGET_SR}, top_k=None)
    return _extract_spoof_prob(results)


def _extract_spoof_prob(results: list) -> float:
    """
    `results` looks like [{"label": "...", "score": 0.91}, ...].
    Map whichever label scheme the checkpoint uses onto "probability
    this is spoofed", using substring hints rather than exact label
    strings (see _SPOOF_LABEL_HINTS / _BONAFIDE_LABEL_HINTS above).
    """
    global _warned_unrecognized_labels

    for r in results:
        label = str(r["label"]).lower()
        if any(hint in label for hint in _SPOOF_LABEL_HINTS):
            return float(np.clip(r["score"], 0.0, 1.0))

    for r in results:
        label = str(r["label"]).lower()
        if any(hint in label for hint in _BONAFIDE_LABEL_HINTS):
            return float(np.clip(1.0 - r["score"], 0.0, 1.0))

    # Label scheme didn't match either hint list — don't silently guess
    # wrong on every chunk. Warn once with the raw labels so you can add
    # a hint (or hardcode the mapping) in under a minute, and return a
    # neutral score in the meantime rather than crashing the pipeline.
    if not _warned_unrecognized_labels:
        raw_labels = [r["label"] for r in results]
        print(f"[spoof_heuristic] unrecognized label scheme {raw_labels} — "
              f"add a matching hint to _SPOOF_LABEL_HINTS/_BONAFIDE_LABEL_HINTS "
              f"in spoof_heuristic.py. Returning neutral 0.5 until fixed.")
        _warned_unrecognized_labels = True
    return 0.5


def _heuristic_score(wav: np.ndarray, sr: int) -> float:
    """
    Signal-processing fallback (no ML dependency): flags flat pitch,
    an atypical spectral centroid, and low MFCC variance — all common,
    if crude, symptoms of TTS/vocoder output. Used automatically when
    the pretrained model can't be loaded (see get_spoof_score above).
    Same heuristic as the original placeholder — kept so behavior
    doesn't regress for anyone still running without the ML deps.
    """
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    mfcc_var = np.var(mfccs)

    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr))

    pitches, _ = librosa.piptrack(y=wav, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    score = 0.0
    if pitch_std < 5.0:
        score += 0.35
    elif pitch_std < 15.0:
        score += 0.15

    if spec_centroid < 1000 or spec_centroid > 4000:
        score += 0.30

    if mfcc_var < 50.0:
        score += 0.35

    return float(np.clip(score, 0.05, 0.99))


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


if __name__ == "__main__":
    # Quick manual smoke test using the repo's existing synthetic stand-ins
    # (see demo_audio.py). This only sanity-checks that the module runs
    # end-to-end (model load or fallback) — see quick_test_spoof.py
    # (suggested below) for a real test against actual genuine/TTS clips,
    # which is what actually matters before the demo.
    import demo_audio

    sr = demo_audio.SR
    genuine = demo_audio.genuine_sample(duration=2.0, sr=sr)
    spoofed = demo_audio.synthetic_sample(duration=2.0, sr=sr)

    print(f"genuine-like sample  -> spoof_score = {get_spoof_score(genuine, sr):.3f}")
    print(f"synthetic-like sample -> spoof_score = {get_spoof_score(spoofed, sr):.3f}")
    print("(expect the second number noticeably higher than the first)")
