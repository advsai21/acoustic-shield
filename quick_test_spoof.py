"""
quick_test_spoof.py — Role 1's calibration script (README "Calibration
checklist before the live demo", steps 1-3).

Point this at a folder of real genuine clips and a folder of TTS/cloned
clips of similar sentences and it will print a spoof_score for each,
plus a pass/fail summary, so you can see quickly whether genuine clips
land LOW and TTS clips land MEDIUM/HIGH (and tune spoof_heuristic.py's
thresholds/label mapping if not).

Usage:
    python3 quick_test_spoof.py path/to/genuine_dir path/to/tts_dir

Each directory should contain a handful of short WAV/MP3 files
(mono or stereo, any sample rate — this script resamples to 16kHz).
"""

import sys
import glob
import os

import numpy as np
import librosa

from spoof_heuristic import get_spoof_score

SR = 16000


def load_clip(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def score_dir(path: str) -> list:
    files = sorted(
        glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.mp3"))
    )
    scores = []
    for f in files:
        wav = load_clip(f)
        score = get_spoof_score(wav, SR)
        scores.append(score)
        print(f"  {os.path.basename(f):40s} spoof_score = {score:.3f}")
    return scores


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    genuine_dir, tts_dir = sys.argv[1], sys.argv[2]

    print(f"\nGenuine clips ({genuine_dir}) — want these LOW:")
    genuine_scores = score_dir(genuine_dir)

    print(f"\nTTS/cloned clips ({tts_dir}) — want these MEDIUM/HIGH:")
    tts_scores = score_dir(tts_dir)

    if not genuine_scores or not tts_scores:
        print("\nNo clips found in one or both directories — check the paths.")
        sys.exit(1)

    genuine_mean = np.mean(genuine_scores)
    tts_mean = np.mean(tts_scores)

    print(f"\nGenuine mean spoof_score: {genuine_mean:.3f}")
    print(f"TTS mean spoof_score:     {tts_mean:.3f}")

    if tts_mean > genuine_mean:
        print("Separation looks right (TTS scoring higher than genuine).")
    else:
        print("WARNING: TTS clips are NOT scoring higher than genuine clips.")
        print("If using the pretrained model: check the label mapping printed")
        print("by spoof_heuristic.py (unrecognized label scheme warning).")
        print("If using the heuristic fallback: tune the thresholds in")
        print("_heuristic_score() in spoof_heuristic.py.")


if __name__ == "__main__":
    main()
