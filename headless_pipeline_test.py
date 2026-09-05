"""
headless_pipeline_test.py — runs the EXACT same pipeline logic as
full_demo.py (enrollment -> chunking -> all 4 scores -> fusion -> action)
but prints to console instead of Streamlit. Use this to sanity-check the
whole system without needing a browser, or to debug quickly.
"""

import numpy as np
import speaker_verification as sv
import prosody
import spoof_heuristic as spoof
import context_signals as ctx
import fusion
import demo_audio
from rolling_buffer import RollingAudioBuffer

SR = 16000
CHUNK_SIZE = 4800


def run_scenario(name, call_audio, enrolled_embedding, transcript="", call_metadata=None):
    print(f"\n{'='*60}\nSCENARIO: {name}\n{'='*60}")
    context_score, matched = ctx.get_context_risk_score(transcript, call_metadata or {})
    print(f"Context risk score: {context_score:.2f}  (matched phrases: {matched})")

    chunks = demo_audio.to_int16_chunks(call_audio, CHUNK_SIZE)
    irs_history = []
    speaker_buffer = RollingAudioBuffer(sr=SR, window_seconds=1.8)
    last_speaker_sim = 0.5  # neutral until the buffer fills

    for i, chunk in enumerate(chunks):
        spoof_score = spoof.get_spoof_score(chunk, SR)
        prosody_score = prosody.get_prosody_anomaly_score(chunk, SR)

        # Speaker verification needs more context than one small chunk —
        # feed it a rolling ~1.8s window instead (see speaker_verification.py note).
        speaker_buffer.push(chunk)
        if speaker_buffer.is_ready():
            last_speaker_sim = sv.verify(speaker_buffer.get(), SR, enrolled_embedding)
        speaker_sim = last_speaker_sim

        irs = fusion.fuse_scores(spoof_score, speaker_sim, prosody_score, context_score)
        smoothed = fusion.smooth(irs_history[-5:] + [irs])
        band, action = fusion.recommended_action(smoothed)
        irs_history.append(smoothed)

        print(f"  chunk {i:2d}: spoof={spoof_score:.2f} speaker_sim={speaker_sim:.2f} "
              f"prosody={prosody_score:.2f} context={context_score:.2f} "
              f"-> IRS={smoothed:5.1f} [{band}]")

    print(f"\n  FINAL: IRS={irs_history[-1]:.1f}  band={band}  action='{action}'")
    return irs_history[-1]


if __name__ == "__main__":
    print("Enrolling reference speaker (synthetic genuine sample)...")
    enrolled = sv.enroll(demo_audio.genuine_sample(seed=1), SR)

    final_genuine = run_scenario(
        "Genuine call, no risky context",
        demo_audio.genuine_sample(seed=2),
        enrolled,
        transcript="Hi, just calling to confirm our meeting tomorrow.",
        call_metadata={"unknown_number": False, "odd_call_time": False, "first_contact": False},
    )

    final_spoofed = run_scenario(
        "Cloned/TTS-like voice + risky phrases + unknown number",
        demo_audio.synthetic_sample(),
        enrolled,
        transcript="This is urgent, please share your OTP now or your account will be suspended.",
        call_metadata={"unknown_number": True, "odd_call_time": True, "first_contact": True},
    )

    final_diff_speaker = run_scenario(
        "Different speaker, neutral context",
        demo_audio.different_speaker_sample(),
        enrolled,
        transcript="Hey, it's me, how's it going?",
        call_metadata={"unknown_number": False, "odd_call_time": False, "first_contact": False},
    )

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Genuine call final IRS:        {final_genuine:5.1f}")
    print(f"Cloned+risky-context final IRS: {final_spoofed:5.1f}")
    print(f"Different speaker final IRS:    {final_diff_speaker:5.1f}")

    assert final_spoofed > final_genuine, "Expected the cloned/risky scenario to score higher than genuine"
    assert final_diff_speaker > final_genuine, "Expected a different speaker to score higher than genuine"
    print("\nPASS: risk ordering behaves as expected across all three scenarios.")
