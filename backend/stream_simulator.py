import io
import time
import wave
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"

AUDIO_FILE = Path(
    "/Users/divyasri/Music/Music/Media.localized/"
    "Music/Unknown Artist/Unknown Album/test_voice_enrollment.wav"
)

CHUNK_SECONDS = 0.3


def enroll():
    print("🎙️ Enrolling reference voice...")

    with open(AUDIO_FILE, "rb") as audio:
        response = requests.post(
            f"{BASE_URL}/enroll",
            files={"audio": ("reference.wav", audio, "audio/wav")},
        )

    response.raise_for_status()
    data = response.json()

    print(f"Speaker ID : {data['speaker_id']}")
    print(f"Session ID : {data['session_id']}")

    return data["session_id"]


def stream_audio(session_id):
    print("\n🔴 Starting simulated live audio stream...")
    print("Press Ctrl+C to stop.\n")

    with wave.open(str(AUDIO_FILE), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        chunk_frames = int(sample_rate * CHUNK_SECONDS)
        chunk_number = 0

        while True:
            frames = wav.readframes(chunk_frames)

            if not frames:
                break

            chunk_number += 1

            buffer = io.BytesIO()

            with wave.open(buffer, "wb") as chunk_wav:
                chunk_wav.setnchannels(channels)
                chunk_wav.setsampwidth(sample_width)
                chunk_wav.setframerate(sample_rate)
                chunk_wav.writeframes(frames)

            buffer.seek(0)

            response = requests.post(
                f"{BASE_URL}/analyze",
                files={
                    "audio": (
                        f"chunk_{chunk_number:03d}.wav",
                        buffer,
                        "audio/wav",
                    )
                },
                data={
                    "session_id": session_id,
                },
            )

            response.raise_for_status()
            result = response.json()

            print(
                f"Chunk {chunk_number:02d} | "
                f"IRS={result['smoothed_irs']:5.1f} | "
                f"{result['risk_band']:<6} | "
                f"{result['recommended_action']}"
            )

            time.sleep(CHUNK_SECONDS)

    print("\n🟢 Stream finished.")


def main():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    session_id = enroll()

    time.sleep(1)

    stream_audio(session_id)


if __name__ == "__main__":
    main()