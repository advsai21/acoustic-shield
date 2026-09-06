"""
Role 4 - Automatic Speech Recognition (ASR)

Uses Faster-Whisper to convert call audio into text.
"""

from pathlib import Path
from typing import Union

import numpy as np


# Keep the model lazy-loaded so importing this module
# does not immediately download/load the Whisper model.
_MODEL = None
_MODEL_SIZE = None


def _get_model(model_size: str = "base"):
    """
    Load the Whisper model only when it is actually needed.
    """

    global _MODEL, _MODEL_SIZE

    if _MODEL is not None and _MODEL_SIZE == model_size:
        return _MODEL

    from faster_whisper import WhisperModel

    print(f"Loading Faster-Whisper model: {model_size}")

    _MODEL = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    _MODEL_SIZE = model_size

    return _MODEL


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio into mono float32 format.
    """

    audio = np.asarray(audio)

    # Convert stereo/multi-channel audio to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert integer PCM values to float range [-1, 1].
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return audio


def _resample_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """
    Resample audio to the sample rate expected by Whisper.
    """

    if original_sample_rate == target_sample_rate:
        return audio

    if len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / original_sample_rate

    new_length = int(duration * target_sample_rate)

    if new_length <= 0:
        return np.array([], dtype=np.float32)

    old_times = np.linspace(
        0,
        duration,
        num=len(audio),
        endpoint=False,
    )

    new_times = np.linspace(
        0,
        duration,
        num=new_length,
        endpoint=False,
    )

    resampled = np.interp(
        new_times,
        old_times,
        audio,
    )

    return resampled.astype(np.float32)


def transcribe_audio(
    audio: Union[str, Path, np.ndarray],
    sample_rate: int = 16000,
    model_size: str = "base",
    language: str = "en",
) -> str:
    """
    Transcribe audio into text.

    Parameters
    ----------
    audio:
        Either:
        - path to an audio file
        - numpy audio array

    sample_rate:
        Sample rate of numpy audio arrays.

    model_size:
        Faster-Whisper model size.

    language:
        Expected language.

    Returns
    -------
    str
        Combined transcript.
    """

    model = _get_model(model_size)

    if isinstance(audio, (str, Path)):
        audio_input = str(audio)

    else:
        audio_input = _to_float32_mono(audio)

        audio_input = _resample_audio(
            audio_input,
            sample_rate,
            16000,
        )

    segments, _ = model.transcribe(
        audio_input,
        language=language,
        vad_filter=True,
    )

    transcript_parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            transcript_parts.append(text)

    return " ".join(transcript_parts).strip()