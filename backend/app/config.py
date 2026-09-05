"""
app/config.py

Loads backend/config.yaml once and exposes it as a validated, typed
settings object. Nothing else in the app should read config.yaml
directly or hardcode thresholds/weights — always go through
get_settings().
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Dict

import yaml

# backend/config.yaml lives one directory above this file (backend/app/config.py -> backend/)
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass
class AudioSettings:
    target_sample_rate: int = 16000
    window_seconds: float = 2.0
    min_window_seconds: float = 1.0
    max_window_seconds: float = 3.0
    denoise: bool = False
    speaker_buffer_seconds: float = 1.8


@dataclass
class FusionWeights:
    spoof: float = 0.40
    speaker: float = 0.30
    prosody: float = 0.15
    context: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "spoof": self.spoof,
            "speaker": self.speaker,
            "prosody": self.prosody,
            "context": self.context,
        }


@dataclass
class FusionSettings:
    weights: FusionWeights = field(default_factory=FusionWeights)
    ema_alpha: float = 0.40


@dataclass
class RiskBandSettings:
    low_max: float = 30
    medium_max: float = 65


@dataclass
class StorageSettings:
    database_url: str = "sqlite:///./voice_integrity.db"


@dataclass
class Settings:
    audio: AudioSettings
    fusion: FusionSettings
    risk_bands: RiskBandSettings
    actions: Dict[str, str]
    storage: StorageSettings
    log_level: str = "INFO"


class ConfigError(ValueError):
    """Raised when config.yaml is missing required/valid values."""


def _validate_weights(weights: FusionWeights) -> None:
    total = weights.spoof + weights.speaker + weights.prosody + weights.context
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            f"fusion.weights must sum to 1.0, got {total} "
            f"(spoof={weights.spoof}, speaker={weights.speaker}, "
            f"prosody={weights.prosody}, context={weights.context})"
        )


def _validate_risk_bands(bands: RiskBandSettings) -> None:
    if not (0 <= bands.low_max < bands.medium_max <= 100):
        raise ConfigError(
            f"risk_bands must satisfy 0 <= low_max < medium_max <= 100, "
            f"got low_max={bands.low_max}, medium_max={bands.medium_max}"
        )


@functools.lru_cache(maxsize=1)
def get_settings(config_path: str = _CONFIG_PATH) -> Settings:
    """
    Load, validate, and cache config.yaml. Cached with lru_cache so the
    file is only parsed once per process; tests that need a different
    config should call get_settings.cache_clear() first.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    audio_raw = raw.get("audio", {})
    audio = AudioSettings(
        target_sample_rate=int(audio_raw.get("target_sample_rate", 16000)),
        window_seconds=float(audio_raw.get("window_seconds", 2.0)),
        min_window_seconds=float(audio_raw.get("min_window_seconds", 1.0)),
        max_window_seconds=float(audio_raw.get("max_window_seconds", 3.0)),
        denoise=bool(audio_raw.get("denoise", False)),
        speaker_buffer_seconds=float(audio_raw.get("speaker_buffer_seconds", 1.8)),
    )

    weights_raw = raw.get("fusion", {}).get("weights", {})
    weights = FusionWeights(
        spoof=float(weights_raw.get("spoof", 0.40)),
        speaker=float(weights_raw.get("speaker", 0.30)),
        prosody=float(weights_raw.get("prosody", 0.15)),
        context=float(weights_raw.get("context", 0.15)),
    )
    _validate_weights(weights)

    fusion = FusionSettings(
        weights=weights,
        ema_alpha=float(raw.get("fusion", {}).get("ema_alpha", 0.40)),
    )

    bands_raw = raw.get("risk_bands", {})
    risk_bands = RiskBandSettings(
        low_max=float(bands_raw.get("low_max", 30)),
        medium_max=float(bands_raw.get("medium_max", 65)),
    )
    _validate_risk_bands(risk_bands)

    actions = dict(
        raw.get(
            "actions",
            {
                "LOW": "ALLOW",
                "MEDIUM": "CALLBACK_VERIFICATION",
                "HIGH": "REQUIRE_MFA_OR_ESCALATE",
            },
        )
    )

    storage = StorageSettings(
        database_url=raw.get("storage", {}).get("database_url", "sqlite:///./voice_integrity.db")
    )

    log_level = raw.get("logging", {}).get("level", "INFO")

    return Settings(
        audio=audio,
        fusion=fusion,
        risk_bands=risk_bands,
        actions=actions,
        storage=storage,
        log_level=log_level,
    )
