"""Dependency-free contracts for the parallel AUREO engine lab."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path


class QualityMode(str, Enum):
    FAST = "FAST"
    PRO = "PRO"
    ULTRA = "ULTRA"


class AureoEngineError(RuntimeError):
    """A stable error code and a safe, public explanation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def valid_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise ValueError("Invalid engine or profile identifier")
    return value


def unit_interval(value: float | None, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")


@dataclass(frozen=True)
class VoiceReference:
    path: Path
    transcript: str | None = None

    def __post_init__(self) -> None:
        if "://" in str(self.path):
            raise ValueError("Voice references must be local files")
        object.__setattr__(self, "path", Path(self.path))
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise ValueError("Reference transcript must be text")


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    reference: VoiceReference | None = None
    mode: QualityMode = QualityMode.PRO
    language: str | None = None
    seed: int | None = None
    variation: float | None = None
    energy: float | None = None
    expression: str | None = None
    streaming: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Synthesis text must not be empty")
        object.__setattr__(self, "mode", QualityMode(self.mode))
        if self.reference is not None and not isinstance(self.reference, VoiceReference):
            raise ValueError("Invalid voice reference")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1):
            raise ValueError("Seed must be an integer between 0 and 2**32 - 1")
        unit_interval(self.variation, "variation")
        unit_interval(self.energy, "energy")
        for name in ("language", "expression"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty text")
        if type(self.streaming) is not bool:
            raise ValueError("streaming must be a boolean")


@dataclass(frozen=True)
class EngineCapabilities:
    voice_reference: bool = False
    requires_voice_reference: bool = False
    seed: bool = False
    variation: bool = False
    energy: bool = False
    expression: bool = False
    streaming: bool = False
    languages: tuple[str, ...] = ()  # empty means the plugin validates languages


@dataclass(frozen=True)
class AudioChunk:
    """Internal mono PCM samples; not a file/export or a transport response."""
    samples: tuple[float, ...]
    sample_rate: int

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("Audio sample rate must be a positive integer")
        values = tuple(self.samples)
        if not values or any(
            isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
            for x in values
        ):
            raise ValueError("Audio must contain finite mono PCM samples")
        object.__setattr__(self, "samples", tuple(float(x) for x in values))

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate
