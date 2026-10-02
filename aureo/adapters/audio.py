"""Normalize SDK mono tensors/arrays without importing their frameworks."""
from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

from ..types import AudioChunk, AureoEngineError


def mono_chunk(waveform: Any, sample_rate: int) -> AudioChunk:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    if hasattr(waveform, "tolist"):
        waveform = waveform.tolist()
    if not isinstance(waveform, (list, tuple)):
        raise AureoEngineError("invalid_audio", "SDK output is not a mono waveform")
    if waveform and isinstance(waveform[0], (list, tuple)):
        if len(waveform) != 1:
            raise AureoEngineError("invalid_audio", "SDK output must contain exactly one channel")
        waveform = waveform[0]
    try:
        return AudioChunk(tuple(waveform), sample_rate)
    except (ValueError, TypeError):
        raise AureoEngineError("invalid_audio", "SDK output contains invalid audio") from None


def write_source_wav(chunk: AudioChunk, path: Path) -> None:
    """Internal OpenVoice conversion input, not an exported synthetic take."""
    pcm = b"".join(struct.pack("<h", round(max(-1, min(1, sample)) * 32767)) for sample in chunk.samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(chunk.sample_rate)
        output.writeframes(pcm)
