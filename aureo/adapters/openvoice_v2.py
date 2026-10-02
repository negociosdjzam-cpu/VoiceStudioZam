"""OpenVoice V2 = explicit base synthesizer + local tone-color converter."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..types import AudioChunk, EngineCapabilities, SynthesisRequest
from .audio import mono_chunk, write_source_wav
from .base import RuntimeAdapter
from .seed import runtime_device, seed_scope


@dataclass(frozen=True)
class OpenVoiceSource:
    audio: AudioChunk
    speaker_embedding: Any


@dataclass(frozen=True)
class OpenVoiceRuntime:
    converter: Any
    # MeloTTS or another caller-configured local TTS supplies the source voice
    # and its matching embedding. Never substitute OmniVoice implicitly.
    synthesize_source: Callable[[SynthesisRequest], OpenVoiceSource]


class OpenVoiceV2Adapter(RuntimeAdapter):
    engine_id = "openvoice-v2"
    display_name = "OpenVoice V2"
    capabilities = EngineCapabilities(voice_reference=True, requires_voice_reference=True, seed=True, variation=True)

    def _generate(self, request: SynthesisRequest) -> AudioChunk:
        reference = request.reference
        assert reference is not None
        with seed_scope(request.seed, device=runtime_device(self._runtime.converter)):
            source = self._runtime.synthesize_source(request)
            converter = self._runtime.converter
            target_embedding = converter.extract_se([str(reference.path)])
            with tempfile.TemporaryDirectory(prefix="aureo-openvoice-") as directory:
                source_path = Path(directory) / "source.wav"
                write_source_wav(source.audio, source_path)
                waveform = converter.convert(
                    audio_src_path=str(source_path), src_se=source.speaker_embedding,
                    tgt_se=target_embedding, output_path=None,
                    tau=0.3 if request.variation is None else 0.1 + 0.8 * request.variation,
                )
            return mono_chunk(waveform, converter.hps.data.sampling_rate)
