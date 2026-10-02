"""Qwen3TTSModel Base voice cloning. Its batch return is not real streaming."""
from __future__ import annotations

from ..types import AudioChunk, AureoEngineError, EngineCapabilities, QualityMode, SynthesisRequest
from .audio import mono_chunk
from .base import RuntimeAdapter
from .seed import runtime_device, seed_scope


class Qwen3TTSAdapter(RuntimeAdapter):
    engine_id = "qwen3-tts"
    display_name = "Qwen3-TTS"
    _LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}
    capabilities = EngineCapabilities(voice_reference=True, requires_voice_reference=True, seed=True, variation=True, languages=tuple(_LANGUAGES))
    _PRESETS = {
        QualityMode.FAST: {"top_k": 20, "top_p": 0.8},
        QualityMode.PRO: {"top_k": 50, "top_p": 0.95},
        QualityMode.ULTRA: {"top_k": 80, "top_p": 0.97},
    }

    def _generate(self, request: SynthesisRequest) -> AudioChunk:
        options = dict(self._PRESETS[request.mode])
        if request.variation is not None:
            options["temperature"] = 0.1 + 0.9 * request.variation
        reference = request.reference
        assert reference is not None  # enforced before model loading
        with seed_scope(request.seed, device=runtime_device(self._runtime)):
            waveforms, sample_rate = self._runtime.generate_voice_clone(
                text=request.text, language=self._LANGUAGES.get(request.language, "Auto"),
                ref_audio=str(reference.path), ref_text=reference.transcript,
                x_vector_only_mode=not bool(reference.transcript and reference.transcript.strip()),
                non_streaming_mode=True, **options,
            )
            if len(waveforms) != 1:
                raise AureoEngineError("invalid_audio", "Expected exactly one synthesized Qwen waveform")
            return mono_chunk(waveforms[0], sample_rate)
